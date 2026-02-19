import os
import re
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from config.constants import (
    COLLECTION_NAME,
    HISTORY_MAX_TURNS,
    MIN_RELEVANCE_SCORE,
    PERSIST_DIRECTORY,
    RETRIEVAL_TOP_K,
    REWRITE_HISTORY_TURNS,
    STRICT_MIN_RELEVANCE_SCORE,
)
from preprocess.language_detector import detect_language
from vector_store.embedder import get_embedder

load_dotenv()


class MenstrualHealthRAG:
    def __init__(self):
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
        self.embedder = get_embedder()
        self.vectordb_cache: Dict[str, Chroma] = {}
        self.chat_history: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=HISTORY_MAX_TURNS)
        )
        self.employee_profiles = self._load_employee_profiles()

        self.rewrite_prompt = PromptTemplate.from_template(
            """You are a query rewriter for conversational retrieval.

Conversation history:
{history}

Follow-up user question:
{question}

Rewrite the follow-up as a single standalone search query.
Rules:
- Keep original meaning.
- Resolve references like "it", "that", "this", "he", "she", "they".
- If the question is already standalone, return it unchanged.
- Return only the rewritten query."""
        )

        self.answer_prompt = PromptTemplate.from_template(
            """You are an HR assistant chatbot for Acme AI Ltd.
You do not have a personal name. And You do not talk about anything without the context of company policies and employee information. If you don't have enough information to answer, say you don't know politely.

User profile:
{user_profile}
(This is only this user's profile. Do not assume anything about other users.)

Conversation history:
{history}

Retrieved context:
{context}

Current user question:
{question}

Instructions:
- Answer using retrieved context first.
- If context is insufficient, then use conversation history and employee profile if relevant.
- If still unknown, say you do not have enough information in a helpful way.
- Keep answer under 120 words.
- Always Respond in English. Never change your language to Bangla or any other language, even if the question is in that language. Just answer in English.
- Do not restate the user's question.
-Do not mention any references. 
- Also don't say "Based on the provided information" or similar phrases. Just provide the answer directly in a polite manner.
- If you think the user query is not relevant to the topic, respond soo shortly and politely that you don't know without trying to answer.
- Also in case of sensitive action, such as, complaining against ceo or manager, first provide a consiquences of the action to the user then help him."""
        )

    def _load_employee_profiles(self) -> Dict[str, str]:
        csv_path = os.path.join("data", "employee_table.csv")
        if not os.path.exists(csv_path):
            return {}

        employees = pd.read_csv(csv_path)
        profile_map: Dict[str, str] = {}
        for _, row in employees.iterrows():
            employee_id = str(row.get("employeeId", "")).strip()
            if not employee_id:
                continue
            profile_map[employee_id] = (
                f"{row.get('Employee Name', 'Unknown')}. "
                f"My gender is {row.get('Gender', 'Unknown')}. "
                f"I have {row.get('Earned Leave', 0)} Earned leaves, "
                f"{row.get('Casual Leave', 0)} Casual Leaves, "
                f"{row.get('Sick Leave', 0)} Sick Leaves, "
                f"{row.get('Maternity Leave', 0)} Maternity Leaves, "
                f"{row.get('Without Pay Leave', 0)} Unpaid Leaves and "
                f"{row.get('Adjustment Leave', 0)} Adjustment Leaves remaining."
            )
        return profile_map

    def _normalize_language(self, language: str) -> str:
        lang = language.lower().strip()
        return lang if lang in {"english", "bangla"} else "english"

    def _get_vectordb(self, language: str) -> Chroma:
        lang = self._normalize_language(language)
        if lang not in self.vectordb_cache:
            persist_dir = f"{PERSIST_DIRECTORY}/{COLLECTION_NAME}_{lang}"
            if not os.path.exists(persist_dir):
                raise ValueError(
                    f"Vector database for language '{lang}' not found at {persist_dir}. "
                    "Please build it first using /build-vectordb."
                )
            self.vectordb_cache[lang] = Chroma(
                embedding_function=self.embedder,
                collection_name=f"{lang}_chunks",
                persist_directory=persist_dir,
            )
        return self.vectordb_cache[lang]

    def _format_history(self, turns: List[Tuple[str, str]], max_turns: int) -> str:
        scoped = turns[-max_turns:]
        if not scoped:
            return "No previous conversation."
        lines = []
        for q, a in scoped:
            lines.append(f"User: {q}")
            lines.append(f"Assistant: {a}")
        return "\n".join(lines)

    def _rewrite_query(self, question: str, turns: List[Tuple[str, str]]) -> str:
        if not turns:
            return question
        history = self._format_history(turns, REWRITE_HISTORY_TURNS)
        chain = self.rewrite_prompt | self.llm
        rewritten = chain.invoke({"history": history, "question": question}).content.strip()
        return rewritten or question

    def _retrieve_context(self, language: str, query: str):
        vectordb = self._get_vectordb(language)
        docs_with_scores = vectordb.similarity_search_with_relevance_scores(
            query, k=RETRIEVAL_TOP_K
        )
        if not docs_with_scores:
            return [], ""

        filtered = [
            (doc, score)
            for doc, score in docs_with_scores
            if score is None or score >= MIN_RELEVANCE_SCORE
        ]
        selected = filtered if filtered else docs_with_scores

        blocks = []
        for idx, (doc, score) in enumerate(selected, 1):
            source = doc.metadata.get("source", "unknown")
            score_str = "n/a" if score is None else f"{score:.3f}"
            blocks.append(
                f"[{idx}] source={source} score={score_str}\n{doc.page_content.strip()}"
            )
        return selected, "\n\n".join(blocks)

    def _extract_query_terms(self, query: str) -> List[str]:
        terms = re.findall(r"[A-Za-z0-9]+", query.lower())
        return [t for t in terms if len(t) >= 3]

    def _has_term_match_in_docs(self, query: str, source_docs) -> bool:
        terms = self._extract_query_terms(query)
        if not terms or not source_docs:
            return False
        texts = " ".join((doc.page_content or "").lower() for doc, _ in source_docs)
        return any(term in texts for term in terms)

    def clear_history(self, user_id: str) -> None:
        self.chat_history.pop(user_id, None)

    def invalidate_vectordb(self, language: str) -> None:
        lang = self._normalize_language(language)
        self.vectordb_cache.pop(lang, None)

    def get_response(self, query: str, user_id: str) -> dict:
        turns = list(self.chat_history[user_id])
        language = self._normalize_language(detect_language(query))
        standalone_query = self._rewrite_query(query, turns)

        try:
            source_docs, context = self._retrieve_context(language, standalone_query)
        except ValueError:
            return {
                "query": query,
                "resolved_query": standalone_query,
                "language": language,
                "answer": f"Knowledge base for '{language}' is not built yet.",
                "sources": [],
            }

        best_score = max((score or 0.0) for _, score in source_docs) if source_docs else 0.0
        has_term_match = self._has_term_match_in_docs(standalone_query, source_docs)
        if not source_docs or (best_score < STRICT_MIN_RELEVANCE_SCORE and not has_term_match):
            answer = (
                "I do not have enough relevant company context to answer that. "
                "Please ask about company policies or employee information."
            )
            self.chat_history[user_id].append((query, answer))
            return {
                "query": query,
                "resolved_query": standalone_query,
                "language": language,
                "answer": answer,
                "sources": [],
            }

        history_text = self._format_history(turns, HISTORY_MAX_TURNS)
        user_profile = self.employee_profiles.get(user_id, "No employee profile found.")

        answer_chain = self.answer_prompt | self.llm
        answer = answer_chain.invoke(
            {
                "user_profile": user_profile,
                "history": history_text,
                "context": context or "No retrieved context.",
                "question": query,
            }
        ).content.strip()

        self.chat_history[user_id].append((query, answer))

        sources = []
        for doc, score in source_docs:
            sources.append(
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "score": score,
                    "chunk_id": doc.metadata.get("chunk_id"),
                }
            )

        return {
            "query": query,
            "resolved_query": standalone_query,
            "language": language,
            "answer": answer,
            "sources": sources,
        }
