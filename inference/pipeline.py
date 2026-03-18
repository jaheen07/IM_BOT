import os
import re
import json
from datetime import date
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
    STRICT_MIN_RELEVANCE_SCORE_BANGLA,
)
from preprocess.language_detector import detect_language
from vector_store.embedder import get_embedder

load_dotenv()


class RAGBOT:
    def __init__(self):
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
        self.embedders: Dict[str, object] = {}
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
        self.translation_prompt = PromptTemplate.from_template(
            """You are a translator for search queries.
Translate the following Bangla query into concise natural English for semantic retrieval.
Return only the translated query text. Do not add explanations.

Bangla query:
{question}"""
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
- Do not restate the user's question.
-Do not mention any references. 
- Also don't say "Based on the provided information" or similar phrases. Just provide the answer directly in a polite manner.
- If you think the user query is not relevant to the topic, respond soo shortly and politely that you don't know without trying to answer.
- Also in case of sensitive action, such as, complaining against ceo or manager, first provide a consiquences of the action to the user then help him."""
        )
        self.answer_prompt_bangla = PromptTemplate.from_template(
            """You are an HR assistant chatbot for Acme AI Ltd.
You must answer ONLY from Retrieved context and User profile below.
Never use outside/world knowledge, even if you know the answer.

User profile:
{user_profile}

Conversation history:
{history}

Retrieved context:
{context}

Current user question:
{question}

Instructions:
- If the answer is not explicitly supported by Retrieved context or User profile, reply exactly:
  I do not have enough relevant company context to answer that. Please ask about company policies or employee information.
- Do not infer facts that are not in the provided context.
- Keep answer under 120 words.
- Do not mention references or sources.
- Respond in English."""
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
    

    def _load_leaves(self) -> Dict[str, str]:
        csv_path = os.path.join("data", "employee_table.csv")
        if not os.path.exists(csv_path):
            return {}

        employees = pd.read_csv(csv_path)
        profile_map: Dict[str, str] = {}
        for _, row in employees.iterrows():
            employee_id = str(row.get("employeeId", "")).strip()
            if not employee_id:
                continue
            profile_map[employee_id] = {
            "gender": row.get("Gender", "Male"),
            "earned_leave": row.get("Earned Leave", 0),
            "casual_leave": row.get("Casual Leave", 0),
            "sick_leave": row.get("Sick Leave", 0),
            "maternity_leave": row.get("Maternity Leave", 0),
            "without_pay_leave": row.get("Without Pay Leave", 0),
            "adjustment_leave": row.get("Adjustment Leave", 0),
            }
        return profile_map
    
    def _load_attendance(self) -> Dict[str, Dict]:
        csv_path = os.path.join("data", "employee_table.csv")
        attendance_path = os.path.join("data", "attendance.csv")

        if not os.path.exists(csv_path) or not os.path.exists(attendance_path):
            return {}

        employees = pd.read_csv(csv_path)
        attendance = pd.read_csv(attendance_path)

        def _normalize_id(value) -> str:
            if pd.isna(value):
                return ""
            text = str(value).strip()
            if not text or text.lower() == "nan":
                return ""
            if text.endswith(".0") and text[:-2].isdigit():
                return text[:-2]
            return text

        attendance["USERID_CLEAN"] = attendance["USERID"].apply(_normalize_id)

        profile_map: Dict[str, Dict] = {}
        today = date.today()
        current_month = today.month
        current_year = today.year

        for _, row in employees.iterrows():
            employee_id = _normalize_id(row.get("employeeId", ""))
            if not employee_id:
                continue

            # Most rows have empty USERID in employee_table, so fall back to employeeId.
            user_id = _normalize_id(row.get("USERID", "")) or employee_id

            # Filter attendance records for this user
            user_attendance = attendance[attendance["USERID_CLEAN"] == user_id].copy()
            if user_attendance.empty:
                profile_map[employee_id] = {
                    "USERID": user_id,
                    "first_entry": None,
                    "last_entry": None,
                    "previous_record": {},
                }
                continue
        
            # Combine date and time, sort chronologically
            user_attendance["datetime"] = pd.to_datetime(
                user_attendance["Date"].astype(str).str.strip()
                + " "
                + user_attendance["Time"].astype(str).str.strip(),
                format="%m/%d/%Y %I:%M %p",
                errors="coerce",
            )
            user_attendance = user_attendance.dropna(subset=["datetime"]).sort_values("datetime")
            if user_attendance.empty:
                profile_map[employee_id] = {
                    "USERID": user_id,
                    "first_entry": None,
                    "last_entry": None,
                    "previous_record": {},
                }
                continue

            # Build current-month daily first entry list for this user.
            monthly_attendance = user_attendance[
                (user_attendance["datetime"].dt.year == current_year)
                & (user_attendance["datetime"].dt.month == current_month)
            ].copy()
            previous_record: Dict[str, Dict[str, str]] = {}
            if not monthly_attendance.empty:
                monthly_attendance["entry_date"] = monthly_attendance["datetime"].dt.date
                daily_first_entries = (
                    monthly_attendance.sort_values("datetime")
                    .groupby("entry_date", as_index=False)
                    .first()
                    .sort_values("entry_date")
                )
                for idx, (_, day_row) in enumerate(daily_first_entries.iterrows(), start=1):
                    previous_record[f"day_{idx}"] = {
                        "date": str(day_row["Date"]),
                        "time": str(day_row["Time"]),
                        "datetime": day_row["datetime"].isoformat(),
                    }

            # Consider only attendance records from today for this user.
            today_attendance = user_attendance[user_attendance["datetime"].dt.date == today]
            if today_attendance.empty:
                profile_map[employee_id] = {
                    "USERID": user_id,
                    "first_entry": None,
                    "last_entry": None,
                    "previous_record": previous_record,
                }
                continue

            today_attendance = today_attendance.sort_values("datetime")
            first_record = today_attendance.iloc[0]
            last_record = today_attendance.iloc[-1] if len(today_attendance) > 1 else None

            profile_map[employee_id] = {
                "USERID": user_id,
                "first_entry": {
                    "date": str(first_record["Date"]),
                    "time": str(first_record["Time"]),
                    "datetime": first_record["datetime"].isoformat()
                },
                "last_entry": (
                    {
                        "date": str(last_record["Date"]),
                        "time": str(last_record["Time"]),
                        "datetime": last_record["datetime"].isoformat()
                    }
                    if last_record is not None
                    else None
                ),
                "previous_record": previous_record,
            }
        return profile_map

    def _normalize_language(self, language: str) -> str:
        lang = language.lower().strip()
        return lang if lang in {"english", "bangla"} else "english"

    def _get_embedder(self, language: str):
        lang = self._normalize_language(language)
        if lang not in self.embedders:
            self.embedders[lang] = get_embedder(lang)
        return self.embedders[lang]

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
                embedding_function=self._get_embedder(lang),
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

    def _translate_bangla_query_to_english(self, query: str) -> str:
        normalized_query = (
            query.replace("\u098f\u0995\u09ae\u09bf", "Acme")
            .replace("\u0985\u09cd\u09af\u09be\u0995\u09ae\u09c7", "Acme")
            .replace("\u098f\u0986\u0987", "AI")
            .replace("\u09b2\u09bf\u09ae\u09bf\u099f\u09c7\u09a1", "Ltd")
        )
        try:
            chain = self.translation_prompt | self.llm
            translated = chain.invoke({"question": normalized_query}).content.strip()
        except Exception:
            return normalized_query
        if not translated:
            return normalized_query
        return translated.strip().strip('"').strip("'")

    def _normalize_bangla_query_for_retrieval(self, query: str) -> str:
        normalized = query
        replacements = {
            "একমি": "অ্যাকমে",
            "একমে": "অ্যাকমে",
            "এআই": "AI",
            "লিমিটেড": "Ltd",
        }
        for src, dst in replacements.items():
            normalized = normalized.replace(src, dst)

        lower_q = normalized.lower()
        has_company_hint = any(
            token in lower_q
            for token in ("অ্যাকমে", "acme", "ai", "ltd", "কোম্পানি", "প্রোফাইল")
        )
        if has_company_hint and "acme ai ltd" not in lower_q:
            normalized = f"{normalized} অ্যাকমে এআই লিমিটেড Acme AI Ltd কোম্পানি প্রোফাইল"
        return normalized

    def _distance_to_similarity(self, distance: float) -> float:
        safe_distance = max(float(distance or 0.0), 0.0)
        return 1.0 / (1.0 + safe_distance)

    def _retrieve_context_bangla(self, vectordb: Chroma, query: str):
        normalized_query = self._normalize_bangla_query_for_retrieval(query)
        query_candidates = [query]
        if normalized_query != query:
            query_candidates.append(normalized_query)

        merged: Dict[str, Dict[str, object]] = {}
        for candidate in query_candidates:
            docs_with_distances = vectordb.similarity_search_with_score(
                candidate, k=max(RETRIEVAL_TOP_K * 4, 12)
            )
            for doc, distance in docs_with_distances:
                source = doc.metadata.get("source", "unknown")
                chunk_id = doc.metadata.get("chunk_id", "na")
                key = f"{source}::{chunk_id}"
                similarity = self._distance_to_similarity(distance)
                if key not in merged or similarity > float(merged[key]["similarity"]):
                    merged[key] = {
                        "doc": doc,
                        "similarity": similarity,
                    }

        if not merged:
            return [], ""

        bangla_terms = self._extract_query_terms(normalized_query, "bangla")
        unique_terms = list(dict.fromkeys(bangla_terms))

        reranked = []
        for item in merged.values():
            doc = item["doc"]
            similarity = float(item["similarity"])
            text = (doc.page_content or "").lower()
            match_count = sum(1 for term in unique_terms if term in text)
            rerank_score = similarity + min(match_count, 6) * 0.03
            reranked.append((rerank_score, match_count, similarity, doc))

        reranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)

        selected = [(doc, similarity) for _, _, similarity, doc in reranked[:RETRIEVAL_TOP_K]]
        filtered = [(doc, score) for doc, score in selected if score >= MIN_RELEVANCE_SCORE]
        selected = filtered if filtered else selected

        blocks = []
        for idx, (doc, score) in enumerate(selected, 1):
            source = doc.metadata.get("source", "unknown")
            blocks.append(
                f"[{idx}] source={source} score={score:.3f}\n{doc.page_content.strip()}"
            )
        return selected, "\n\n".join(blocks)

    def _retrieve_context(self, language: str, query: str):
        lang = self._normalize_language(language)
        vectordb = self._get_vectordb(language)
        if lang == "bangla":
            return self._retrieve_context_bangla(vectordb, query)

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

    def _extract_query_terms(self, query: str, language: str) -> List[str]:
        lang = self._normalize_language(language)
        if lang == "bangla":
            # Include Bangla script tokens plus any mixed Latin terms in Bangla queries.
            terms = re.findall(r"[\u0980-\u09FF]+|[A-Za-z0-9]+", query.lower())
            # Keep only meaningful-length terms; filters out many Bangla particles.
            filtered = [t for t in terms if len(t) >= 3]
            return filtered

        # Keep English behavior exactly the same as before.
        terms = re.findall(r"[A-Za-z0-9]+", query.lower())
        return [t for t in terms if len(t) >= 3]

    def _has_term_match_in_docs(self, query: str, source_docs, language: str) -> bool:
        lang = self._normalize_language(language)
        query_for_match = (
            self._normalize_bangla_query_for_retrieval(query) if lang == "bangla" else query
        )
        terms = self._extract_query_terms(query_for_match, language)
        if not terms or not source_docs:
            return False
        texts = " ".join((doc.page_content or "").lower() for doc, _ in source_docs)
        matched_terms = {term for term in terms if term in texts}
        if lang == "bangla":
            # Bangla path is stricter to avoid passing weak/irrelevant context.
            return len(matched_terms) >= 2
        return len(matched_terms) >= 1

    def clear_history(self, user_id: str) -> None:
        self.chat_history.pop(user_id, None)

    def invalidate_vectordb(self, language: str) -> None:
        lang = self._normalize_language(language)
        self.vectordb_cache.pop(lang, None)

    def apply_erp(self) -> Dict:
        tutorials_path = os.path.join("data", "tutorials.json")
        if not os.path.exists(tutorials_path):
            return {}
        with open(tutorials_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("leave_application", {})

    def _is_apply_erp_query(self, query: str, resolved_query: str) -> bool:
        text = f"{query} {resolved_query}".lower()
        has_erp = "erp" in text
        has_leave = any(token in text for token in ("ERP","leave", "ইআরপি","ইআরপিতে" "ছুটির আবেদন", "লিভ", "ছুটি", "লিভ আবেদন", "ছুটির জন্য"))
        has_apply_intent = any(
            token in text
            for token in (
                "apply",
                "application",
                "request",
                "take",
                "how to",
                "ERP",
                "আবেদন",
                "কীভাবে",
                "কিভাবে",
                "করব",
                "করবো",
                "নিতে",
                "নেও",
                "নিব",
                "নিবো",
                "করতে হয়",
                "করতে হবে",
            )
        )
        return has_erp and has_leave and has_apply_intent

    def get_response(self, query: str, user_id: str) -> dict:
        turns = list(self.chat_history[user_id])
        language = self._normalize_language(detect_language(query))
        retrieval_language = language
        if language == "bangla":
            # Bangla path: translate query to English, then retrieve from English KB.
            retrieval_language = "english"
            standalone_query = self._translate_bangla_query_to_english(query)
        else:
            standalone_query = self._rewrite_query(query, turns)

        if self._is_apply_erp_query(query, standalone_query):
            return {
                "query": query,
                "resolved_query": standalone_query,
                "language": language,
                "answer": "Here is the ERP leave application process based on company policy:",
                "sources": [],
                "tutorial": self.apply_erp(),
            }

        try:
            source_docs, context = self._retrieve_context(retrieval_language, standalone_query)
        except ValueError:
            return {
                "query": query,
                "resolved_query": standalone_query,
                "language": language,
                "answer": f"Knowledge base for '{language}' is not built yet.",
                "sources": [],
            }

        best_score = max((score or 0.0) for _, score in source_docs) if source_docs else 0.0
        has_term_match = self._has_term_match_in_docs(
            standalone_query, source_docs, retrieval_language
        )
        strict_threshold = (
            STRICT_MIN_RELEVANCE_SCORE_BANGLA
            if retrieval_language == "bangla"
            else STRICT_MIN_RELEVANCE_SCORE
        )
        should_block = False
        if retrieval_language == "bangla":
            # Bangla requires both decent retrieval score and lexical evidence in docs.
            should_block = (
                not source_docs or best_score < strict_threshold or not has_term_match
            )
        else:
            # Keep existing English behavior unchanged.
            should_block = not source_docs or (
                best_score < strict_threshold and not has_term_match
            )

        if should_block:
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
        history_for_prompt = (
            history_text if language == "english" else "Ignore history and use retrieved context only."
        )

        answer_prompt = (
            self.answer_prompt_bangla if language == "bangla" else self.answer_prompt
        )
        answer_chain = answer_prompt | self.llm
        answer = answer_chain.invoke(
            {
                "user_profile": user_profile,
                "history": history_for_prompt,
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
