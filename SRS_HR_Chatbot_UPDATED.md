# Software Requirements Specification (SRS)
## HR Chatbot Backend (RAG)

**Document Version:** 2026-03-11
**Project:** probahini-backend
**Prepared by:** Codex (update draft)

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for the HR Chatbot Backend. The system provides a Retrieval-Augmented Generation (RAG) API that answers HR policy questions and returns employee leave and attendance information for Acme AI Ltd. It also supports building and refreshing the vector database from HR policy documents.

### 1.2 Scope
The backend exposes REST APIs for:
- Building a vector database from HR policy documents (English and Bangla).
- Chatting with an HR assistant that answers based on company context and user profile.
- Retrieving leave balances for a given employee.
- Retrieving attendance summaries and current-day entries for a given employee.
- Resetting per-user chat history.

Out of scope:
- Authentication and authorization.
- Frontend UI.
- Direct ERP integrations (only tutorial steps are provided).

### 1.3 Definitions, Acronyms, and Abbreviations
- **RAG:** Retrieval-Augmented Generation.
- **LLM:** Large Language Model.
- **Chroma:** Local vector store used for retrieval.
- **Ollama:** Local LLM server used for inference.
- **HR:** Human Resources.

### 1.4 References
- Source code: `e:\probahini-backend`.

### 1.5 Overview
Section 2 describes the product context and constraints. Section 3 lists system features. Section 4 defines external interfaces. Section 5 lists non-functional requirements.

---

## 2. Overall Description

### 2.1 Product Perspective
The system is a Python/FastAPI backend that performs document ingestion, vector indexing, retrieval, and LLM-based response generation. It uses a local Chroma vector store persisted on disk and an Ollama server for LLM inference.

### 2.2 Product Functions
- Build vector database from policy documents in `data/raw/english` and `data/raw/bangla`.
- Answer HR policy questions using retrieved context and user profile context.
- Return leave balances from `data/employee_table.csv`.
- Return attendance summaries from `data/attendance.csv`.
- Provide an ERP leave application tutorial from `data/tutorials.json` when the query indicates ERP leave application intent.
- Reset per-user chat history.

### 2.3 User Classes and Characteristics
- **Employees:** Ask HR questions, view leave balance, and get attendance information.
- **HR/Admin:** Maintain policy documents and employee data; rebuild vector database.
- **Integrators/Developers:** Integrate API endpoints into a frontend or other systems.

### 2.4 Operating Environment
- Python 3.x
- FastAPI + Uvicorn
- LangChain
- Chroma vector database (local file persistence)
- Ollama server running locally (default `http://localhost:11434`)
- File system with CSV/JSON/document inputs

### 2.5 Design and Implementation Constraints
- Requires local access to policy documents and employee data files.
- Uses local inference via Ollama (no cloud LLM dependency is required).
- Vector database is persisted on disk under `db_Policy/`.
- Language support limited to English and Bangla.

### 2.6 Assumptions and Dependencies
- Ollama is installed and running with the configured model.
- Policy documents exist as `.txt` files under `data/raw/{english,bangla}`.
- Employee and attendance CSV files are available and formatted as expected.

---

## 3. System Features

### 3.1 Build Vector Database
**Description:** Ingests `.txt` documents, splits into chunks, generates embeddings, and persists a Chroma vector store for the chosen language.

**Endpoint:** `POST /build-vectordb`

**Inputs:**
- `language` (string): `english` or `bangla`.

**Processing:**
- Reads `.txt` files from `data/raw/{language}`.
- Splits text into chunks using `RecursiveCharacterTextSplitter`.
- Creates embeddings and stores vectors in Chroma under `db_Policy/Information_chunks_{language}`.
- Clears cached vector database for that language.

**Outputs:**
- `message`, `chunks_stored`, `path`.

**Error Conditions:**
- Unsupported language.
- Missing data directory or no `.txt` files.

### 3.2 Chat with HR Assistant
**Description:** Answers questions using a RAG pipeline with policy context and user profile context.

**Endpoint:** `POST /chat`

**Inputs:**
- `user_id` (string)
- `question` (string)

**Processing:**
- Detects language (English/Bangla).
- Rewrites follow-up questions to standalone queries (English path).
- Translates Bangla queries to English for retrieval.
- Retrieves relevant chunks from Chroma.
- Generates answer using Ollama with prompt constraints.
- Maintains per-user conversation history (max 8 turns).

**Outputs:**
- `query`, `resolved_query`, `language`, `answer`, `sources`.

**Special Behavior:**
- If query indicates ERP leave application request, returns the ERP tutorial steps from `data/tutorials.json`.
- If context is insufficient, responds with a constrained fallback message.

### 3.3 Get Leave Balances
**Description:** Returns leave balances from `data/employee_table.csv`.

**Endpoint:** `POST /leaves`

**Inputs:**
- `user_id` (string)

**Outputs:**
- Leave balance fields (`earned_leave`, `casual_leave`, `sick_leave`, `maternity_leave`, `without_pay_leave`, `adjustment_leave`) and gender.

### 3.4 Get Attendance Summary
**Description:** Returns attendance summary for the current day and a list of daily first entries for the current month.

**Endpoint:** `POST /attendance`

**Inputs:**
- `user_id` (string)

**Outputs:**
- `USERID`, `first_entry`, `last_entry`, `previous_record`.

### 3.5 Reset Chat History
**Description:** Clears stored chat history for a user.

**Endpoint:** `POST /chat/reset`

**Inputs:**
- `user_id` (string)

**Outputs:**
- Confirmation message.

---

## 4. External Interface Requirements

### 4.1 API Interface
- Base URL: `http://<host>:<port>` (default FastAPI port `8000`).
- JSON request/response format.

### 4.2 Data Interfaces
- **Policy documents:** `.txt` files in `data/raw/english` and `data/raw/bangla`.
- **Employee data:** `data/employee_table.csv`.
- **Attendance data:** `data/attendance.csv`.
- **Tutorial data:** `data/tutorials.json`.

### 4.3 Software Interfaces
- **Ollama:** Local server for LLM inference (`OLLAMA_BASE_URL`).
- **Chroma:** Local vector store for retrieval (persisted on disk).

### 4.4 Communication Interfaces
- HTTP/JSON over REST.

---

## 5. Non-Functional Requirements

### 5.1 Performance
- Chat response time should be acceptable for interactive use; retrieval is limited to top-k results (default 5).
- Vector database build time depends on document size and embedding model.

### 5.2 Reliability and Availability
- Service should return clear error messages for missing data or invalid inputs.
- Vector database cache is invalidated when rebuilding.

### 5.3 Security
- The system currently does not implement authentication or authorization.
- Employee data is accessed by `user_id` only; proper access controls are required for production.

### 5.4 Maintainability
- Modular code structure for preprocessing, vector store, and inference.
- Configuration via environment variables and constants.

### 5.5 Portability
- Runs on standard Python environments with the listed dependencies.

---

## 6. Data Requirements

### 6.1 Employee Table (`data/employee_table.csv`)
Expected columns include:
- `employeeId`, `Employee Name`, `Gender`, `Earned Leave`, `Casual Leave`, `Sick Leave`, `Maternity Leave`, `Without Pay Leave`, `Adjustment Leave`.

### 6.2 Attendance (`data/attendance.csv`)
Expected columns include:
- `USERID`, `Date`, `Time`.

### 6.3 Policy Documents
- `.txt` documents only are ingested by the vector builder.
- `.docx` or other formats should be converted to `.txt` before ingestion.

---

## 7. Future Enhancements
- Authentication and role-based access control.
- Attendance regularization tutorial integration via endpoint.
- Automated document ingestion from `.docx` and `.pdf` without manual conversion.
- Observability and structured logging.

---

## 8. Appendix
- Default embedding models are configured via environment variables:
  - `EMBED_MODEL_ENGLISH`
  - `EMBED_MODEL_BANGLA`
- Default LLM model is configured via `OLLAMA_MODEL`.
