# Agentic RAG Pipeline

A self-correcting Retrieval-Augmented Generation (RAG) system built with **LangGraph** that goes beyond basic RAG by actively evaluating the quality of retrieved content. When retrieved chunks are deemed irrelevant, the pipeline automatically rewrites the query and retries up to 3 times before admitting it cannot find a confident answer — eliminating hallucinations and ensuring every response is grounded in the source document.

---

## How It Works

```
START
  │
  ▼
retrieve ──► grade
              │
        relevant?
        ┌─────┴──────┐
       YES            NO
        ▼             ▼
     generate    attempts < 3?
        │         ┌────┴────┐
       END       YES        NO
                  ▼          ▼
               rewrite    give_up
                  │          │
                  ▼         END
               retrieve
```

1. **`retrieve`** — fetches top document chunks from ChromaDB using the current question
2. **`grade`** — uses Gemini to evaluate whether the retrieved chunks are actually relevant to the question (`yes` / `no`)
3. **`route_after_grade`** — conditional edge: if relevant → generate answer; if not and retries < 3 → rewrite; if retries exhausted → give up
4. **`rewrite`** — rewrites the original question to be clearer and more specific, increments the attempt counter
5. **`generate`** — synthesises a final answer from the relevant chunks
6. **`give_up`** — returns an honest non-answer instead of hallucinating when no relevant content is found after 3 attempts

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | Google Gemini 2.5 Flash via LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Document Loading | LangChain PyPDFLoader |
| Config | python-dotenv |

---

## Setup & Running

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd agentic-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. Run the pipeline

```bash
python agentic_rag.py
```

The pipeline will load the PDF, build the vector store, run the agentic RAG graph, and print the final answer.

---

## Project Structure

```
agentic-rag/
├── agentic_rag.py                        # LangGraph pipeline
├── AAAI-2025-PresPanel-Report-FINAL.pdf  # Source document
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key |
