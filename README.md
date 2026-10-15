# AgenticInsight 🧠⚡

> **A local-first, privacy-preserving agentic assistant for structured data and unstructured documents.**  
> Built for Apple Silicon (8GB M1 budget) with zero external cloud API dependencies, zero Docker overhead, and strict read-only execution safety.

---

## 🏗 Architecture Diagram

```mermaid
flowchart TD
    User["User Browser / Client"] -->|HTTP / SSE Stream| UI["Frontend (React 18 + TS + Tailwind + Recharts)"]
    UI -->|POST /api/chat| FastApiApp["FastAPI Backend Server (Port 8000)"]
    
    subgraph ObservabilityLayer["Observability & Session Memory"]
        Tracer["Tracer Decorator (@trace_step)"]
        SessionStore[("SQLite Session DB (Last 6 Turns)")]
    end
    FastApiApp --> Tracer
    FastApiApp --> SessionStore

    subgraph AgentCore["Agent Orchestration Engine"]
        Router["Router Agent (LLM Classifier + Fallback)"]
        Synthesizer["Dual-Source Synthesizer (asyncio.gather)"]
        
        Router -->|'sql'| SQLAgent["SQL Agent Workflow"]
        Router -->|'docs'| DocAgent["Document RAG Agent Workflow"]
        Router -->|'both'| Synthesizer
        Router -->|'chitchat'| FastApiApp
        
        Synthesizer --> SQLAgent
        Synthesizer --> DocAgent
    end
    Tracer --> Router

    subgraph SQLPipeline["Structured Data Engine"]
        Prune["1. Prune Schema (LLM JSON)"]
        GenSQL["2. Few-Shot SQL Generator (Temp 0.0)"]
        ValidateAST["3. sqlglot AST Validator & Limit Injector"]
        DuckDBConn[("4. DuckDB Read-Only Ingest & Store")]
        InferChart["5. Rule-Based Chart Spec Heuristics"]
        
        SQLAgent --> Prune --> GenSQL --> ValidateAST --> DuckDBConn --> InferChart
        ValidateAST -.->|Syntax/Semantic Error Retry Loop (Max 3)| GenSQL
    end

    subgraph DocumentRAG["Unstructured Document Engine"]
        PDFParser["PDF Chunker (pdfplumber, ~500 tok / 80 overlap)"]
        VectorStore[("SQLite Doc Store (float32 BLOB vectors)")]
        Sanitizer["Context Sanitizer (Prompt-Injection Stripper)"]
        HybridRetriever["Hybrid Retrieval (NumPy Cosine + rank_bm25 via RRF)"]
        RefusalGate{"Cosine Score >= 0.45?"}
        CitationNode["Answer Node with [doc p.N] Citations"]
        RefusalNode["Refusal Node: 'Not found in the documents'"]

        DocAgent --> HybridRetriever
        HybridRetriever --> RefusalGate
        RefusalGate -->|No| RefusalNode
        RefusalGate -->|Yes| Sanitizer --> CitationNode
    end

    subgraph LocalInference["Local Ollama Daemon (Port 11434)"]
        OllamaLLM["LLM Chat: qwen2.5:3b"]
        OllamaEmbed["Dense Embeddings: nomic-embed-text (768-dim)"]
    end

    GenSQL -.-> OllamaLLM
    CitationNode -.-> OllamaLLM
    HybridRetriever -.-> OllamaEmbed
    PDFParser -.-> VectorStore
```

---

## 📋 Prerequisites

Before running AgenticInsight, ensure the following native binaries are installed on your system:

| Dependency | Minimum Version | Installation / Verification |
| :--- | :---: | :--- |
| **Python** | `3.11+` | `python3 --version` |
| **Node.js** | `20+` (LTS) | `node --version && npm --version` |
| **Ollama** | Latest | `ollama --version` ([Download Ollama](https://ollama.ai)) |

---

## 🚀 Exact Setup Commands

### Step 1: Pull Local LLM & Embedding Models
AgenticInsight relies entirely on local models running via Ollama:

```bash
# Start Ollama service (if not already running as a daemon)
ollama serve &

# Pull the lightweight chat model (Qwen 2.5 3B)
ollama pull qwen2.5:3b

# Pull the high-performance local embedding model (768-dim)
ollama pull nomic-embed-text
```

### Step 2: Configure Python Virtual Environment & Backend Dependencies

```bash
# Clone and enter the repository root
cd onelogica

# Create a Python virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install backend dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create your local environment configuration
cp .env.example .env
```

### Step 3: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## 🏃 Running the Application

### Option A: One-Click Startup (Recommended)
Launch both the FastAPI backend and Vite frontend together with clean signal trapping:

```bash
./run.sh
```

### Option B: Manual Multi-Terminal Startup

**Terminal 1 — Backend (FastAPI + DuckDB + SQLite):**
```bash
source .venv/bin/activate
export PYTHONPATH="backend:$PYTHONPATH"
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend (React 18 + Vite + Tailwind):**
```bash
cd frontend
npm run dev
```

* **Frontend Dashboard**: [http://localhost:5173](http://localhost:5173)
* **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 📊 Evaluation Benchmark Results

The evaluation benchmark measures accuracy and latency across structured SQL questions, hybrid document retrieval, and refusal gating on out-of-domain questions:

```bash
# Run the complete 50-item benchmark
python -m app.evals.run
```

### Accuracy & Retrieval Hit Rate

| Metric Category | Target Baseline | Observed Score | Tested Samples | Status |
| :--- | :---: | :---: | :---: | :---: |
| **SQL Execution Accuracy** | $\ge 85.0\%$ | **95.0%** (19/20) | 20 | **PASS** |
| **Document Retrieval Hit Rate (Top-5)** | $\ge 90.0\%$ | **100.0%** (20/20) | 20 | **PASS** |
| **Refusal Rate on Unanswerables** | $\ge 90.0\%$ | **100.0%** (10/10) | 10 | **PASS** |
| **Weighted Benchmark Index** | $\ge 88.0\%$ | **98.0%** | 50 | **PASS** |

### Latency Percentiles Distribution (Apple M1 8GB)

| Evaluation Category | Samples | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) | Mean (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SQL Agent (E2E Generation & Exec)** | 20 | **8,770** | **12,106** | 7,285 | 16,515 | 9,333 |
| **Document RAG (Dense + BM25 + RRF)** | 20 | **30** | **62** | 26 | 65 | 33 |
| **Unanswerable Refusal Decision** | 10 | **45** | **100** | 34 | 116 | 53 |
| **Entire 50-Item Benchmark** | 50 | **50** | **10,452** | 26 | 16,515 | 3,757 |

*Detailed per-question breakdown and test logs are stored in [`evals/results.md`](evals/results.md).*

---

## 🔄 Swap the Model

AgenticInsight is decoupled from hardcoded models. Any Ollama-compatible open weights can be configured via environment variables.

### 1. Update `.env`
Edit `.env` at the project root:

```ini
# Chat Model Options: llama3.2:3b | mistral:7b | phi3:mini | qwen2.5-coder:7b
CHAT_MODEL=llama3.2:3b

# Embedding Model Options: nomic-embed-text | all-minilm | bge-m3
EMBED_MODEL=nomic-embed-text

# Ollama Endpoint
OLLAMA_URL=http://localhost:11434
```

### 2. Pull the New Model
```bash
ollama pull llama3.2:3b
```

### 3. Calibrate the Similarity Threshold (Optional)
Different embedding models produce different baseline cosine distributions:
- `nomic-embed-text` (768 dimensions): Recommended threshold is **`0.45`**
- `all-minilm` (384 dimensions): Recommended threshold is **`0.50`**
- `bge-m3` (1024 dimensions): Recommended threshold is **`0.55`**

To adjust, modify `DEFAULT_SIMILARITY_THRESHOLD` in [`backend/app/agents/doc_agent.py`](backend/app/agents/doc_agent.py#L14).

---

## ⚠️ Known Limitations

1. **Hardware & Single-User Memory Profile**:
   - Tuned specifically for local single-user execution on 8GB Apple Silicon hardware.
   - Session context utilizes an in-process SQLite sliding window retaining the last 6 turns (12 messages) to stay within local RAM limits.
2. **Read-Only Analytics Constraints**:
   - DuckDB connections operate strictly in read-only mode.
   - `sqlglot` AST parsing forbids all DDL/DML (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`) and blocks filesystem scan functions (`read_csv`, `read_parquet`, `glob`) before reaching the engine.
3. **PDF Text Layer Requirement**:
   - Documents are ingested via `pdfplumber`. Non-digitized, scanned image PDFs require an external OCR pre-processing step before ingestion.
4. **Cold-Start Model Weight Loading**:
   - The initial query after starting the Ollama daemon requires ~10-15s to load weights into unified memory; subsequent queries run at typical interactive speeds (~30-60ms for RAG retrieval).
5. **No External Vector Database Overhead**:
   - Embeddings are stored natively as float32 BLOBs in SQLite. Designed for collections up to ~100,000 chunks without needing heavy client/server database processes.

---

## 🧪 Testing

Run the full pytest suite:

```bash
# Run all unit, integration, and security tests
pytest -v

# Run with test coverage
pytest -v --durations=10
```
