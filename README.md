# Multi-Agent RAG Chatbot

A multi-agent RAG (Retrieval Augmented Generation) chatbot system for JupiterIQ that handles customer inquiries across multiple departments (HR, Tech, Finance, Legal).

## Live Demo & Monitoring

<!-- - **Product Demo**: [https://rag-chatbot-fe-self.vercel.app/](https://rag-chatbot-fe-self.vercel.app) -->

- **API Documentation**: [https://multi-agent-rag-chatbot.onrender.com/docs](https://multi-agent-rag-chatbot.onrender.com/docs)
- **Status Page**: [https://stats.uptimerobot.com/5z2EBCHShQ](https://stats.uptimerobot.com/5z2EBCHShQ)
<!-- - **Metrics**: [https://rag-based-chatbot-96uz.onrender.com/api/v1/query/metrics](https://rag-based-chatbot-96uz.onrender.com/api/v1/query/metrics) -->
- **Report**: [https://github.com/jerrybuks/multi-agent-rag-chatbot/blob/main/reports/REPORT.md](https://github.com/jerrybuks/multi-agent-rag-chatbot/blob/main/reports/REPORT.md)


## Documentation

- **[📊 Technical Report](reports/REPORT.md)** - Comprehensive architecture, design decisions, and system overview
- **[🔄 Flow Diagrams](FLOW_DIAGRAM.md)** - Component interaction diagram and key decision points
- **[📚 Golden Datasets](data/golden_datasets/README.md)** - Test dataset documentation

## API Key Setup

1. Create an API key in [OpenRouter](https://openrouter.ai/).

   You will also need to setup [langfuse](https://cloud.langfuse.com/)
   - sign up on langfuse
   - creat a project and get your keys (secret and private key)


2. Add your keys to the `.env` file:
   ```bash
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   LANGFUSE_PUBLIC_KEY=langfuse_public_key
   LANGFUSE_SECRET_KEY=langfuse_secret_key
   ```
   OpenRouter uses `https://openrouter.ai/api/v1` by default. The existing vector
   store keeps using the same Ada 002 embedding model, so it does not need to be
   rebuilt for this provider change. Rebuild it if you change `EMBEDDING_MODEL`.

## Starting Local Server Setup

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install .
```

### 3. Build the Vector Index

Run the indexing script to process handbooks and generate embeddings:

```bash
python src/build_index.py
```

This will:
- Load all handbook markdown files from the `data/` folder
- Intelligently chunk each handbook using markdown header-based splitting
- Generate embeddings for each chunk
- Save chunks and embeddings to JSONL files in the `data/` folder

### Output Files

The script generates the following files in the `data/` folder:

- `{handbook_name}_chunks.jsonl` - Chunks without embeddings (one per handbook)
- `{handbook_name}_embeddings.jsonl` - Chunks with embeddings (one per handbook)
- `all_handbooks_chunks.jsonl` - Combined chunks from all handbooks
- `all_handbooks_embeddings.jsonl` - Combined chunks with embeddings

## Running the Application

```bash
fastapi dev src/main.py
```

The server will be available at `http://localhost:8000`

- **API Docs**: http://localhost:8000/docs

## Running Tests

The system includes a test runner that uses golden datasets to validate the chatbot's responses with automatic quality scoring via Langfuse.

### Quick Start

Run all tests:
```bash
python tests/test_runner.py
```

Run a specific dataset:
```bash
python tests/test_runner.py --dataset finance.jsonl
```

Run with custom settings:
```bash
python tests/test_runner.py \
  --dataset finance.jsonl \
  --max-tests 5 \
  --min-similarity 0.75
```

## Project Structure

```
.
├── data/                    # Handbook markdown files and generated indexes
│   ├── handbooks/          # Source handbook markdown files
│   ├── jsonl/              # Generated chunks (JSONL format)
│   ├── vectorstore/        # Chroma vector stores (one per handbook)
│   └── golden_datasets/    # Test datasets for evaluation
├── src/
│   ├── build_index.py      # Index building script
│   ├── main.py             # FastAPI application
│   ├── indexing/           # Indexing utilities (parsing, chunking, embeddings)
│   ├── querying/           # Query processing (agents, routes, tools)
│   ├── evaluation/         # Langfuse evaluator for quality scoring
│   ├── config/             # Configuration settings
│   └── utils/              # Utility functions (LLM, storage)
├── tests/                  # Test files
│   └── test_runner.py      # Golden dataset test runner
├── reports/                # Documentation and reports
│   └── REPORT.md           # Comprehensive technical report
└── pyproject.toml          # Python dependencies
```

## Known Limitations

1. **Session Management**: Session IDs are generated from IP addresses, which means users behind the same NAT/proxy will share session context.
2. **Context Window**: Conversation history is limited to the last 20 messages to prevent context bloat and maintain performance.
3. **Vector Store**: Vector stores are preloaded at startup and stored in memory; very large knowledge bases may require additional memory resources.

