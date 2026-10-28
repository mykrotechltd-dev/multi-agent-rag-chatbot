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

## Browser Chat, Sign-In, and Uploaded Sources

The FastAPI app serves the chat at `/`, the sign-in page at `/login`, and the source manager at `/sources`. Chat and query APIs require a shared-password session. Source upload, listing, chunk preview, and deletion additionally require the administrator token.

Create three separate random values of at least 32 characters for `CHAT_ACCESS_PASSWORD`, `CHAT_SESSION_SECRET`, and `SOURCE_ADMIN_TOKEN`. For example, run this command three times and use a different output each time:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set the values in local `.env` and in the Vercel project's Production, Preview, and Development environments as appropriate. Never commit real secrets. Also set:

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
# Optional; defaults to uploaded_sources
QDRANT_COLLECTION=uploaded_sources
```

`QDRANT_API_KEY` may be omitted for a local self-hosted Qdrant instance. The app uses the configured OpenRouter/OpenAI embedding key and `EMBEDDING_MODEL`; the Qdrant collection must use cosine distance and match that model's vector size. Uploaded chunks record the embedding model and are only searched with the same model. If you change `EMBEDDING_MODEL`, delete and re-upload those sources.

The source manager accepts PDF, DOCX, TXT, and Markdown. Each upload is limited to 4 MB, 500 PDF pages, 25 MB of expanded DOCX contents, 500,000 extracted characters, and 500 chunks. Scanned PDFs are not OCR'd. The original binary is discarded after indexing; chunk text and source metadata remain in Qdrant. Vercel functions reject request bodies over 4.5 MB, which is why the app's upload limit is lower ([Vercel function limits](https://vercel.com/docs/functions/limitations)).

Every signed-in user can query all uploaded source content. The agent scope selector affects which specialist searches a file; it is not per-user access control. The shared password is suitable for a private beta, not individual customer accounts. A user must sign in before searching; source-management actions also need the separate admin token.

Same-origin browser requests need no CORS setting. If an external frontend must call the API, set `CORS_ALLOWED_ORIGINS` to a comma-separated list of exact origins.

### Source API

- `GET /api/v1/auth/session`, `POST /api/v1/auth/login`, `POST /api/v1/auth/logout`
- `GET /api/v1/sources` lists indexed files and chunk counts.
- `POST /api/v1/sources` accepts multipart fields `file` and `scope` (`all`, `finance`, `hr`, `legal`, `tech`, or `general_knowledge`).
- `GET /api/v1/sources/{source_id}/chunks` previews stored chunks.
- `DELETE /api/v1/sources/{source_id}` deletes that source and its chunks.

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


