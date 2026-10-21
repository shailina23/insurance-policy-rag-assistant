# Insurance Policy RAG Assistant

> **Status: learning portfolio project in progress.** This is a local prototype over entirely fictional policy documents. It is not an insurance advice tool and is not connected to any insurer, customer, or claims system.

A small retrieval-augmented generation (RAG) command-line application. It chunks synthetic policy documents, creates embeddings, retrieves the most similar passages, and asks a language model to answer using those passages with source labels.

## What this project demonstrates

- Markdown document ingestion and heading-aware chunking
- Embedding generation and cosine-similarity retrieval
- Grounded answer generation with source references
- An answerable and unanswerable question set for manual evaluation
- A clear separation between retrieved evidence and generated answer

## Repository layout

```text
data/
  policies/                 Fictional policy documents
  evaluation/questions.jsonl Questions and expected evidence for evaluation
src/insurance_policy_rag.py CLI for indexing and asking questions
docs/architecture.md        Data flow, decisions, and known limits
docs/evaluation-plan.md     Evaluation rubric and next milestones
```

## Quick start

Requires Python 3.10+ and an OpenAI API key with access to an embeddings model and a Responses API model.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

Add your API key to `.env` locally. Never commit `.env` or paste API keys into source files.

Build the local embedding index and ask a question:

```bash
python src/insurance_policy_rag.py index
python src/insurance_policy_rag.py ask "What is the collision deductible?"
```

The first command sends document text to the configured embeddings API. The second sends the question and retrieved passages to the configured Responses API model. API usage may incur charges; check provider pricing and your account limits before running.

## Configuration

Copy `.env.example` to `.env` and set:

- `OPENAI_API_KEY`: your secret API key
- `OPENAI_EMBEDDING_MODEL`: defaults to `text-embedding-3-small`
- `OPENAI_CHAT_MODEL`: set to a text model available in your API account
- `RAG_TOP_K`: number of passages to retrieve (default 4)

The generated local index is written to `data/index.json` and ignored by Git. Re-run `index` after changing the policy corpus.

## Dataset

Three hand-authored synthetic documents cover auto, home, and travel policy examples. The terms, limits, amounts, insurer, and product names are fictional and designed only to support retrieval experiments.

## Evaluation

Start with the questions in `data/evaluation/questions.jsonl`. Review whether retrieval returns the expected document and whether the answer uses only evidence in the retrieved passages. See `docs/evaluation-plan.md` for the rubric and planned metrics. No evaluation results are claimed yet.

## Limitations and next work

- The initial index is a local JSON file and a linear scan; it is for a small corpus, not production scale.
- Chunking uses character lengths and paragraph boundaries, not token-aware segmentation.
- The prototype does not yet include automated evaluation, a web interface, authentication, access control, monitoring, or deployment.
- Similarity ranking is not a guarantee that a passage answers a question. Evaluate retrieval and answer quality before using this pattern on real documents.
- The app is a learning prototype and must not be used to make coverage, claims, legal, or financial decisions.

## References

- [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings)
- [OpenAI text generation guide](https://developers.openai.com/api/docs/guides/text)
