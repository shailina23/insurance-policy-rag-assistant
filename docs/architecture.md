# Architecture and design notes

## Current flow

```text
Synthetic Markdown policies
        ↓ heading-aware paragraph chunking
OpenAI embeddings API → local JSON index
        ↓ query embedding + cosine similarity
Top-k passages with document and section metadata
        ↓ Responses API with grounded-answer instructions
Answer plus source list printed in the CLI
```

## Retrieval design

The first implementation chunks on Markdown headings and paragraph boundaries with a character limit. Each chunk stores a stable ID, filename, section heading, text, and embedding. A query is embedded with the same embedding model. Because the returned vectors are normalized, their dot product ranks cosine similarity. The small local index is scanned linearly and the highest-scoring `RAG_TOP_K` chunks are passed to the answer step.

## Grounding behavior

The generation instructions tell the model to answer only from the supplied passages, cite passage labels such as `[S1]`, and say when the passages do not support an answer. Retrieved text is untrusted content: the model is told not to follow instructions contained inside source documents. The CLI displays retrieved source IDs and metadata so a person can inspect the evidence.

## Important limits

- Linear search and JSON storage are suitable only for a tiny portfolio corpus.
- Character-based chunk sizes do not map exactly to model tokens.
- The app currently has no authentication, tenant isolation, document deletion workflow, audit log, UI, or production monitoring.
- A prompt instruction does not guarantee correct abstention or citations. These behaviors need evaluation.
- Embedding and generation requests send the submitted text to the configured API provider. Do not index confidential client information in this public learning project.

## Planned improvements

1. Compare chunk size and overlap choices against the provided questions.
2. Add automated retrieval and grounded-answer evaluation.
3. Add metadata filters and version-aware retrieval.
4. Add an interface and explicit evidence links.
5. Compare the custom pipeline with a managed retrieval tool.
