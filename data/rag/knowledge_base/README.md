## News Credibility — RAG Knowledge Base (local)

This folder contains **small, curated markdown notes** derived from trusted, high-level public guidance.
They are used as a **demo-friendly local corpus** for the RAG pipeline (MiniLM embeddings + FAISS).

### How it is used

- Build / rebuild the FAISS index:

```bash
python scripts/build_rag_index.py
```

Outputs:
- `data/rag/faiss.index`
- `data/rag/chunks.json`

### Sources

The markdown files include links to the original references and are written as short summaries/checklists.
They are **not** a live web crawl and **not** a substitute for professional fact-checking.

