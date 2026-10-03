Markdown
# Minimal Agentic RAG with LangGraph

A lightweight, production-ready implementation of an **Agentic Retrieval-Augmented Generation (RAG)** pipeline built using **LangGraph**, **LangChain**, **ChromaDB**, and **Anthropic Claude**.

Unlike standard RAG pipelines that follow a rigid linear flow (Retrieve → Generate), this agentic workflow dynamically evaluates whether retrieved context is sufficient. If the retrieved documents fail grading, the system automatically rewrites the query and retries vector search up to a configurable retry limit (`MAX_RETRIES`).

---

## 🔄 Architecture & Graph Flow

The pipeline executes as a state graph:

[Entry] ──► retrieve ──► grade ──► decide_next?
│
┌────────────────┴────────────────┐
▼                                 ▼
(is_relevant)                  (!is_relevant)
│                                 │
▼                          (retries < MAX_RETRIES)
generate                             │
│                                 ▼
▼                              rewrite
[END]                               │
└──► retrieve (Loop)
