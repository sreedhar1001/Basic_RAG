"""
Minimal Agentic RAG with LangGraph
Flow: retrieve -> grade -> (relevant? generate : rewrite -> retrieve again) -> generate
"""
from typing import TypedDict, List

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# ---------- 1. Setup: LLM + vector store ----------
llm = ChatAnthropic(model="claude-sonnet-5", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Replace these with your own documents (PDF loaders, text splitters, etc.)
docs = [
    Document(page_content="Our refund policy allows returns within 30 days of purchase."),
    Document(page_content="Shipping inside India takes 3-5 business days."),
    Document(page_content="Premium support is available 24/7 for enterprise customers."),
    Document(page_content="Passwords must be at least 12 characters long."),
]
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

MAX_RETRIES = 2  # always cap the loop!


# ---------- 2. State: the data passed between steps ----------
class State(TypedDict):
    question: str
    documents: List[Document]
    relevant: bool
    retries: int
    answer: str


# ---------- 3. Nodes: each is just a function ----------
def retrieve(state: State) -> dict:
    print(f"[retrieve] searching for: {state['question']}")
    return {"documents": retriever.invoke(state["question"])}


def grade(state: State) -> dict:
    """Ask the LLM: are these documents useful for the question?"""
    context = "\n".join(d.page_content for d in state["documents"])
    prompt = (
        f"Question: {state['question']}\n\nDocuments:\n{context}\n\n"
        "Do the documents contain information that helps answer the question? "
        "Reply with only 'yes' or 'no'."
    )
    verdict = llm.invoke(prompt).content.strip().lower()
    is_relevant = verdict.startswith("yes")
    print(f"[grade] relevant = {is_relevant}")
    return {"relevant": is_relevant}


def rewrite(state: State) -> dict:
    """Rewrite the question to get better search results."""
    prompt = (
        f"Rewrite this question to be clearer for a document search. "
        f"Return only the new question.\n\nQuestion: {state['question']}"
    )
    new_q = llm.invoke(prompt).content.strip()
    print(f"[rewrite] new question: {new_q}")
    return {"question": new_q, "retries": state["retries"] + 1}


def generate(state: State) -> dict:
    context = "\n".join(d.page_content for d in state["documents"])
    prompt = (
        "Answer the question using only the context below. "
        "If the context is not enough, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}"
    )
    return {"answer": llm.invoke(prompt).content}


# ---------- 4. Decision function: this is the "agentic" part ----------
def decide_next(state: State) -> str:
    if state["relevant"]:
        return "generate"
    if state["retries"] >= MAX_RETRIES:
        return "generate"  # give up retrying, answer with what we have
    return "rewrite"


# ---------- 5. Build the graph ----------
graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade)
graph.add_node("rewrite", rewrite)
graph.add_node("generate", generate)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", decide_next, {"generate": "generate", "rewrite": "rewrite"})
graph.add_edge("rewrite", "retrieve")  # the loop
graph.add_edge("generate", END)

app = graph.compile()


# ---------- 6. Run it ----------
if __name__ == "__main__":
    result = app.invoke({
        "question": "how long do i have to send stuff back?",
        "documents": [],
        "relevant": False,
        "retries": 0,
        "answer": "",
    })
    print("\nFINAL ANSWER:", result["answer"])
