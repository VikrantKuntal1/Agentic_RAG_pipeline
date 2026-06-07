from langsmith import traceable
import streamlit as st
import tempfile
import os
import hashlib
os.environ["FASTEMBED_CACHE_PATH"] = os.path.expanduser("~/fastembed_cache")
os.makedirs(os.path.expanduser("~/fastembed_cache"), exist_ok=True)

from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv(override=True)

print("TRACING:", os.getenv("LANGSMITH_TRACING"))
print("PROJECT:", os.getenv("LANGSMITH_PROJECT"))
print("API KEY:", os.getenv("LANGSMITH_API_KEY")[:10] if os.getenv("LANGSMITH_API_KEY") else "NOT SET")
print("DIRECT TEST:", os.getenv("LANGSMITH_API_KEY"))


@st.cache_resource
def load_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def build_graph(file_bytes, file_hash):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    embeddings = load_embeddings()
    # Unique collection per document — prevents Chroma reusing "langchain"
    # collection and mixing chunks from different documents
    db = Chroma.from_documents(chunks, embeddings, collection_name=f"doc_{file_hash[:16]}")
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 6, "score_threshold": 0.3}
    )
    retriever_broad = db.as_retriever(search_kwargs={"k": 10})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    intent_prompt = PromptTemplate.from_template("""
Classify the user's request into one of two categories:
1. "summarise" - if they want a summary, overview, key points, or general understanding of the document.
2. "question" - if they have a specific question about the document that they want answered.
Request: {question}
Reply with only one word, either "summarise" or "question".""")

    summary_prompt = PromptTemplate.from_template("""
You are summarising a document. Using the content below, provide a clear and structured
summary covering the main points, key details, and overall insights. Be concise but
comprehensive, ensuring that the essence of the document is captured effectively.

Document content:
{content}
Summary:""")

    grade_prompt = PromptTemplate.from_template("""
You are grading whether retrieved chunks contain ANY information that could help answer the question.
Be lenient - if the chunks are even partially relevant, reply yes.
Only reply no if the chunks are completely unrelated to the question.
Question: {question}
Chunks: {chunks}
Reply with only yes or no.""")

    generate_prompt = PromptTemplate.from_template("""
Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:""")

    rewrite_prompt = PromptTemplate.from_template("""
The following question did not retrieve relevant results.
Rewrite it to be clearer and more specific for searching a document.
Original question: {question}
Rewritten question:""")

    class RAGState(TypedDict):
        question: str
        rewritten_question: str
        chunks: list
        answer: str
        attempts: int
        verdict: str
        intent: str

    def detect_intent_node(state):
        intent = (intent_prompt | llm | StrOutputParser()).invoke({
            "question": state["question"]
        })
        return {"intent": intent.strip().lower()}

    def summarise_node(state):
        broad_chunks = retriever_broad.invoke("main topics overview summary")
        content = "\n\n".join(doc.page_content for doc in broad_chunks)
        answer = (summary_prompt | llm | StrOutputParser()).invoke({
            "content": content
        })
        return {"answer": answer}

    def retrieve_node(state):
        return {"chunks": retriever.invoke(state["rewritten_question"])}

    def grade_node(state):
        chunks_text = "\n\n".join(doc.page_content for doc in state["chunks"])
        result = (grade_prompt | llm | StrOutputParser()).invoke({
            "question": state["question"],
            "chunks": chunks_text
        })
        return {"verdict": result.strip().lower()}

    def generate_node(state):
        chunks_text = "\n\n".join(doc.page_content for doc in state["chunks"])
        answer = (generate_prompt | llm | StrOutputParser()).invoke({
            "context": chunks_text,
            "question": state["question"]
        })
        return {"answer": answer}

    def rewrite_node(state):
        new_question = (rewrite_prompt | llm | StrOutputParser()).invoke({
            "question": state["question"]
        })
        return {
            "rewritten_question": new_question.strip(),
            "attempts": state["attempts"] + 1
        }

    def give_up_node(state):
        return {"answer": "I couldn't find a confident answer to that in the document."}

    def route_after_intent(state):
        if "summar" in state["intent"]:
            return "summarise"
        return "retrieve"

    def route_after_grade(state):
        if state["verdict"] == "yes":
            return "generate"
        elif state["attempts"] < 3:
            return "rewrite"
        else:
            return "give_up"

    builder = StateGraph(RAGState)
    builder.add_node("detect_intent", detect_intent_node)
    builder.add_node("summarise", summarise_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("generate", generate_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("give_up", give_up_node)

    builder.add_edge(START, "detect_intent")
    builder.add_conditional_edges("detect_intent", route_after_intent, {
        "summarise": "summarise",
        "retrieve": "retrieve"
    })
    builder.add_edge("summarise", END)
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate",
        "rewrite": "rewrite",
        "give_up": "give_up"
    })
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("give_up", END)

    os.unlink(tmp_path)
    return builder.compile()

load_embeddings()

st.title("Doc Enquirer")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Rebuild graph only when a different document is uploaded (per session)
    if st.session_state.get("file_hash") != file_hash:
        with st.spinner("Processing document..."):
            st.session_state.graph = build_graph(file_bytes, file_hash)
            st.session_state.file_hash = file_hash

    st.success(f"Loaded: {uploaded_file.name}")
    graph = st.session_state.graph

    @traceable(name="doc-enquirer-run")
    def run_graph(question):
        return graph.invoke({
            "question": question,
            "rewritten_question": question,
            "chunks": [],
            "answer": "",
            "attempts": 0,
            "verdict": "",
            "intent": ""
        })

    question = st.text_input("Ask a question about the document")

    if st.button("Ask"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and thinking..."):
                result = run_graph(question)
            st.write("### Answer")
            st.write(result["answer"])
