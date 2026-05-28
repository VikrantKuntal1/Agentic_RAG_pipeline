import streamlit as st
import tempfile
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()
@st.cache_resource
def build_graph(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY"))
    

    grade_prompt = PromptTemplate.from_template("""
You are grading whether retrieved chunks are relevant to a question.
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

    def route_after_grade(state):
        if state["verdict"] == "yes":
            return "generate"
        elif state["attempts"] < 3:
            return "rewrite"
        else:
            return "give_up"

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("generate", generate_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("give_up", give_up_node)

    builder.add_edge(START, "retrieve")
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


st.title("Doc Enquirer")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    st.success(f"Loaded: {uploaded_file.name}")

    with st.spinner("Processing document..."):
        graph = build_graph(uploaded_file.read())

    question = st.text_input("Ask a question about the document")

    if st.button("Ask"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and thinking..."):
                result = graph.invoke({
                    "question": question,
                    "rewritten_question": question,
                    "chunks": [],
                    "answer": "",
                    "attempts": 0,
                    "verdict": "",
                })
            st.write("### Answer")
            st.write(result["answer"])
