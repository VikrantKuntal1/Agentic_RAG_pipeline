from typing import TypedDict, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END

import os

from dotenv import load_dotenv

load_dotenv()


loader = PyPDFLoader("AAAI-2025-PresPanel-Report-FINAL.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embeddings)
retriever = db.as_retriever()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


class RAGstate(TypedDict):
    question: str
    rewritten_question: str
    chunks: list
    answer: str
    attempts: int
    verdict: str


def retrieve(state):
    chunks = retriever.invoke(state["rewritten_question"])
    return {"chunks": chunks}

grade_prompt = PromptTemplate.from_template(""" 
you are grading whether the retrieved chunks are relevant to the question.
question: {question}
chunks: {chunks}
Are the chunks relevant to the answering the question? Reply with only "yes" or "no".""")

def grade(state):
    chunks_text = "\n\n".join(doc.page_content for doc in state["chunks"])
    result = (grade_prompt|llm|StrOutputParser()).invoke({
        "question":state["question"],
        "chunks": chunks_text,
    })
    verdict = result.strip().lower()
    return {"verdict": verdict}

generate_prompt = PromptTemplate.from_template("""
Use the following context to answer the question.
context : {context}
question: {question}
Answer: """)

def generate(state):
    chunks_text = "\n\n".join(doc.page_content for doc in state["chunks"])
    answer = (generate_prompt|llm|StrOutputParser()).invoke({
        "context":chunks_text,
        "question": state["question"]
    })
    return ({"answer": answer})

rewrite_prompt = PromptTemplate.from_template(""" 
The following question did not retrieve  relevatn results.
Rewrite it to be clearer and  more specific for searching a document.
original_question: {question}
rewritten_question: 
""")

def rewrite(state):
    new_question = (rewrite_prompt|llm|StrOutputParser()).invoke({
        "question": state["question"]
    })
    return{
        "rewritten_question": new_question.strip(),
        "attempts": state["attempts"] + 1
    }

def route_after_grade(state):
    if state["verdict"] == "yes":
        return "generate"
    elif state["attempts"] < 3:
        return "rewrite"
    else:
        return "give_up"
    
def give_up(state):
    return {"answer": "could not find a condfident answer to that in the document."}
    

graph_builder = StateGraph(RAGstate)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("grade", grade)
graph_builder.add_node("generate", generate)
graph_builder.add_node("rewrite", rewrite)
graph_builder.add_node("give_up", give_up)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve","grade")
graph_builder.add_conditional_edges(
    "grade",
    route_after_grade,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "give_up": "give_up"
    })
graph_builder.add_edge("rewrite", "retrieve")
graph_builder.add_edge("generate", END)
graph_builder.add_edge("give_up", END)

graph = graph_builder.compile()


question = "What are the main findings of this report?"

initial_state = {
    "question": question,
    "rewritten_question": question,
    "chunks": [],
    "answer": "",
    "attempts": 0,
    "verdict": "",
}

final_state = graph.invoke(initial_state)
print(final_state["answer"])