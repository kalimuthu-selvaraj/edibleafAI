import streamlit as st
import os

# LangChain imports (latest)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# 🔐 Set OpenAI Key
# -----------------------------
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# -----------------------------
# 📄 Load & Process PDF
# -----------------------------
@st.cache_resource
def load_data():
    loader = PyPDFLoader("Edibleaf.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 8})

retriever = load_data()

# -----------------------------
# 🧠 Prompt (Safe + Health Focus)
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant for Edibleaf products.

Rules:
- Answer ONLY from given context
- Focus on product benefits and natural preparation
- Convert strong medical claims into general wellness language
- Use phrases like "may help", "traditionally used"
- Do NOT give medical advice
- Answer in short sentence, bullet list, and small paragraph
- If not found, say: "Not available in catalogue"

Context:
{context}

Question:
{question}
""")

# -----------------------------
# 🤖 LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------
# ⚠️ Safety Layer
# -----------------------------
def add_disclaimer(response, query):
    health_keywords = [
        "disease", "diabetes", "bp", "pressure",
        "gastric", "cancer", "pain", "fever",
        "treatment", "medicine", "cold", "cough"
    ]

    if any(word in query.lower() for word in health_keywords):
        response += "\n\n⚠️ This is general wellness information, not medical advice."

    return response

# -----------------------------
# 🎨 UI
# -----------------------------
st.set_page_config(page_title="Edibleaf AI Assistant", layout="centered")

st.title("🌿 Edibleaf AI Assistant")
st.write("Discover natural, chemical-free products and their benefits.")

# -----------------------------
# 💡 Predefined Questions
# -----------------------------
st.subheader("💡 Try these questions")

pre_questions = [
    "What are the benefits of forest honey?",
    "Which oil is best for cooking?",
    "Tell me about ghee benefits",
    "What helps digestion?",
    "Are your products chemical free?",
    "Which products improve immunity?"
]

cols = st.columns(2)
selected_question = None

for i, q in enumerate(pre_questions):
    if cols[i % 2].button(q):
        selected_question = q

# -----------------------------
# 💬 User Input
# -----------------------------
user_input = st.text_input("Ask your question:")

query = selected_question if selected_question else user_input

# -----------------------------
# 🔎 Get Answer
# -----------------------------
if query:
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(query)
        response = add_disclaimer(response, query)

    st.subheader("🧠 Answer")
    st.write(response)
