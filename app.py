
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import streamlit as st
import os

api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key

# -------------------------
# Load API Key
# -------------------------
load_dotenv()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Samsung AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.main{
    background:#f5f8fc;
}

.header{
    background:linear-gradient(90deg,#1428A0,#0f7cf0);
    color:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    margin-bottom:20px;
}

.header h1{
    font-size:42px;
}

.header p{
    font-size:18px;
}

.stChatMessage{
    border-radius:15px;
}

.user-box{
    background:#eaf4ff;
    padding:12px;
    border-radius:10px;
}

.answer-box{
    background:white;
    padding:15px;
    border-radius:12px;
    border-left:6px solid #1428A0;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

.sidebar-title{
    color:#1428A0;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='header'>
<h1>🤖 Samsung AI Assistant</h1>
<p>Ask anything about your Samsung Washing Machine Manual</p>
</div>
""", unsafe_allow_html=True)

col1,col2,col3=st.columns(3)

col1.metric("📚 Manual","Loaded")

col2.metric("🧠 Model","GPT-4o-mini")

col3.metric("💾 Database","ChromaDB")


# -------------------------with st.sidebar:
with st.sidebar:
    
    st.title("📘 Samsung Assistant")

    st.success("Manual Ready")

    st.markdown("---")

    st.subheader("Example Questions")

    st.write("• How do I clean the drum?")

    st.write("• How do I use Child Lock?")

    st.write("• What is Eco Mode?")

    st.write("• How do I pause the cycle?")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages=[]

        st.rerun()

    st.markdown("---")

    st.caption("Made with Streamlit + LangChain")


# Models
# -------------------------
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

DB_FOLDER = "chroma_db"

# -------------------------
# Create Vector Database
# -------------------------
if not os.path.exists(DB_FOLDER):

    with st.spinner("Preparing manual..."):

        loader = UnstructuredHTMLLoader("1.html")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=DB_FOLDER
        )

# -------------------------
# Load Database
# -------------------------
db = Chroma(
    persist_directory=DB_FOLDER,
    embedding_function=embedding
)

retriever = db.as_retriever(search_kwargs={"k":3})

# -------------------------
# Chat History
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


question = st.chat_input("💬 Ask your Samsung question...")

if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Create docs FIRST
    docs = retriever.invoke(question)

    # THEN use docs
    with st.expander("📄 Retrieved Manual Sections"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"### Section {i}")
            st.write(doc.page_content[:400] + "...")

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
    You are a Samsung Washing Machine expert.

    Manual:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt).content

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

