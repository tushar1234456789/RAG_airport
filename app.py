import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# --- Sidebar Config ---
st.sidebar.title("🔐 Groq API Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
base_url = "https://api.groq.com/openai/v1"
model = st.sidebar.selectbox("Select Groq Model", [
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it"
])

temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.05)

# --- Main UI ---
st.title("💬 RAG QA with Groq & FAISS")
query = st.text_input("Ask a question about Changi Airport:")
run_button = st.button("Get Answer")

# --- RAG Logic ---
if api_key and run_button and query:
    with st.spinner("Loading and searching..."):

        # Load FAISS vector store
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("changi_faiss", embedding, allow_dangerous_deserialization=True)

        # Use Groq as LLM
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url
        )

        # RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_type="similarity", k=4),
            return_source_documents=True
        )

        # Get result
        response = qa_chain(query)
        st.success("✅ Answer received!")

        st.subheader("🧠 Answer")
        st.write(response["result"])

        st.subheader("📄 Source Documents")
        for i, doc in enumerate(response["source_documents"], 1):
            st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'N/A')}")
            st.code(doc.page_content[:1000], language="markdown")

elif run_button:
    st.warning("⚠️ Please enter your Groq API key to continue.")
