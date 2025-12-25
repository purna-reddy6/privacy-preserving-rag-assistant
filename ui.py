import streamlit as st
from app.rag import load_rag

st.set_page_config(page_title="Privacy-Preserving RAG", layout="wide")
st.title("🔐 Privacy-Preserving Research Assistant")

rag_chain = load_rag()

query = st.text_input("Ask a question about your research papers")

if query:
    with st.spinner("Thinking locally..."):
        answer = rag_chain.invoke(query)
        st.subheader("Answer")
        st.write(answer)
