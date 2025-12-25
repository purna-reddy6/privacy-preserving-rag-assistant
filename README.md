
# Privacy-Preserving RAG Assistant (Local LLM)

A fully offline, privacy-first Retrieval-Augmented Generation (RAG) system for analyzing research papers using local LLMs.
No cloud APIs. No data leakage. Everything runs on your machine.

**Features**
Ingest and analyze research PDFs
Semantic search using FAISS vector database
Local LLM inference via Ollama
100% offline & privacy-preserving
Interactive Streamlit UI
Built with LangChain 1.x (LCEL) – future-proof

**Tech Stack**
Python 3.10+
LangChain 1.x (LCEL)
FAISS – vector similarity search
HuggingFace sentence-transformers
Ollama – local LLM server
Streamlit – web UI

**Setup**
->Clone the repository
git clone https://github.com/your-purna-reddy6/privacy-rag-assistant.git
cd privacy-rag-assistant

->Install Python dependencies
pip install -r requirements.txt

->Install & setup Ollama
macOS (recommended)
brew install ollama


Or download the app:
 https://ollama.com/download

Pull a lightweight local model:
ollama pull phi

**Run**

Step 1: Start Ollama server
ollama serve
(Keep this running in a terminal or open Ollama.app)

Step 2: Add research papers
Place your PDF files inside:
data/

Step 3: Ingest PDFs (run once)
python3 app/ingest.py

Step 4: Launch the UI
streamlit run ui.py
 
