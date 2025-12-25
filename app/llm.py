from langchain_community.llms import Ollama

def load_llm():
    return Ollama(
        model="phi",
        temperature=0.1
    )
