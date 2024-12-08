import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Testing compass here
from compass import Compass
c = Compass(dir_path="test_data/Checkers")

# Comparing RAG QA / instruct with compass and repo embeddings
from chains import QAChain, ContextRetriever
retriever = ContextRetriever(data_path="test_data/Checkers", data_type="repo", retrieval_type="similarity")
chain = QAChain(context_retriever=retriever)

print(chain.query("What does the AI code do?"))