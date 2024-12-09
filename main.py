import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Testing compass here
from compass import Compass
from compass.feature_retriever import CompassFeatureRetriever

# Compass summaries
compass_checkers = Compass(dir_path="test_repos/Checkers")

# Identifying features in compass summaries
features = CompassFeatureRetriever.get_features(compass=compass_checkers, known_features=["board", "player", "AI", "menu"])

# # Comparing RAG QA / instruct with compass and repo embeddings
# from chains import QAChain, ContextRetriever
# retriever = ContextRetriever(data_path="test_data/Checkers", data_type="repo", retrieval_type="similarity")
# chain = QAChain(context_retriever=retriever)

# print(chain.query("What does the AI code do?"))