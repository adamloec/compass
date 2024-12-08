import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# # Testing compass here
# from compass import Compass
# c = Compass(dir_path="test_data/Checkers")

from feature_extractor import FeatureExtractorPipeline
pipeline = FeatureExtractorPipeline(known_features=["board", "player", "menu", "pieces"])
feature_dict = pipeline.run_pipeline()
# output_file = "feature_test_cases.txt"
# with open(output_file, "w") as f:
#     for feature_name in feature_dict.keys():
#         test_cases = pipeline.generate_test_cases_for_feature(feature_dict, feature_name)
#         f.write(f"Feature: {feature_name}\n")
#         f.write("Test Cases:\n")
#         f.write(test_cases)
#         f.write("\n\n")

# # Comparing RAG QA / instruct with compass and repo embeddings
# from chains import QAChain, ContextRetriever
# retriever = ContextRetriever(data_path="test_data/Checkers", data_type="repo", retrieval_type="similarity")
# chain = QAChain(context_retriever=retriever)

# print(chain.query("What does the AI code do?"))