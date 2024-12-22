import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def write_results_to_file(file_path=None, data_dict=None):
    import json
    with open(file_path, 'w') as file:
        json.dump(data_dict, file, indent=4)

def write_test_cases_to_file(test_cases, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("TEST CASES\n")
        f.write("=" * 50 + "\n\n")
        
        for outer_key, inner_dict in test_cases.items():
            for category, content in inner_dict.items():
                # Format category title
                category_title = category.upper().replace('_', ' ')
                f.write(f"\n{category_title}\n")
                f.write("-" * len(category_title) + "\n\n")
                
                # Write the test cases content
                if isinstance(content, str):
                    f.write(content.strip())
                    f.write("\n\n" + "=" * 50 + "\n")

""" #################################################################

### TODO
# Test different snippets, code, summaries, code + summaries?
# Feature generation needs a lot of work


# Test Repositories
# https://github.com/VikramBombhi/Checkers.git
# https://github.com/jironghuang/chess.git

    ################################################################# """


"""Test code for compass, feature agent, and test case agent"""
from compass import Compass
from compass.vector_store import VectorStore
from compass.feature_agent import FeatureAgent
from compass.iterative_feature_agent import IterativeFeatureAgent
from compass.test_case_agent import TestCaseAgent

from langchain.chains.sequential import SequentialChain

# Parse feature_agent_results.json, print features
# with open("zero_core_vdb/feature_agent_results.json", "r") as file:
#     import json
#     features = json.load(file)
#     for feature in features.get("feature_dict"):
#         print(feature)

# Compass
vdb_dir = "chroma_db/zero_core_vdb"

if not os.path.exists(vdb_dir):
    chess_compass = Compass(dir_path="test_repos/ZeroCore")
    vector_store = VectorStore.from_compass(chess_compass, persist=True)
else:
    vector_store = VectorStore.from_persist_storage(vdb_dir)

# Creating the agents
feature_agent = IterativeFeatureAgent.as_chain(vector_store=vector_store)
features = feature_agent.invoke({})
write_results_to_file(file_path="feature_agent_results.json", data_dict=features)

# test_case_agent = TestCaseAgent.as_chain(feature_dict={}, compass=compass_checkers)

# # Actually running the sequential chain (this will be inside each job)
# # Jobs might have, on init, a member var of chains that gets put into seq chain when the job is ran idk
# chain = SequentialChain(
#     chains=[feature_agent, test_case_agent],
#     input_variables=[],
#     output_variables=["test_cases"],
#     verbose=True
# )
# result = chain.invoke({})
# write_test_cases_to_file(result, "example_output.txt")