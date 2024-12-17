import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def write_results_to_file(file_path=None, data_dict=None):
    import json
    with open(file_path, 'w') as file:
        json.dump(data_dict, file, indent=4)

# Test Repositories
# https://github.com/VikramBombhi/Checkers.git
# https://github.com/jironghuang/chess.git

"""Test code for compass, feature agent, and test case agent"""
from compass import Compass
from compass.vector_store import VectorStore
from compass.feature_agent import FeatureAgent
from compass.test_case_agent import TestCaseAgent

from langchain.chains.sequential import SequentialChain

# Compass
compass_checkers = Compass(dir_path="test_repos/chess")

# Vector Storage
vector_store = VectorStore(source=compass_checkers)

# Creating the agents
feature_agent = FeatureAgent.as_chain(vector_store=vector_store)
test_case_agent = TestCaseAgent.as_chain(feature_dict={}, compass=compass_checkers)

# Actually running the sequential chain (this will be inside each job)
# Jobs might have, on init, a member var of chains that gets put into seq chain when the job is ran idk
chain = SequentialChain(
    chains=[feature_agent, test_case_agent],
    input_variables=[],
    output_variables=["test_cases"],
    verbose=True
)
result = chain({})