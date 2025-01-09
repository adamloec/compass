import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

""" #################################################################

### TODO
The goal with class summaries is to summarize the parent classes, and then pass that summary to the child class when summarizing.


# Test Repositories
# https://github.com/VikramBombhi/Checkers.git
# https://github.com/jironghuang/chess.git

    ################################################################# """


"""Test code for compass, feature agent, and test case agent"""
from compass import Compass
from compass.utils import write_dict_to_file
from compass.vector_store import VectorStore
from compass.agents.forwards_feature_agent import ForwardsFeatureAgent
from compass.agents.backwards_feature_agent import BackwardsFeatureAgent

from langchain.chains.sequential import SequentialChain

# Compass
compass = Compass(dir_path="test_repos/Checkers")

vector_store = VectorStore.from_compass(compass, persist=True)

forwards_feature_agent = ForwardsFeatureAgent.as_chain(vector_store)

feature_dict = forwards_feature_agent.invoke({})

for feature_name, feature_dict in feature_dict.items():
    for feature_type, feature_data in feature_dict.items():
        print(feature_type)
        print(feature_data)
