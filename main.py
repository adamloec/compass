import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
from compass.utils import write_dict_to_file
from compass.vector_store import VectorStore
from compass.agents.forwards_feature_agent import ForwardsFeatureAgent
from compass.agents.backwards_feature_agent import BackwardsFeatureAgent

from langchain.chains.sequential import SequentialChain

# Compass
compass = Compass(dir_path="test_repos/chess")

# Combine all 3 summary types into dict with seperators inbetween each summary type
# Add separator entries between summary types
summary_dict = {
    "FILE_SUMMARIES_START": "=" * 50,
    **compass.file_summaries,
    "FILE_SUMMARIES_END": "=" * 50,
    "CLASS_SUMMARIES_START": "=" * 50, 
    **compass.class_summaries,
    "CLASS_SUMMARIES_END": "=" * 50,
    "METHOD_SUMMARIES_START": "=" * 50,
    **compass.method_summaries,
    "METHOD_SUMMARIES_END": "=" * 50
}
write_dict_to_file(file_path="compass_summaries.json", data_dict=summary_dict)