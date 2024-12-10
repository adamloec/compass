import os
from dotenv import load_dotenv
import openai

from compass import Compass
from compass.feature_retriever import CompassFeatureRetriever

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def write_results_to_file(file_path=None, data_dict=None):
    import json
    with open(file_path, 'w') as file:
        json.dump(data_dict, file, indent=4)

# Testing

# Test Repositories
# https://github.com/VikramBombhi/Checkers.git
# https://github.com/jironghuang/chess.git

# Compass: summaries
compass_checkers = Compass(dir_path="test_repos/chess")

write_results_to_file(file_path="results/compass_summaries.json", data_dict=compass_checkers.method_summaries)

# Compass Feature extraction: identifies N number of clusters given compass_summaries, generates N number of features based on chunk data
features = CompassFeatureRetriever.get_features(compass=compass_checkers, known_features=["piece", "board"])

write_results_to_file(file_path="results/features_and_methods.json", data_dict=features)