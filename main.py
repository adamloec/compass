import os
from dotenv import load_dotenv
import openai

from compass import Compass
from compass.feature_retriever import CompassFeatureRetriever

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Compass: summaries
compass_checkers = Compass(dir_path="test_repos/Checkers")

# Feature extraction: identifies N number of features given compass_summaries
features = CompassFeatureRetriever.get_features(compass=compass_checkers, known_features=["board", "player", "AI", "menu"])