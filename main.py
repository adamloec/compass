import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# from compass import Compass

# c = Compass(dir_path="compass/test_data/Checkers")
# c.build()