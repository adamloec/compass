import os
import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .parsers import PARSERS

from .logger import Logger
LOGGER = Logger.create(__name__)

class Compass:
    def __init__(self, dir_path=None):
        self.dir_path = Path(dir_path) if dir_path else None

        self._method_call_dict = {}
        self._method_code_dict = {}
        self._file_methods_dict = {}

        self.method_summaries = {}

        self.model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self._build()

    def _parse_file(self, file_path, file_extension):

        parser = PARSERS.get(file_extension)
        if not parser:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        tree = parser.parse_code(code)
        methods, calls = parser.extract_methods_and_calls(tree.root_node, code)

        self._file_methods_dict[str(file_path)] = methods
        for method in methods:
            self._method_code_dict[method] = methods[method]
            if method not in self._method_call_dict:
                self._method_call_dict[method] = set()
            self._method_call_dict[method].update(calls)
    
    def _summarize(self):
        LOGGER.info("Summarizing code from repository.")
        
        for filename, methods in self._file_methods_dict.items():
            for method_name, method_code in methods.items():

                if method_name in self._method_code_dict:
                    existing_code = self._method_code_dict[method_name]
                    if isinstance(existing_code, str):
                        self.method_summaries[method_name] = {"code": existing_code}
                    else:
                        self.method_summaries[method_name] = existing_code
                else:
                    self.method_summaries[method_name] = {"code": method_code}

                if "summary" not in self.method_summaries[method_name]:
                    LOGGER.debug(f"Generating summary for method: {method_name}")
                    prompt = f"""
                            You are a highly skilled software engineer. Given the following code, provide a short, high-level summary of what this function or method does in simple terms. 
                            Focus on the underlying feature or component it relates to, not just the code itself.

                            Code:
                            {self.method_summaries[method_name]['code']}
                            Summary:
                            """
                    summary = self.model.invoke(prompt).content.strip()
                    self.method_summaries[method_name]["summary"] = summary

    def _multiprocess_summarize(self):

        LOGGER.info("Summarizing code from repository.")
        lock = threading.Lock()

        def summarize_method(method_name, method_code):
            if method_name in self._method_code_dict:
                existing_code = self._method_code_dict[method_name]
                if isinstance(existing_code, str):
                    code_to_summarize = existing_code
                else:
                    code_to_summarize = existing_code
            else:
                code_to_summarize = method_code

            if "summary" not in self.method_summaries.get(method_name, {}):
                LOGGER.debug(f"Generating summary for method: {method_name}")
                prompt = f"""
                    You are a highly skilled software engineer. Given the following code, provide a short, high-level summary of what this function or method does in simple terms. 
                    Focus on the underlying feature or component it relates to, not just the code itself.

                    Code:
                    {code_to_summarize}
                    Summary:
                """
                summary = self.model.invoke(prompt).content.strip()

                with lock:
                    self.method_summaries[method_name] = {"code": code_to_summarize, "summary": summary}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            for filename, methods in self._file_methods_dict.items():
                for method_name, method_code in methods.items():
                    futures.append(executor.submit(summarize_method, method_name, method_code))

            for future in as_completed(futures):
                try:
                    future.result()  # Raise exceptions if any
                except Exception as e:
                    LOGGER.error(f"Error summarizing a method: {e}")

        LOGGER.info("Code summarization completed.")

    def _build(self) -> None:
        LOGGER.info("Starting compass build process.")
        if not self.dir_path:
            raise ValueError("Directory path not set")

        self._method_call_dict.clear()
        self._method_code_dict.clear()
        self._file_methods_dict.clear()

        LOGGER.info("Parsing files from repository.")
        
        for root, _, files in os.walk(self.dir_path):
            for file in files:
                file_extension = file.split('.')[-1]
                if file_extension in PARSERS:
                    file_path = os.path.join(root, file)
                    self._parse_file(file_path, file_extension)

        self._multiprocess_summarize()