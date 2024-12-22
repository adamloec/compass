# compass.py

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

        self._method_call_dict = {}  # method_name -> set(of called methods)
        self._method_code_dict = {}  # method_name -> code
        self._file_methods_dict = {} # filename -> { method_name -> code }
        self._class_inheritance = {} # new: store class inheritance, e.g. { "Rook": set(["Piece"]) }

        # new: store file-level summaries
        self.file_summaries = {}     
        self.method_summaries = {}
    
        self.model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self._build()

    def _parse_file(self, file_path, file_extension):
        parser = PARSERS.get(file_extension)
        if not parser:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # We'll store the entire file code for file-level summary
        self.file_summaries[str(file_path)] = {"code": code}

        tree = parser.parse_code(code)
        methods, calls = parser.extract_methods_and_calls(tree.root_node, code)

        # Merge the parser's discovered inheritance info
        if hasattr(parser, "inheritance_map"):
            for child_class, parents in parser.inheritance_map.items():
                if child_class not in self._class_inheritance:
                    self._class_inheritance[child_class] = set()
                self._class_inheritance[child_class].update(parents)
        
        self._file_methods_dict[str(file_path)] = methods
        for method in methods:
            self._method_code_dict[method] = methods[method]
            if method not in self._method_call_dict:
                self._method_call_dict[method] = set()
            self._method_call_dict[method].update(calls)

    def _summarize_files(self):
        LOGGER.info("Summarizing files from repository.")
        
        lock = threading.Lock()

        def summarize_file(file_path, file_code):
            with lock:
                if "summary" in self.file_summaries[file_path]:
                    return
                
            LOGGER.debug(f"Summarizing file: {file_path}")
            prompt = f"""
                You are a skilled software engineer. Summarize the purpose of this entire file,
                focusing on the major feature or functionality it implements. Keep it high-level.

                File content:
                {file_code}
                Summary:
            """
            summary = self.model.invoke(prompt).content.strip()
            
            with lock:
                self.file_summaries[file_path]["summary"] = summary

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            for file_path, data in self.file_summaries.items():
                code = data["code"]
                futures.append(executor.submit(summarize_file, file_path, code))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    LOGGER.error(f"Error summarizing file: {e}")

        LOGGER.info("File summarization completed.")

    def _summarize_methods(self):
        LOGGER.info("Summarizing methods in repository.")
        lock = threading.Lock()

        def summarize_method(method_name, method_code):
            # First check with lock to avoid race condition
            with lock:
                if method_name in self.method_summaries and "summary" in self.method_summaries[method_name]:
                    return
                # Initialize or update the dictionary atomically
                self.method_summaries[method_name] = {"code": method_code}
            
            LOGGER.debug(f"Summarizing method: {method_name}")
            prompt = f"""
                You are a highly skilled software engineer. Summarize the purpose of this function or method
                in simple, high-level terms. Avoid low-level details; aim for conceptual or user-facing meaning.

                Code:
                {method_code}
                Summary:
            """
            summary = self.model.invoke(prompt).content.strip()
            
            with lock:
                self.method_summaries[method_name]["summary"] = summary

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            for filename, methods in self._file_methods_dict.items():
                for method_name, method_code in methods.items():
                    futures.append(executor.submit(summarize_method, method_name, method_code))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    LOGGER.error(f"Error summarizing method: {e}")

        LOGGER.info("Method summarization completed.")

    def _clear_data(self):
        self._method_call_dict.clear()
        self._method_code_dict.clear()
        self._file_methods_dict.clear()
        self.file_summaries.clear()
        self.method_summaries.clear()
        self._class_inheritance.clear()

    def _build(self) -> None:
        LOGGER.info("Starting compass build process.")
        if not self.dir_path:
            raise ValueError("Directory path not set")

        self._clear_data()

        LOGGER.info("Parsing files from repository.")
        for root, _, files in os.walk(self.dir_path):
            for file in files:
                file_extension = file.split('.')[-1]
                if file_extension in PARSERS:
                    file_path = os.path.join(root, file)
                    try:
                        self._parse_file(file_path, file_extension)
                    except Exception as e:
                        LOGGER.error(f"Error parsing file {file_path}: {e}")
                        continue

        # Summarize files at a high level
        self._summarize_files()
        # Summarize methods
        self._summarize_methods()