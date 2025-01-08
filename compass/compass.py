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
        self._file_classes_dict = {} # filename -> { class_name -> code }
        self._class_methods_dict = {} # class_name -> { method_name -> code }
        
        self.file_summaries = {}     
        self.method_summaries = {}
        self.class_summaries = {}

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
        methods, calls, classes = parser.extract_methods_and_calls(tree.root_node, code)
        
        # Update class methods dictionary
        for class_name, class_methods in classes.items():
            if class_name not in self._class_methods_dict:
                self._class_methods_dict[class_name] = {}
            self._class_methods_dict[class_name].update(class_methods)
        
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
            # Iterate through class_methods_dict instead of file_methods_dict
            for class_name, methods in self._class_methods_dict.items():
                for method_name, method_code in methods.items():
                    futures.append(executor.submit(summarize_method, method_name, method_code))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    LOGGER.error(f"Error summarizing method: {e}")

        LOGGER.info("Method summarization completed.")

    def _summarize_classes(self):
        LOGGER.info("Summarizing classes in repository.")
        lock = threading.Lock()

        def summarize_class(class_name, methods):
            with lock:
                if class_name in self.class_summaries and "summary" in self.class_summaries[class_name]:
                    return
                self.class_summaries[class_name] = {}

            # Collect all method summaries for this class
            method_summaries = []
            for method_name in methods.keys():
                if method_name in self.method_summaries:
                    method_summaries.append(f"- {method_name}: {self.method_summaries[method_name]['summary']}")

            LOGGER.debug(f"Summarizing class: {class_name}")
            prompt = f"""
                You are a highly skilled software engineer. Summarize the purpose and functionality of this class
                based on its methods and their behaviors. Focus on the class's role in the system and its key capabilities.

                Class: {class_name}
                Methods and their purposes:
                {chr(10).join(method_summaries)}

                Summary:
            """
            summary = self.model.invoke(prompt).content.strip()
            
            with lock:
                self.class_summaries[class_name]["summary"] = summary
                self.class_summaries[class_name]["methods"] = methods

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            for class_name, methods in self._class_methods_dict.items():
                futures.append(executor.submit(summarize_class, class_name, methods))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    LOGGER.error(f"Error summarizing class: {e}")

        LOGGER.info("Class summarization completed.")

    def _clear_data(self):
        self._method_call_dict.clear()
        self._method_code_dict.clear()

        self.file_summaries.clear()
        self.method_summaries.clear()
        self.class_summaries.clear()

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

        # Summarize files
        self._summarize_files()
        # Summarize methods
        self._summarize_methods()
        # Summarize classes
        self._summarize_classes()
