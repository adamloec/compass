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

        # Old structures
        self._class_methods_dict = {}  # class_name -> {method_name -> code}
        self._global_methods_dict = {} # method_name -> code

        self.symbol_graph = {}         # symbol -> set(symbol references)
        self.file_summaries = {}       # file_path -> { code, summary }
        self.class_summaries = {}      # class_name -> { summary }
        self.method_summaries = {}     # method_name -> { code, summary }

        self.model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self._build()

    def _clear_data(self):
        self._class_methods_dict.clear()
        self._global_methods_dict.clear()
        self.symbol_graph.clear()
        self.file_summaries.clear()
        self.class_summaries.clear()
        self.method_summaries.clear()

    def _parse_file(self, file_path, file_extension):
        parser = PARSERS.get(file_extension)
        if not parser:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        self.file_summaries[str(file_path)] = {"code": code}

        tree = parser.parse_code(code)
        class_methods, global_methods, symbol_graph = parser.extract_symbols_and_calls(tree.root_node, code)

        # Merge class methods
        for cname, cdict in class_methods.items():
            if cname not in self._class_methods_dict:
                self._class_methods_dict[cname] = {}
            self._class_methods_dict[cname].update(cdict)

        # Merge global methods
        self._global_methods_dict.update(global_methods)

        # Merge symbol graphs
        for symbol, refs in symbol_graph.items():
            if symbol not in self.symbol_graph:
                self.symbol_graph[symbol] = set()
            self.symbol_graph[symbol].update(refs)

    def _summarize_files(self):
        LOGGER.info("Summarizing files from repository.")
        lock = threading.Lock()

        def summarize_file(file_path, file_code):
            with lock:
                if "summary" in self.file_summaries[file_path]:
                    return
                
            LOGGER.debug(f"Summarizing file: {file_path}")
            prompt = f"""
                You are a skilled software engineer. Summarize the purpose of this file in a high-level way:
                {file_code}
                
                Summary:
            """
            summary = self.model.invoke(prompt).content.strip()
            with lock:
                self.file_summaries[file_path]["summary"] = summary

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            for fpath, data in self.file_summaries.items():
                code = data["code"]
                futures.append(executor.submit(summarize_file, fpath, code))
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    LOGGER.error(f"Error summarizing file: {e}")

        LOGGER.info("File summarization completed.")

    def _summarize_classes(self):
        LOGGER.info("Summarizing classes in repository.")
        lock = threading.Lock()

        def summarize_class(cname, methods_dict):
            # Combine method code
            all_code = []
            for m in methods_dict:
                all_code.append(methods_dict[m])
            joined = "\n\n".join(all_code)

            with lock:
                if cname not in self.class_summaries:
                    self.class_summaries[cname] = {}
            
            LOGGER.debug(f"Summarizing class: {cname}")
            prompt = f"""
                Summarize the purpose of class {cname}, 
                given these method definitions:
                {joined}

                Provide a conceptual, high-level description:
            """
            summary = self.model.invoke(prompt).content.strip()

            with lock:
                self.class_summaries[cname]["summary"] = summary

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            for cname, methods_dict in self._class_methods_dict.items():
                futures.append(executor.submit(summarize_class, cname, methods_dict))

            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    LOGGER.error(f"Error summarizing class: {e}")

        LOGGER.info("Class summarization completed.")

    def _summarize_methods(self):
        """
        Summarize each method, including a structural fingerprint from symbol_graph.
        """
        LOGGER.info("Summarizing methods in repository.")
        lock = threading.Lock()

        def build_structural_fingerprint(method_name):
            """
            For usage-based context: gather direct references from this symbol, 
            plus who references it, to form a usage fingerprint.
            """
            if method_name not in self.symbol_graph:
                return "No references known."

            # direct references
            out = []
            out.append(f"References: {', '.join(sorted(self.symbol_graph[method_name]))}")

            # who references me
            ref_me = []
            for sym, targets in self.symbol_graph.items():
                if method_name in targets:
                    ref_me.append(sym)
            if ref_me:
                out.append(f"Referenced by: {', '.join(sorted(ref_me))}")
            return "\n".join(out)

        def summarize_method(method_name, method_code):
            structural_fp = build_structural_fingerprint(method_name)

            with lock:
                if method_name in self.method_summaries and "summary" in self.method_summaries[method_name]:
                    return
                self.method_summaries[method_name] = {"code": method_code}

            LOGGER.debug(f"Summarizing method: {method_name}")
            prompt = f"""
                You are a highly skilled software engineer. Summarize the purpose of this method in high-level terms.
                Then factor in the following usage context to refine your summary:
                {structural_fp}

                Method code:
                {method_code}

                Summary:
            """
            summary = self.model.invoke(prompt).content.strip()

            with lock:
                self.method_summaries[method_name]["summary"] = summary
                self.method_summaries[method_name]["structural_fingerprint"] = structural_fp

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            # Summarize class methods
            for cname, methods_dict in self._class_methods_dict.items():
                for mname, mcode in methods_dict.items():
                    futures.append(executor.submit(summarize_method, f"{cname}::{mname}", mcode))
            # Summarize global methods
            for gm, gcode in self._global_methods_dict.items():
                futures.append(executor.submit(summarize_method, gm, gcode))

            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    LOGGER.error(f"Error summarizing method: {e}")

        LOGGER.info("Method summarization completed.")

    def _build(self):
        LOGGER.info("Starting compass build process.")
        if not self.dir_path:
            raise ValueError("Directory path not set")

        self._clear_data()

        LOGGER.info("Parsing files from repository.")
        for root, _, files in os.walk(self.dir_path):
            for file in files:
                file_ext = file.split('.')[-1]
                if file_ext in PARSERS:
                    file_path = os.path.join(root, file)
                    try:
                        self._parse_file(file_path, file_ext)
                    except Exception as e:
                        LOGGER.error(f"Error parsing file {file_path}: {e}")
                        continue

        # Summaries
        self._summarize_files()
        self._summarize_classes()
        self._summarize_methods()