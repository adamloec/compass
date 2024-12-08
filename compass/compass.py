import os
import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

from .parsers import PARSERS

# TODO
# Add summarizations here, summarize each function in a file and store in json data, export to file. Do all of this in build func

class Compass:
    def __init__(self, dir_path=None):
        
        self.dir_path = Path(dir_path) if dir_path else None

        self.method_call_dict = {}
        self.method_code_dict = {}
        self.file_methods_dict = {}

        self._build()

    def _parse_file(self, file_path, file_extension):

        parser = PARSERS.get(file_extension)
        if not parser:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        tree = parser.parse_code(code)
        methods, calls = parser.extract_methods_and_calls(tree.root_node, code)

        self.file_methods_dict[str(file_path)] = methods
        for method in methods:
            self.method_code_dict[method] = methods[method]
            if method not in self.method_call_dict:
                self.method_call_dict[method] = set()
            self.method_call_dict[method].update(calls)

    def _build(self) -> None:

        if not self.dir_path:
            raise ValueError("Directory path not set")

        self.method_call_dict.clear()
        self.method_code_dict.clear()
        self.file_methods_dict.clear()

        for root, _, files in os.walk(self.dir_path):
            for file in files:
                file_extension = file.split('.')[-1]
                if file_extension in PARSERS:
                    file_path = os.path.join(root, file)
                    print(f"Parsing {file_path}")
                    self._parse_file(file_path, file_extension)

        self.dicts_to_json(self.method_call_dict, "method_call_dict.json")
        self.dicts_to_json(self.method_code_dict, "method_code_dict.json")
        self.dicts_to_json(self.file_methods_dict, "file_methods_dict.json")

    def dicts_to_json(self, dicts_list, filename):
        
        def default_serializer(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(filename, 'w') as f:
            json.dump(dicts_list, f, indent=4, default=default_serializer)