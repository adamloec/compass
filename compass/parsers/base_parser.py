# base_parser.py

from tree_sitter import Parser, Language

class BaseParser:
    def __init__(self):
        """
        A base parser that sets up a tree-sitter parser with a chosen language.
        Child classes must implement _get_language().
        """
        language = Language(self._get_language())
        self.parser = Parser(language)
        self.class_methods = {}  # New: maps class names to their methods

    @classmethod
    def _get_language(cls):
        raise NotImplementedError

    def parse_code(self, code: str):
        """
        Parse the raw code string into a parse tree.
        """
        return self.parser.parse(bytes(code, 'utf8'))

    def extract_methods_and_calls(self, node, code: str):
        """
        Walk the AST, discover:
        - function definitions/declarations -> `methods`
        - function calls -> `calls`
        - class methods -> stored in `self.class_methods`
        Child classes can also detect inheritance in `self.inheritance_map`.
        """
        methods = {}
        calls = []
        self.class_methods = {}  # Reset for each new parse
        self._traverse(node, methods, calls, code)
        return methods, calls, self.class_methods

    def _traverse(self, node, methods, calls, code):
        """
        Recursively traverse the AST, calling _process_node on each node.
        """
        self._process_node(node, methods, calls, code)
        for child in node.children:
            self._traverse(child, methods, calls, code)

    def _process_node(self, node, methods, calls, code):
        """
        Must be implemented by child classes to detect function definitions,
        calls, and optionally inheritance or anything else.
        """
        raise NotImplementedError