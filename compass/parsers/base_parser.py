from tree_sitter import Parser, Language

class BaseParser:
    def __init__(self):
        """
        A base parser that sets up a tree-sitter parser with a chosen language.
        Child classes must implement _get_language() returning a compiled tree-sitter Language.
        """
        language = Language(self._get_language())
        self.parser = Parser(language)

        # Class -> { method_name -> code }
        self.class_methods = {}
        # Map symbol -> set(symbols it references)
        # e.g. "Player" -> {"Board", "Pieces"}
        self.symbol_graph = {}
        # Keep track of all top-level functions: method_name -> code
        self.global_methods = {}

    @classmethod
    def _get_language(cls):
        raise NotImplementedError

    def parse_code(self, code: str):
        """
        Parse the raw code string into a parse tree.
        """
        return self.parser.parse(bytes(code, 'utf8'))

    def extract_symbols_and_calls(self, node, code: str):
        """
        Instead of just returning methods/calls/class_methods, we fill:
          - self.class_methods
          - self.global_methods
          - self.symbol_graph  (mapping symbol -> references)
        """
        self.class_methods = {}
        self.global_methods = {}
        self.symbol_graph = {}
        self._traverse(node, code)
        return self.class_methods, self.global_methods, self.symbol_graph

    def _traverse(self, node, code: str):
        self._process_node(node, code)
        for child in node.children:
            self._traverse(child, code)

    def _process_node(self, node, code: str):
        """
        Child classes implement detection logic for:
          - classes
          - methods
          - calls
          - references
        """
        raise NotImplementedError

    def _add_reference(self, from_symbol: str, to_symbol: str):
        """
        Utility to record an edge in the symbol graph
        """
        if not from_symbol in self.symbol_graph:
            self.symbol_graph[from_symbol] = set()
        self.symbol_graph[from_symbol].add(to_symbol)