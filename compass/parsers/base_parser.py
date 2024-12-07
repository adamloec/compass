from tree_sitter import Parser, Language

class BaseParser:
    
    def __init__(self):

        language = Language(self._get_language())
        self.parser = Parser(language)

    @classmethod
    def _get_language(cls):

        raise NotImplementedError

    def parse_code(self, code):

        return self.parser.parse(bytes(code, 'utf8'))

    def extract_methods_and_calls(self, node, code):

        methods = {}
        calls = []
        self._traverse(node, methods, calls, code)

        return methods, calls

    def _traverse(self, node, methods, calls, code):

        self._process_node(node, methods, calls, code)
        for child in node.children:
            self._traverse(child, methods, calls, code)

    def _process_node(self, node, methods, calls, code):
        
        raise NotImplementedError