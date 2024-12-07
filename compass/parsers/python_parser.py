import tree_sitter_python

from .base_parser import BaseParser

class PythonParser(BaseParser):

    @classmethod
    def _get_language(cls):
        
        return tree_sitter_python.language()

    def _process_node(self, node, methods, calls, code):

        if node.type == 'function_definition':
            func_name = node.child_by_field_name('name').text.decode('utf-8')
            methods[func_name] = code[node.start_byte:node.end_byte]
        elif node.type == 'call':
            func_name_node = node.child_by_field_name('function')
            if func_name_node:
                calls.append(func_name_node.text.decode('utf-8'))