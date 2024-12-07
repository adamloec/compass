import tree_sitter_javascript

from .base_parser import BaseParser

class JavaScriptParser(BaseParser):

    @classmethod
    def _get_language(cls):
        
        return tree_sitter_javascript.language()

    def _process_node(self, node, methods, calls, code):

        if node.type == 'function_declaration':
            name_node = node.child_by_field_name('name')
            if name_node:
                methods[name_node.text.decode('utf-8')] = code[node.start_byte:node.end_byte]
        elif node.type == 'call_expression':
            func_node = node.child_by_field_name('function')
            if func_node:
                if func_node.type == 'identifier':
                    calls.append(func_node.text.decode('utf-8'))
                elif func_node.type == 'member_expression':
                    property_node = func_node.child_by_field_name('property')
                    if property_node:
                        calls.append(property_node.text.decode('utf-8'))