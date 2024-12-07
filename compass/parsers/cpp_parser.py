import tree_sitter_cpp

from .base_parser import BaseParser

class CppParser(BaseParser):

    @classmethod
    def _get_language(cls):

        return tree_sitter_cpp.language()

    def _process_node(self, node, methods, calls, code):

        if node.type in {'function_definition', 'declaration'}:
            func_name = self._extract_function_name(node)
            if func_name:
                methods[func_name] = code[node.start_byte:node.end_byte]
        elif node.type == 'call_expression':
            self._handle_call(node, calls)

    def _extract_function_name(self, node):

        declarator = None
        if node.type == 'function_definition':
            declarator = node.child_by_field_name('declarator')
        elif node.type == 'declaration':
            declarator_node = node.child_by_field_name('declarator')
            if declarator_node and declarator_node.type == 'function_declarator':
                declarator = declarator_node
        return self._get_name_from_declarator(declarator) if declarator else None

    def _get_name_from_declarator(self, node):

        if not node:
            return None
        if node.type == 'function_declarator':
            declarator = node.child_by_field_name('declarator')
            return self._get_name_from_declarator(declarator)
        elif node.type == 'identifier':
            return node.text.decode('utf-8')
        
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8')
        return None

    def _handle_call(self, node, calls):
        
        func_node = node.child_by_field_name('function')
        if func_node:
            if func_node.type == 'identifier':
                calls.append(func_node.text.decode('utf-8'))
            elif func_node.type == 'field_expression':
                field = func_node.child_by_field_name('field')
                if field:
                    calls.append(field.text.decode('utf-8'))