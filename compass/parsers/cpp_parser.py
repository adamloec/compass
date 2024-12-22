# cpp_parser.py

import tree_sitter_cpp
from .base_parser import BaseParser

class CppParser(BaseParser):
    @classmethod
    def _get_language(cls):
        """
        Return the compiled tree-sitter C++ language object.
        """
        return tree_sitter_cpp.language()

    def _process_node(self, node, methods, calls, code):
        """
        Called during AST traversal. We detect:
         - function_definition / declaration -> store in `methods`
         - call_expression -> store in `calls`
         - class_specifier -> detect inheritance
        """

        # 1) Function definitions or declarations
        if node.type in {'function_definition', 'declaration'}:
            func_name = self._extract_function_name(node)
            if func_name:
                methods[func_name] = code[node.start_byte: node.end_byte]

        # 2) Call expressions
        elif node.type == 'call_expression':
            self._handle_call(node, calls)

        # 3) Class specifiers (e.g. "class Rook : public Piece")
        if node.type == 'class_specifier':
            self._handle_class_specifier(node, code)

    def _extract_function_name(self, node):
        """
        Return the function name if we detect a function_definition or function_declarator.
        """
        declarator = None
        if node.type == 'function_definition':
            declarator = node.child_by_field_name('declarator')
        elif node.type == 'declaration':
            declarator_node = node.child_by_field_name('declarator')
            if declarator_node and declarator_node.type == 'function_declarator':
                declarator = declarator_node
        return self._get_name_from_declarator(declarator) if declarator else None

    def _get_name_from_declarator(self, node):
        """
        Recursively find the 'identifier' in a function_declarator chain.
        """
        if not node:
            return None
        if node.type == 'function_declarator':
            sub_decl = node.child_by_field_name('declarator')
            return self._get_name_from_declarator(sub_decl)
        elif node.type == 'identifier':
            return node.text.decode('utf-8')

        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8')
        return None

    def _handle_call(self, node, calls):
        """
        If we see something like foo(), store 'foo' in calls.
        If it's something like obj.foo(), store 'foo' as well.
        """
        func_node = node.child_by_field_name('function')
        if func_node:
            if func_node.type == 'identifier':
                calls.append(func_node.text.decode('utf-8'))
            elif func_node.type == 'field_expression':
                field = func_node.child_by_field_name('field')
                if field:
                    calls.append(field.text.decode('utf-8'))

    def _handle_class_specifier(self, node, code):
        """
        Detect 'class Rook : public Piece' or multiple parents, store them in self.inheritance_map
        e.g. self.inheritance_map["Rook"] = set(["Piece"])
        """
        class_name_node = node.child_by_field_name('name')
        if not class_name_node:
            return

        child_class = class_name_node.text.decode('utf-8')

        # base_class_clause might be a child_by_field_name('base_class_clause')
        base_clause = node.child_by_field_name('base_class_clause')
        if not base_clause:
            return  # no inheritance

        # Within base_class_clause, we can look for 'type_identifier' nodes
        parent_classes = set()
        for child in base_clause.children:
            if child.type == 'type_identifier':
                parent_classes.add(child.text.decode('utf-8'))

        if not parent_classes:
            return

        if child_class not in self.inheritance_map:
            self.inheritance_map[child_class] = set()

        self.inheritance_map[child_class].update(parent_classes)