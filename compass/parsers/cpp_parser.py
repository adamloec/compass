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
         - class_specifier -> detect class methods
        """
        # 3) Class specifiers (process these first to ensure proper context)
        if node.type == 'class_specifier':
            class_name = self._handle_class_specifier(node, code)
            if class_name:
                # Initialize empty dict for class methods if not exists
                if class_name not in self.class_methods:
                    self.class_methods[class_name] = {}
                
                # Look for the class body
                body = node.child_by_field_name('body')
                if body:
                    # Process all method declarations and definitions in the class body
                    for child in body.children:
                        # Check for both function definitions and declarations
                        if child.type in {'function_definition', 'declaration', 'field_declaration'}:
                            method_name = self._extract_function_name(child)
                            if method_name:
                                # Get the full method text
                                method_text = code[child.start_byte:child.end_byte]
                                self.class_methods[class_name][method_name] = method_text

        # 1) Function definitions (outside of classes or class method implementations)
        elif node.type == 'function_definition':
            func_name = self._extract_function_name(node)
            if func_name:
                # Check if this is a class method implementation
                scope = self._get_scope(node)
                if scope:
                    class_name = scope
                    if class_name not in self.class_methods:
                        self.class_methods[class_name] = {}
                    self.class_methods[class_name][func_name] = code[node.start_byte:node.end_byte]
                else:
                    methods[func_name] = code[node.start_byte:node.end_byte]

        # 2) Call expressions
        elif node.type == 'call_expression':
            self._handle_call(node, calls)

    def _extract_function_name(self, node):
        """
        Return the function name if we detect a function_definition or function_declarator.
        """
        if node.type == 'function_definition':
            declarator = node.child_by_field_name('declarator')
        elif node.type == 'declaration':
            declarator = node.child_by_field_name('declarator')
            if declarator and declarator.type == 'function_declarator':
                return self._get_name_from_declarator(declarator)
            return None
        else:
            declarator = None

        # Handle function definitions
        if declarator:
            # Try to get the name directly from the declarator
            name = self._get_name_from_declarator(declarator)
            if name:
                return name

            # If that fails, try to find the identifier in the children
            for child in declarator.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
                elif child.type == 'function_declarator':
                    name = self._get_name_from_declarator(child)
                    if name:
                        return name

        return None

    def _get_name_from_declarator(self, node):
        """
        Recursively find the 'identifier' in a function_declarator chain.
        """
        if not node:
            return None

        # Direct identifier check
        if node.type == 'identifier':
            return node.text.decode('utf-8')

        # Handle function declarators
        if node.type == 'function_declarator':
            declarator = node.child_by_field_name('declarator')
            if declarator:
                return self._get_name_from_declarator(declarator)

        # Check all children for identifiers
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8')
            elif child.type in {'function_declarator', 'pointer_declarator', 'qualified_identifier'}:
                name = self._get_name_from_declarator(child)
                if name:
                    return name

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
        Extract the class name from a class specifier node
        """
        class_name_node = node.child_by_field_name('name')
        if not class_name_node:
            return None

        return class_name_node.text.decode('utf-8')

    def _get_scope(self, node):
        """
        Check if a function definition has a class scope (e.g., ClassName::method)
        """
        declarator = node.child_by_field_name('declarator')
        if declarator:
            for child in declarator.children:
                if child.type == 'qualified_identifier':
                    # Look for the class name in the qualified identifier
                    for scope_child in child.children:
                        if scope_child.type == 'namespace_identifier':
                            return scope_child.text.decode('utf-8')
        return None