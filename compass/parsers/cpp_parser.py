import tree_sitter_cpp
from .base_parser import BaseParser

class CppParser(BaseParser):

    @classmethod
    def _get_language(cls):
        return tree_sitter_cpp.language()

    def _process_node(self, node, code: str):
        node_type = node.type

        # 1) Class declarations
        if node_type == 'class_specifier':
            class_name = self._extract_class_name(node, code)
            if class_name:
                if class_name not in self.class_methods:
                    self.class_methods[class_name] = {}
                if class_name not in self.symbol_graph:
                    self.symbol_graph[class_name] = set()
                # Also parse the class body to find method declarations
                body_node = node.child_by_field_name('body')
                if body_node:
                    for child in body_node.children:
                        if child.type in {'function_definition','declaration'}:
                            mname, mcode = self._extract_method(child, code, class_name)
                            if mname:
                                self.class_methods[class_name][mname] = mcode
                                # record that class references this method symbol
                                full_method_symbol = f"{class_name}::{mname}"
                                if not class_name in self.symbol_graph:
                                    self.symbol_graph[class_name] = set()
                                self.symbol_graph[class_name].add(full_method_symbol)

        # 2) Top-level function definitions
        elif node_type == 'function_definition':
            mname, mcode = self._extract_method(node, code, None)
            if mname:
                self.global_methods[mname] = mcode
                # record in symbol graph
                if mname not in self.symbol_graph:
                    self.symbol_graph[mname] = set()

        # 3) Call expressions
        elif node_type == 'call_expression':
            func_node = node.child_by_field_name('function')
            if func_node:
                callee_symbol = self._extract_callee(func_node, code)
                if callee_symbol:
                    # We need a "from_symbol" context. For simplicity, assume we're in "global" if no class context is found
                    from_symbol = self._current_scope_symbol(node)
                    if not from_symbol:
                        from_symbol = "Global::(unknown)"  
                    self._add_reference(from_symbol, callee_symbol)

    def _extract_class_name(self, node, code):
        name_node = node.child_by_field_name('name')
        if name_node:
            return name_node.text.decode('utf-8')
        return None

    def _extract_method(self, node, code, class_name: str = None):
        """
        Returns (method_name, method_code).
        If class_name is provided, the method belongs to that class.
        """
        mcode = code[node.start_byte:node.end_byte]
        # For simplicity, let's say the method_name is any identifier child with type 'function_declarator'
        # or we can do something more robust.
        mname = None
        decl_node = node.child_by_field_name('declarator')
        if decl_node:
            # find the identifier
            mname = self._find_identifier(decl_node)
        if not mname:
            return None, None
        return (mname, mcode)

    def _find_identifier(self, node):
        """Recursively look for an 'identifier' node."""
        if node.type == 'identifier':
            return node.text.decode('utf-8')
        for child in node.children:
            res = self._find_identifier(child)
            if res:
                return res
        return None

    def _extract_callee(self, node, code):
        """
        For a call_expression -> (function) child, if it's an identifier, we treat that as e.g. "myFunction".
        If it's something like 'object.method', we might parse it as "ClassName::method" if we can guess the class name.
        For now, just return the text.
        """
        if node.type == 'identifier':
            return node.text.decode('utf-8')
        elif node.type == 'field_expression':
            # e.g. object.method()
            field_child = node.child_by_field_name('field')
            if field_child and field_child.type == 'identifier':
                return field_child.text.decode('utf-8')
        return None

    def _current_scope_symbol(self, node):
        """
        Attempt to find the method or class scope in which this node lives.
        For brevity, we won't do a full AST walk upward. 
        A robust approach might climb up the tree to find an enclosing function_definition or class_specifier.
        """
        # This is quite naive:
        parent = node.parent
        while parent:
            if parent.type == 'function_definition':
                # get function name
                decl_node = parent.child_by_field_name('declarator')
                if decl_node:
                    fn = self._find_identifier(decl_node)
                    if fn:
                        return fn
            if parent.type == 'class_specifier':
                cname = self._extract_class_name(parent, "")
                return cname
            parent = parent.parent
        return None