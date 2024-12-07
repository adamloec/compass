from .python_parser import PythonParser
from .cpp_parser import CppParser
from .javascript_parser import JavaScriptParser

__all__ = [PythonParser, JavaScriptParser, CppParser]

PARSERS = {
    "py": PythonParser(),
    "js": JavaScriptParser(),
    "cpp": CppParser(),
    "h": CppParser(),
}