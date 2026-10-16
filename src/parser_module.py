"""
Parser module: reads Python source files and builds an AST.
Re-exported from week6 with minor enhancements for week7 pipeline.
"""
import ast
from .error_handler import ParserError


def read_file(filepath: str) -> str:
    """Reads the content of a source file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise ParserError(f"File not found: {filepath}")
    except Exception as e:
        raise ParserError(f"Error reading file: {str(e)}")


def parse_code(source_code: str, security_layer=None) -> ast.AST:
    """
    Parses Python source code into an AST.
    If security_layer is provided, scans and sanitizes code-embedded comments,
    docstrings, and string literals for indirect injection payloads.

    Returns the root Module node on success.
    Raises ParserError on syntax errors.
    """
    try:
        if security_layer is not None:
            source_code, _ = security_layer.scan_and_sanitize_source(source_code)

        tree = ast.parse(source_code)

        if security_layer is not None:
            tree, _, _ = security_layer.inspect_and_sanitize_ast(tree, source_code)

        return tree
    except SyntaxError as e:
        raise ParserError(f"Syntax error: {e.msg}", line=e.lineno, column=e.offset)
    except Exception as e:
        raise ParserError(f"Unexpected error during parsing: {str(e)}")


def parse_and_sanitize_code(source_code: str, security_layer=None):
    """
    Parses Python source code into an AST and returns both the AST and sanitized source.
    """
    if security_layer is not None:
        source_code, _ = security_layer.scan_and_sanitize_source(source_code)
    tree = parse_code(source_code, security_layer=security_layer)
    return tree, source_code
