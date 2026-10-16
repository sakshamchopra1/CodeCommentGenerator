"""
Root re-export of security module for direct imports.
Exposes AgenticSecurityLayer, SecurityException, AuditLogEntry, and SecurityDecision.
"""
import sys
import os

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from src.security import (
    AgenticSecurityLayer,
    SecurityDecision,
    SecurityException,
    AuditLogEntry,
    SecurityPattern,
    DEFAULT_DIRECT_PATTERNS,
    DEFAULT_INDIRECT_PATTERNS,
)

__all__ = [
    "AgenticSecurityLayer",
    "SecurityDecision",
    "SecurityException",
    "AuditLogEntry",
    "SecurityPattern",
    "DEFAULT_DIRECT_PATTERNS",
    "DEFAULT_INDIRECT_PATTERNS",
]
