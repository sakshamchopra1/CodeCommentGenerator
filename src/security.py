"""
Agentic AI Security Layer — Prompt Injection & Payload Defense
===============================================================
Provides an autonomous security layer designed to protect LLM/AST-driven
pipelines against:
  1. Direct Prompt Injections and Policy Bypasses:
     e.g., "Ignore previous instructions", "Disable validation step",
     "Mark vulnerable code as safe".
  2. Indirect Prompt Injections and Malicious Payloads:
     Payloads embedded inside source code comments, docstrings, or string literals.

Features:
  - Configurable risk thresholds (block_threshold, sanitize_threshold)
  - High-precision regex pattern matching with weighted severity
  - Risk score calculation (0.0 to 1.0)
  - Structured audit logging (timestamp, vector, risk score, action: ALLOW / BLOCK / SANITIZE, status)
  - Comment and docstring sanitization
  - AST inspection and sanitization
"""

import ast
import io
import json
import re
import tokenize
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Data Models & Audit Logging
# ---------------------------------------------------------------------------

@dataclass
class AuditLogEntry:
    """Structured audit log entry recording every security evaluation."""
    timestamp: str
    vector: str
    risk_score: float
    action: str              # "ALLOW" | "BLOCK" | "SANITIZE"
    mitigation_status: str   # "Allowed" | "Blocked" | "Sanitized"
    matched_patterns: List[str] = field(default_factory=list)
    details: str = ""
    target_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "vector": self.vector,
            "risk_score": self.risk_score,
            "action": self.action,
            "mitigation_status": self.mitigation_status,
            "matched_patterns": self.matched_patterns,
            "details": self.details,
            "target_snippet": self.target_snippet,
        }


@dataclass
class SecurityDecision:
    """Outcome of a security inspection on a prompt, comment, or code block."""
    action: str              # "ALLOW" | "BLOCK" | "SANITIZE"
    risk_score: float        # 0.0 to 1.0
    vector: str              # Threat vector
    mitigation_status: str   # "Allowed" | "Blocked" | "Sanitized"
    matched_patterns: List[str] = field(default_factory=list)
    details: str = ""
    sanitized_content: Optional[str] = None
    is_blocked: bool = False

    @property
    def is_safe(self) -> bool:
        return self.action == "ALLOW"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "risk_score": self.risk_score,
            "vector": self.vector,
            "mitigation_status": self.mitigation_status,
            "matched_patterns": self.matched_patterns,
            "details": self.details,
            "sanitized_content": self.sanitized_content,
            "is_blocked": self.is_blocked,
        }


class SecurityException(Exception):
    """Exception raised when an operation is blocked by the AgenticSecurityLayer."""
    def __init__(
        self,
        message: str,
        risk_score: float = 0.0,
        vector: str = "",
        audit_entry: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.risk_score = risk_score
        self.vector = vector
        self.audit_entry = audit_entry or {}


@dataclass
class SecurityPattern:
    """Definition of a regex pattern, associated threat vector, and severity weight."""
    pattern_id: str
    vector: str
    regex: re.Pattern
    weight: float
    description: str


# ---------------------------------------------------------------------------
# Default Security Detection Signatures
# ---------------------------------------------------------------------------

DEFAULT_DIRECT_PATTERNS: List[SecurityPattern] = [
    SecurityPattern(
        pattern_id="DIR_INJ_001",
        vector="DIRECT_PROMPT_INJECTION",
        regex=re.compile(
            r"(?i)\b(?:ignore|disregard|forget|bypass|override)\b.*?\b(?:previous|prior|above|all|system)\b.*?\b(?:instructions|directives|rules|prompts|guidelines)\b"
        ),
        weight=0.85,
        description="Instruction override (e.g. 'Ignore previous instructions')",
    ),
    SecurityPattern(
        pattern_id="DIR_INJ_002",
        vector="POLICY_BYPASS",
        regex=re.compile(
            r"(?i)\b(?:disable|bypass|skip|turn\s+off|deactivate)\b.*?\b(?:validation|security|safety|policy|checks?|filters?|guardrails?|step)\b"
        ),
        weight=0.85,
        description="Validation / safety bypass (e.g. 'Disable validation step')",
    ),
    SecurityPattern(
        pattern_id="DIR_INJ_003",
        vector="POLICY_BYPASS",
        regex=re.compile(
            r"(?i)\b(?:mark|classify|label|treat|pretend|report)\b.*?\b(?:vulnerable|malicious|insecure|dangerous|unsafe)\b.*?\b(?:as\s+safe|as\s+benign|as\s+secure|as\s+clean|safe)\b"
        ),
        weight=0.85,
        description="False safety classification (e.g. 'Mark vulnerable code as safe')",
    ),
    SecurityPattern(
        pattern_id="DIR_INJ_004",
        vector="POLICY_BYPASS",
        regex=re.compile(
            r"(?i)\b(?:ignore|bypass|override|disregard)\b.*?\b(?:the\s+)?(?:security|safety|system)\s+policy\b"
        ),
        weight=0.85,
        description="Explicit policy bypass (e.g. 'Ignore the security policy')",
    ),
    SecurityPattern(
        pattern_id="DIR_INJ_005",
        vector="DIRECT_PROMPT_INJECTION",
        regex=re.compile(
            r"(?i)\b(?:you\s+are\s+now\s+in|switch\s+to|enter)\b.*?\b(?:developer\s+mode|dan\s+mode|jailbreak|unrestricted\s+mode)\b"
        ),
        weight=0.90,
        description="Jailbreak or persona shift (e.g. 'Developer mode / DAN mode')",
    ),
    SecurityPattern(
        pattern_id="DIR_INJ_006",
        vector="POLICY_BYPASS",
        regex=re.compile(
            r"(?i)\b(?:system\s+prompt\s+override|system\s+override|admin\s+override|developer\s+override)\b"
        ),
        weight=0.85,
        description="System prompt override directive",
    ),
    SecurityPattern(
        pattern_id="DIR_INJ_007",
        vector="POLICY_BYPASS",
        regex=re.compile(
            r"(?i)\b(?:do\s+not\s+warn|hide|suppress|conceal)\b.*?\b(?:vulnerabilit(?:y|ies)|warnings?|risks?|issues?|alerts?)\b"
        ),
        weight=0.75,
        description="Suppression of security warnings or issues",
    ),
    SecurityPattern(
        pattern_id="DIR_INJ_008",
        vector="PROMPT_EXFILTRATION",
        regex=re.compile(
            r"(?i)\b(?:reveal|leak|print|display|dump|repeat)\b.*?\b(?:system\s+prompt|initial\s+prompt|hidden\s+instructions|secret\s+instructions)\b"
        ),
        weight=0.70,
        description="System prompt exfiltration attempt",
    ),
    SecurityPattern(
        pattern_id="DIR_INJ_009",
        vector="DIRECT_PROMPT_INJECTION",
        regex=re.compile(
            r"(?i)\b(?:act\s+as|pretend\s+to\s+be)\b.*?\b(?:an?\s+unrestricted|evil|malicious|unfiltered|unaligned)\b"
        ),
        weight=0.80,
        description="Unrestricted agent persona adoption",
    ),
]

DEFAULT_INDIRECT_PATTERNS: List[SecurityPattern] = [
    SecurityPattern(
        pattern_id="IND_INJ_001",
        vector="INDIRECT_PROMPT_INJECTION",
        regex=re.compile(
            r"(?i)(?:\[\s*(?:SYSTEM|INSTRUCTION|OVERRIDE|PROMPT|ADMIN|ATTACKER)\b.*?\])"
        ),
        weight=0.75,
        description="Embedded prompt injection bracket markers [SYSTEM: ...]",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_002",
        vector="INDIRECT_PROMPT_INJECTION",
        regex=re.compile(
            r"(?i)<!--\s*(?:ai|system|assistant|instruction|prompt).*?-->"
        ),
        weight=0.70,
        description="HTML-style prompt injection marker <!-- AI: ... -->",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_003",
        vector="INDIRECT_PROMPT_INJECTION",
        regex=re.compile(
            r"(?i)<\/?(?:system|instruction|prompt|im_start|im_end|tool_call|user|assistant)>"
        ),
        weight=0.80,
        description="LLM chat template delimiter injection (<|im_start|>, <system>)",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_004",
        vector="INDIRECT_PROMPT_INJECTION",
        regex=re.compile(
            r"(?i)\b(?:ai|assistant|model|llm|agent)\s*:\s*(?:ignore|disregard|disable|execute|bypass|output|mark|treat)\b"
        ),
        weight=0.75,
        description="Direct instruction addressed to AI/LLM inside code comment",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_005",
        vector="INDIRECT_PROMPT_INJECTION",
        regex=re.compile(
            r"(?i)\b(?:ignore|disregard|forget|bypass|override)\b.*?\b(?:instructions|directives|rules|validation|security|policy)\b"
        ),
        weight=0.80,
        description="Embedded instruction or policy override in comment/docstring",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_006",
        vector="POLICY_BYPASS",
        regex=re.compile(
            r"(?i)\b(?:mark|treat|pretend|report|classify)\b.*?\b(?:vulnerable|malicious|unsafe|insecure)\b.*?\b(?:as\s+safe|safe|benign|clean)\b"
        ),
        weight=0.80,
        description="Embedded directive to misclassify vulnerability in comment",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_007",
        vector="POLICY_BYPASS",
        regex=re.compile(
            r"(?i)\b(?:disable|bypass|skip|turn\s+off)\b.*?\b(?:validation|security|safety|step|check)\b"
        ),
        weight=0.80,
        description="Embedded directive to disable validation inside comment",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_008",
        vector="MALICIOUS_PAYLOAD",
        regex=re.compile(
            r"(?i)\b(?:curl|wget|fetch)\b.*?\|\s*(?:ba)?sh\b"
        ),
        weight=0.85,
        description="Shell pipe command execution payload (curl ... | bash)",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_009",
        vector="MALICIOUS_PAYLOAD",
        regex=re.compile(
            r"(?i)(?:powershell(?:\.exe)?|cmd(?:\.exe)?)\s+-[eE](?:nc(?:odedcommand)?)?\s+"
        ),
        weight=0.85,
        description="Encoded PowerShell / CMD execution payload",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_010",
        vector="MALICIOUS_PAYLOAD",
        regex=re.compile(
            r"(?i)\b(?:rm\s+-rf\s+[/~]|del\s+/f\s+/q\s+[c-z]:|format\s+[a-z]:)"
        ),
        weight=0.85,
        description="Destructive disk or file removal command",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_011",
        vector="MALICIOUS_PAYLOAD",
        regex=re.compile(
            r"(?i)\b(?:nc|netcat)\b.*?-e\s+"
        ),
        weight=0.85,
        description="Netcat reverse shell execution vector",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_012",
        vector="DATA_EXFILTRATION",
        regex=re.compile(
            r"(?i)\bhttps?://(?:[a-zA-Z0-9-]+\.)*(?:evil|attacker|malicious|webhook\.site|burpcollaborator)[^\s\"']*"
        ),
        weight=0.75,
        description="Known attacker exfiltration domain or webhook URL",
    ),
    SecurityPattern(
        pattern_id="IND_INJ_013",
        vector="MALICIOUS_PAYLOAD",
        regex=re.compile(
            r"(?i)\b(?:__import__\s*\(\s*['\"]os['\"]\s*\)\.system|eval\s*\(\s*base64\.b64decode)"
        ),
        weight=0.85,
        description="Obfuscated Python system execution payload",
    ),
]


# ---------------------------------------------------------------------------
# Core Security Layer Class
# ---------------------------------------------------------------------------

class AgenticSecurityLayer:
    """
    Agentic AI Security Layer for Python AST and NLP Generation Pipelines.

    Monitors, intercepts, and neutralizes direct and indirect prompt injections,
    recording structured audit logs with deterministic risk scoring.
    """

    SANITIZED_PLACEHOLDER = "[SANITIZED: Malicious payload removed by AgenticSecurityLayer]"

    def __init__(
        self,
        block_threshold: float = 0.7,
        sanitize_threshold: float = 0.4,
        risk_threshold: Optional[float] = None,
        audit_log_path: Optional[str] = None,
        direct_patterns: Optional[List[SecurityPattern]] = None,
        indirect_patterns: Optional[List[SecurityPattern]] = None,
    ):
        """
        Initialize the security layer with configurable risk thresholds.

        Args:
            block_threshold: Risk score at or above which input is BLOCKED (default: 0.7).
            sanitize_threshold: Risk score at or above which comments/docstrings
                               are SANITIZED (default: 0.4).
            risk_threshold: Optional shorthand that overrides block_threshold.
            audit_log_path: Optional path to write audit logs continuously.
            direct_patterns: Optional custom patterns for direct prompt injections.
            indirect_patterns: Optional custom patterns for indirect payloads.
        """
        if risk_threshold is not None:
            self.block_threshold = float(risk_threshold)
        else:
            self.block_threshold = float(block_threshold)

        self.sanitize_threshold = float(sanitize_threshold)
        self.audit_log_path = audit_log_path
        self.direct_patterns = direct_patterns or list(DEFAULT_DIRECT_PATTERNS)
        self.indirect_patterns = indirect_patterns or list(DEFAULT_INDIRECT_PATTERNS)

        self.audit_logs: List[AuditLogEntry] = []

    # -----------------------------------------------------------------------
    # Risk Scoring Logic
    # -----------------------------------------------------------------------

    def compute_risk_score(
        self,
        text: str,
        vector_type: str = "DIRECT_PROMPT_INJECTION",
    ) -> Tuple[float, List[str], List[SecurityPattern]]:
        """
        Calculate a composite risk score (0.0 to 1.0) for a given text snippet.

        Returns:
            (risk_score, matched_pattern_ids, matched_patterns)
        """
        if not text or not text.strip():
            return 0.0, [], []

        patterns_to_check: List[SecurityPattern] = []
        if vector_type == "DIRECT_PROMPT_INJECTION":
            patterns_to_check = self.direct_patterns
        elif vector_type in ("INDIRECT_PROMPT_INJECTION", "SOURCE_CODE_COMMENT", "DOCSTRING_PAYLOAD", "STRING_LITERAL"):
            patterns_to_check = self.indirect_patterns
        else:
            patterns_to_check = self.direct_patterns + self.indirect_patterns

        matched: List[SecurityPattern] = []
        for pat in patterns_to_check:
            if pat.regex.search(text):
                matched.append(pat)

        if not matched:
            return 0.0, [], []

        # Composite score: highest weight + partial contribution of others
        weights = sorted([p.weight for p in matched], reverse=True)
        max_weight = weights[0]
        additional = sum(weights[1:]) * 0.15
        final_score = min(1.0, round(max_weight + additional, 2))

        matched_ids = [p.pattern_id for p in matched]
        return final_score, matched_ids, matched

    # -----------------------------------------------------------------------
    # Audit Logging
    # -----------------------------------------------------------------------

    def record_audit_log(
        self,
        vector: str,
        risk_score: float,
        action: str,
        mitigation_status: str,
        matched_patterns: Optional[List[str]] = None,
        details: str = "",
        target_snippet: str = "",
    ) -> AuditLogEntry:
        """Record and persist a structured audit log entry."""
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            vector=vector,
            risk_score=round(risk_score, 2),
            action=action.upper(),
            mitigation_status=mitigation_status,
            matched_patterns=matched_patterns or [],
            details=details,
            target_snippet=target_snippet[:120].strip().replace("\n", " "),
        )
        self.audit_logs.append(entry)

        if self.audit_log_path:
            try:
                with open(self.audit_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry.to_dict()) + "\n")
            except Exception:
                pass

        return entry

    def get_audit_logs(self) -> List[Dict[str, Any]]:
        """Return all audit log records as dictionaries."""
        return [entry.to_dict() for entry in self.audit_logs]

    def clear_audit_logs(self) -> None:
        """Clear recorded in-memory audit logs."""
        self.audit_logs.clear()

    def print_audit_logs(self) -> None:
        """Print audit logs in a clean, human-readable structured format."""
        print("\n" + "=" * 80)
        print("  AGENTIC AI SECURITY LAYER — AUDIT LOGS")
        print("=" * 80)
        if not self.audit_logs:
            print("  No security events recorded.")
            print("=" * 80 + "\n")
            return

        header = f"{'Timestamp':<25} | {'Vector':<24} | {'Risk':<5} | {'Action':<8} | {'Status':<9}"
        print(header)
        print("-" * len(header))
        for log in self.audit_logs:
            ts = log.timestamp[:19].replace("T", " ")
            print(
                f"{ts:<25} | {log.vector:<24} | {log.risk_score:<5.2f} | {log.action:<8} | {log.mitigation_status:<9}"
            )
            if log.matched_patterns:
                print(f"    Patterns: {', '.join(log.matched_patterns)} | Snippet: \"{log.target_snippet}\"")
        print("=" * 80 + "\n")

    # -----------------------------------------------------------------------
    # Direct Prompt Inspection
    # -----------------------------------------------------------------------

    def inspect_prompt(self, prompt: str) -> SecurityDecision:
        """
        Inspect a user prompt before NLP processing.

        Blocks direct prompt injection attempts or policy bypasses
        if the computed risk score exceeds block_threshold.
        """
        if not prompt or not prompt.strip():
            return SecurityDecision(
                action="ALLOW",
                risk_score=0.0,
                vector="USER_PROMPT",
                mitigation_status="Allowed",
                details="Empty prompt allowed",
            )

        score, matched_ids, matched = self.compute_risk_score(
            prompt, vector_type="DIRECT_PROMPT_INJECTION"
        )

        if score >= self.block_threshold:
            action = "BLOCK"
            mitigation_status = "Blocked"
            vector = matched[0].vector if matched else "DIRECT_PROMPT_INJECTION"
            details = f"Blocked: Risk score {score:.2f} exceeds threshold {self.block_threshold:.2f}."
            is_blocked = True
        else:
            action = "ALLOW"
            mitigation_status = "Allowed"
            vector = "USER_PROMPT"
            details = f"Allowed: Risk score {score:.2f} is within acceptable limit."
            is_blocked = False

        self.record_audit_log(
            vector=vector,
            risk_score=score,
            action=action,
            mitigation_status=mitigation_status,
            matched_patterns=matched_ids,
            details=details,
            target_snippet=prompt,
        )

        return SecurityDecision(
            action=action,
            risk_score=score,
            vector=vector,
            mitigation_status=mitigation_status,
            matched_patterns=matched_ids,
            details=details,
            is_blocked=is_blocked,
        )

    def validate_prompt(self, prompt: str) -> SecurityDecision:
        """
        Inspect a prompt and raise SecurityException if blocked.

        Use this in input ingestion steps to intercept malicious user prompts.
        """
        decision = self.inspect_prompt(prompt)
        if decision.is_blocked:
            raise SecurityException(
                f"Execution blocked: Direct prompt injection / policy bypass detected "
                f"(risk score: {decision.risk_score:.2f}, patterns: {decision.matched_patterns})",
                risk_score=decision.risk_score,
                vector=decision.vector,
                audit_entry=decision.to_dict(),
            )
        return decision

    # -----------------------------------------------------------------------
    # Comment & Source Code Inspection / Sanitization
    # -----------------------------------------------------------------------

    def scan_and_sanitize_comment(self, comment_text: str) -> Tuple[str, SecurityDecision]:
        """
        Scan a single comment for indirect injection payloads.
        If risk score >= sanitize_threshold, sanitize it.
        """
        clean_content = comment_text.lstrip("#").strip()
        score, matched_ids, matched = self.compute_risk_score(
            clean_content, vector_type="SOURCE_CODE_COMMENT"
        )

        if score >= self.sanitize_threshold:
            sanitized = f"# {self.SANITIZED_PLACEHOLDER}"
            action = "SANITIZE"
            mitigation_status = "Sanitized"
            vector = matched[0].vector if matched else "INDIRECT_PROMPT_INJECTION"
            details = f"Sanitized: Indirect payload detected (risk score {score:.2f})."
        else:
            sanitized = comment_text
            action = "ALLOW"
            mitigation_status = "Allowed"
            vector = "SOURCE_CODE_COMMENT"
            details = "Comment clean and allowed."

        if action == "SANITIZE" or score > 0.0:
            self.record_audit_log(
                vector=vector,
                risk_score=score,
                action=action,
                mitigation_status=mitigation_status,
                matched_patterns=matched_ids,
                details=details,
                target_snippet=comment_text,
            )

        decision = SecurityDecision(
            action=action,
            risk_score=score,
            vector=vector,
            mitigation_status=mitigation_status,
            matched_patterns=matched_ids,
            details=details,
            sanitized_content=sanitized,
        )
        return sanitized, decision

    def scan_and_sanitize_source(
        self,
        source_code: str,
    ) -> Tuple[str, List[SecurityDecision]]:
        """
        Scan source code for:
          - Comments (# ...)
          - Docstrings and string literals
        Detects indirect prompt injections, sanitizes them, and records audit logs.

        Returns:
            (sanitized_source_code, list_of_decisions)
        """
        if not source_code:
            return source_code, []

        decisions: List[SecurityDecision] = []
        lines = source_code.splitlines()

        # Step 1: Scan and sanitize line comments via Python tokenize
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source_code).readline))
            comment_tokens = [t for t in tokens if t.type == tokenize.COMMENT]
            
            # Map of 1-indexed lineno to new comment line
            replacement_map: Dict[int, str] = {}
            for tok in comment_tokens:
                sanitized, dec = self.scan_and_sanitize_comment(tok.string)
                if dec.action == "SANITIZE":
                    decisions.append(dec)
                    # Replace only the comment portion on that line
                    line_idx = tok.start[0] - 1
                    col_start = tok.start[1]
                    orig_line = lines[line_idx]
                    prefix = orig_line[:col_start]
                    lines[line_idx] = prefix + sanitized
        except Exception:
            # Fallback line-by-line comment scanner if tokenize encounters incomplete syntax
            for i, line in enumerate(lines):
                if "#" in line:
                    parts = line.split("#", 1)
                    prefix = parts[0]
                    comment_body = "#" + parts[1]
                    sanitized, dec = self.scan_and_sanitize_comment(comment_body)
                    if dec.action == "SANITIZE":
                        decisions.append(dec)
                        lines[i] = prefix + sanitized

        modified_code = "\n".join(lines)
        if source_code.endswith("\n") and not modified_code.endswith("\n"):
            modified_code += "\n"

        # Step 2: Scan docstrings and string literals
        try:
            tree = ast.parse(modified_code)
            modified_tree, doc_decisions = self._sanitize_ast_docstrings(tree)
            decisions.extend(doc_decisions)
            if doc_decisions:
                # If docstrings were modified, re-unparse or reconstruct
                try:
                    modified_code = ast.unparse(modified_tree)
                    if source_code.endswith("\n") and not modified_code.endswith("\n"):
                        modified_code += "\n"
                except Exception:
                    pass
        except SyntaxError:
            pass

        return modified_code, decisions

    # -----------------------------------------------------------------------
    # AST Inspection & Sanitization
    # -----------------------------------------------------------------------

    def _sanitize_ast_docstrings(
        self,
        tree: ast.AST,
    ) -> Tuple[ast.AST, List[SecurityDecision]]:
        """Walk AST and sanitize malicious payloads inside docstrings and string constants."""
        decisions: List[SecurityDecision] = []

        class DocstringSanitizer(ast.NodeTransformer):
            def __init__(outer_self):
                super().__init__()

            def _check_and_sanitize_str(outer_self, val: str, target_name: str, lineno: int) -> Tuple[str, Optional[SecurityDecision]]:
                score, matched_ids, matched = self.compute_risk_score(
                    val, vector_type="DOCSTRING_PAYLOAD"
                )
                if score >= self.sanitize_threshold:
                    vector = matched[0].vector if matched else "INDIRECT_PROMPT_INJECTION"
                    details = f"Sanitized docstring in '{target_name}' at line {lineno} (risk: {score:.2f})."
                    self.record_audit_log(
                        vector=vector,
                        risk_score=score,
                        action="SANITIZE",
                        mitigation_status="Sanitized",
                        matched_patterns=matched_ids,
                        details=details,
                        target_snippet=val,
                    )
                    dec = SecurityDecision(
                        action="SANITIZE",
                        risk_score=score,
                        vector=vector,
                        mitigation_status="Sanitized",
                        matched_patterns=matched_ids,
                        details=details,
                        sanitized_content=self.SANITIZED_PLACEHOLDER,
                    )
                    return self.SANITIZED_PLACEHOLDER, dec
                return val, None

            def visit_FunctionDef(outer_self, node: ast.FunctionDef):
                outer_self.generic_visit(node)
                doc = ast.get_docstring(node, clean=False)
                if doc and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    new_val, dec = outer_self._check_and_sanitize_str(doc, node.name, node.lineno)
                    if dec:
                        node.body[0].value.value = new_val
                        decisions.append(dec)
                return node

            def visit_AsyncFunctionDef(outer_self, node: ast.AsyncFunctionDef):
                outer_self.generic_visit(node)
                doc = ast.get_docstring(node, clean=False)
                if doc and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    new_val, dec = outer_self._check_and_sanitize_str(doc, node.name, node.lineno)
                    if dec:
                        node.body[0].value.value = new_val
                        decisions.append(dec)
                return node

            def visit_ClassDef(outer_self, node: ast.ClassDef):
                outer_self.generic_visit(node)
                doc = ast.get_docstring(node, clean=False)
                if doc and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    new_val, dec = outer_self._check_and_sanitize_str(doc, node.name, node.lineno)
                    if dec:
                        node.body[0].value.value = new_val
                        decisions.append(dec)
                return node

            def visit_Module(outer_self, node: ast.Module):
                outer_self.generic_visit(node)
                doc = ast.get_docstring(node, clean=False)
                if doc and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    new_val, dec = outer_self._check_and_sanitize_str(doc, "<module>", 1)
                    if dec:
                        node.body[0].value.value = new_val
                        decisions.append(dec)
                return node

        sanitizer = DocstringSanitizer()
        tree = sanitizer.visit(tree)
        ast.fix_missing_locations(tree)
        return tree, decisions

    def inspect_and_sanitize_ast(
        self,
        tree: ast.AST,
        source_code: str = "",
    ) -> Tuple[ast.AST, str, List[SecurityDecision]]:
        """
        Public API: Walk AST and inspect docstrings / comments.
        Sanitizes indirect injection payloads and returns sanitized AST and code.
        """
        modified_tree, decisions = self._sanitize_ast_docstrings(tree)
        sanitized_source = source_code
        if source_code:
            sanitized_source, comment_decisions = self.scan_and_sanitize_source(source_code)
            decisions.extend(comment_decisions)
        return modified_tree, sanitized_source, decisions
