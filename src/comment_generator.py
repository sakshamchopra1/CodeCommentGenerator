"""
Comment Generator — AST-Driven NLP Engine
==========================================
Two complementary strategies:

  1. Rule-Based Engine  — deterministic, always available.
                          Driven by AST analysis: loops, conditionals, calls,
                          variables, complexity, decorators, return types, raises.

  2. ML Engine (T5-AST) — activated via ml_generate_comments().
                          Feeds structured AST feature objects directly into a
                          fine-tuned T5 model.  The model ONLY sees AST-derived
                          information — never raw source code or the function name.

Output: List[CommentItem] — each with location, text, and kind.
"""
import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .ast_extractor import ModuleFeatures, FunctionFeature, ClassFeature, ParamFeature
from .context_analyzer import ContextGraph, FunctionContext
from .ast_body_extractor import (
    extract_body_snippet,
    extract_raises,
    extract_returned_types,
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class CommentItem:
    node_id: str
    node_type: str       # "function" | "class" | "inline"
    lineno: int          # line BEFORE which the comment should be inserted
    col_offset: int      # indentation of the target node
    text: str            # the full comment text (may be multi-line docstring)
    kind: str            # "docstring" | "inline" | "block"
    target_name: str = ""


# ---------------------------------------------------------------------------
# NLP token utilities
# ---------------------------------------------------------------------------

# Common English stop words to skip when forming descriptions
_STOP_WORDS = {
    "a", "an", "the", "to", "of", "in", "for", "is", "it",
    "on", "at", "by", "as", "or", "be", "do", "if", "so",
    "up", "no", "my", "we", "us"
}

# Verb map: first meaningful token → verb phrase
_VERB_MAP = {
    "get":       "Retrieves",
    "fetch":     "Fetches",
    "load":      "Loads",
    "read":      "Reads",
    "set":       "Sets",
    "update":    "Updates",
    "write":     "Writes",
    "save":      "Saves",
    "store":     "Stores",
    "put":       "Puts",
    "add":       "Adds",
    "append":    "Appends",
    "insert":    "Inserts",
    "push":      "Pushes",
    "remove":    "Removes",
    "delete":    "Deletes",
    "clear":     "Clears",
    "pop":       "Pops",
    "calc":      "Calculates",
    "calculate": "Calculates",
    "compute":   "Computes",
    "count":     "Counts",
    "sum":       "Sums",
    "find":      "Finds",
    "search":    "Searches",
    "check":     "Checks",
    "validate":  "Validates",
    "verify":    "Verifies",
    "is":        "Checks whether",
    "has":       "Checks if",
    "can":       "Determines whether",
    "parse":     "Parses",
    "format":    "Formats",
    "convert":   "Converts",
    "transform": "Transforms",
    "encode":    "Encodes",
    "decode":    "Decodes",
    "process":   "Processes",
    "handle":    "Handles",
    "run":       "Runs",
    "execute":   "Executes",
    "start":     "Starts",
    "stop":      "Stops",
    "init":      "Initializes",
    "initialize":"Initializes",
    "setup":     "Sets up",
    "build":     "Builds",
    "create":    "Creates",
    "make":      "Creates",
    "generate":  "Generates",
    "render":    "Renders",
    "display":   "Displays",
    "show":      "Shows",
    "print":     "Prints",
    "log":       "Logs",
    "send":      "Sends",
    "receive":   "Receives",
    "connect":   "Connects",
    "open":      "Opens",
    "close":     "Closes",
    "sort":      "Sorts",
    "filter":    "Filters",
    "map":       "Maps",
    "merge":     "Merges",
    "split":     "Splits",
    "join":      "Joins",
    "extract":   "Extracts",
    "load_data": "Loads data from",
    "test":      "Tests",
    "assert":    "Asserts",
    "analyze":   "Analyzes",
    "analyse":   "Analyses",
    "detect":    "Detects",
    "track":     "Tracks",
    "collect":   "Collects",
    "gather":    "Gathers",
    "compare":   "Compares",
    "evaluate":  "Evaluates",
    "apply":     "Applies",
    "dispatch":  "Dispatches",
    "emit":      "Emits",
    "publish":   "Publishes",
    "subscribe": "Subscribes",
    "resolve":   "Resolves",
    "notify":    "Notifies",
    "configure": "Configures",
    "register":  "Registers",
    "unregister":"Unregisters",
    "enable":    "Enables",
    "disable":   "Disables",
    "reset":     "Resets",
    "refresh":   "Refreshes",
    "clone":     "Clones",
    "copy":      "Copies",
    "move":      "Moves",
    "rename":    "Renames",
    "serialize": "Serializes",
    "deserialize":"Deserializes",
    "dump":      "Dumps",
    "restore":   "Restores",
    "backup":    "Backs up",
    "import":    "Imports",
    "export":    "Exports",
    "upload":    "Uploads",
    "download":  "Downloads",
    "compress":  "Compresses",
    "decompress":"Decompresses",
    "hash":      "Hashes",
    "sign":      "Signs",
    "authenticate": "Authenticates",
    "authorize": "Authorizes",
    "tokenize":  "Tokenizes",
    "stem":      "Stems",
    "lemmatize": "Lemmatizes",
    "predict":   "Predicts",
    "infer":     "Infers",
    "classify":  "Classifies",
    "cluster":   "Clusters",
    "embed":     "Embeds",
    "train":     "Trains",
    "fit":       "Fits",
    "score":     "Scores",
    "plot":      "Plots",
    "draw":      "Draws",
    "paint":     "Paints",
    "resize":    "Resizes",
    "crop":      "Crops",
    "rotate":    "Rotates",
    "flip":      "Flips",
}


def _sanitize_docstring_content(text: str) -> str:
    if not text:
        return text

    # Remove raw string prefixes that might appear at line starts
    text = re.sub(r'^([rubfRUBF]+)"""', r'"\1"', text, flags=re.MULTILINE)
    text = re.sub(r"^([rubfRUBF]+)'''", r"'\1'", text, flags=re.MULTILINE)

    # Replace triple quotes with single quotes to prevent docstring breaks
    text = text.replace('"""', "''")
    text = text.replace("'''", "''")

    # Prevent lines from starting with ''' or """
    text = re.sub(r'^\s*("""|\'\'\')', r'', text, flags=re.MULTILINE)

    return text


def _split_identifier(name: str) -> List[str]:
    """Split snake_case and CamelCase identifiers into lowercase tokens."""
    parts = name.split("_")
    tokens = []
    for part in parts:
        sub = re.sub(r'([A-Z][a-z]+)', r' \1', part)
        sub = re.sub(r'([A-Z]+)([A-Z][a-z])', r' \1 \2', sub)
        tokens.extend(sub.strip().lower().split())
    return [t for t in tokens if t]


def _meaningful_tokens(name: str) -> List[str]:
    """Return non-stop-word tokens from an identifier."""
    return [t for t in _split_identifier(name) if t not in _STOP_WORDS]


def _pick_verb(tokens: List[str]) -> str:
    """Pick the best verb phrase for a list of name tokens."""
    for t in tokens:
        if t in _VERB_MAP:
            return _VERB_MAP[t]
    return "Handles"


def _humanize(name: str) -> str:
    """Turn a snake_case or camelCase identifier into a readable phrase."""
    return " ".join(_split_identifier(name))


# ---------------------------------------------------------------------------
# AST-driven body analysis helpers
# ---------------------------------------------------------------------------

def _describe_body(
    ff: FunctionFeature,
    fc: Optional[FunctionContext],
    source_code: str = "",
) -> List[str]:
    """
    Compose natural-language sentences that describe what the function body
    *actually does*, derived entirely from extracted AST features.

    Returns a list of plain-text sentences (may be empty).
    """
    sentences: List[str] = []

    # ── Async marker ─────────────────────────────────────────────────────────
    if ff.is_async:
        sentences.append("Runs asynchronously (coroutine).")

    # ── Loop analysis ────────────────────────────────────────────────────────
    if ff.loops == 1:
        sentences.append("Iterates over a sequence using 1 loop.")
    elif ff.loops > 1:
        sentences.append(f"Iterates over sequences using {ff.loops} loops.")

    # ── Conditional / branching ──────────────────────────────────────────────
    if ff.conditionals == 1:
        sentences.append("Applies 1 conditional branch to control flow.")
    elif ff.conditionals > 1:
        sentences.append(
            f"Applies {ff.conditionals} conditional branches to control flow."
        )

    # ── Cyclomatic complexity ────────────────────────────────────────────────
    if fc:
        cc = fc.cyclomatic_complexity
        label = fc.complexity_label
        if label == "moderate":
            sentences.append(
                f"Has moderate control-flow complexity (cyclomatic complexity = {cc})."
            )
        elif label == "complex":
            sentences.append(
                f"Has high control-flow complexity (cyclomatic complexity = {cc}); "
                "consider splitting into smaller functions."
            )
        elif label == "very_complex":
            sentences.append(
                f"Has very high control-flow complexity (cyclomatic complexity = {cc}); "
                "refactoring is strongly recommended."
            )

    # ── Internal calls (within this module) ─────────────────────────────────
    if fc and fc.calls_internal:
        call_list = ", ".join(f"`{c}()`" for c in fc.calls_internal[:5])
        suffix = " and more" if len(fc.calls_internal) > 5 else ""
        sentences.append(f"Delegates to module-internal function(s): {call_list}{suffix}.")

    # ── External calls ───────────────────────────────────────────────────────
    if fc and fc.calls_external:
        # Show up to 4 most informative external calls (exclude builtins)
        _builtins = {"print", "len", "range", "str", "int", "float", "bool",
                     "list", "dict", "set", "tuple", "type", "isinstance",
                     "hasattr", "getattr", "setattr", "super", "enumerate",
                     "zip", "map", "filter", "sorted", "reversed", "any", "all",
                     "max", "min", "sum", "abs", "round", "repr", "vars",
                     "open", "iter", "next"}
        ext = [c for c in fc.calls_external if c.split(".")[0] not in _builtins][:4]
        if ext:
            call_list = ", ".join(f"`{c}`" for c in ext)
            sentences.append(f"Calls external function(s): {call_list}.")

    # ── Variable / data-structure usage ─────────────────────────────────────
    if fc and fc.variables:
        types_seen: Set[str] = set()
        for vi in fc.variables:
            if vi.inferred_type and vi.inferred_type not in (
                "None", "NoneType", "bool", "int", "float", "str"
            ):
                types_seen.add(vi.inferred_type)
        interesting = sorted(types_seen - {"object"})[:4]
        if interesting:
            sentences.append(
                "Works with data structure(s): " + ", ".join(interesting) + "."
            )

    # ── Raises ───────────────────────────────────────────────────────────────
    if source_code:
        raises = extract_raises(source_code, ff.lineno,
                                ff.lineno + ff.body_lines)
        if raises:
            exc_list = ", ".join(raises[:4])
            sentences.append(f"May raise: {exc_list}.")

    # ── Decorators ───────────────────────────────────────────────────────────
    for dec in ff.decorators:
        if "property" in dec:
            sentences.append("Defined as a property accessor.")
        elif "staticmethod" in dec:
            sentences.append("Defined as a static method (no instance binding).")
        elif "classmethod" in dec:
            sentences.append("Defined as a class method.")
        elif "abstractmethod" in dec:
            sentences.append("Declared as an abstract method; must be overridden by subclasses.")
        elif "cache" in dec.lower() or "lru_cache" in dec.lower():
            sentences.append("Results are memoized using a cache decorator.")
        elif "overload" in dec.lower():
            sentences.append("Uses @overload to support multiple call signatures.")

    # ── Body length note ─────────────────────────────────────────────────────
    if ff.body_lines > 50:
        sentences.append(
            f"Note: This function spans {ff.body_lines} lines — "
            "consider breaking it into smaller, focused units."
        )

    return sentences


def _describe_class_attributes(cf: ClassFeature) -> List[str]:
    """Return description lines for class-level variables."""
    if not cf.class_variables:
        return []
    public = [v for v in cf.class_variables if not v.startswith("_")]
    private = [v for v in cf.class_variables if v.startswith("_")]
    lines = []
    if public:
        lines.append("Attributes:")
        for v in public[:8]:
            lines.append(f"    {v}: {_humanize(v)}.")
    if private:
        lines.append("")
        lines.append(f"    Internal attribute(s): {', '.join(private[:5])}.")
    return lines


# ---------------------------------------------------------------------------
# Rule-based docstring builders
# ---------------------------------------------------------------------------

def _generate_function_docstring(
    ff: FunctionFeature,
    fc: Optional[FunctionContext] = None,
    source_code: str = "",
) -> str:
    """
    Generate a rich docstring for a function using full AST analysis.

    Incorporates:
    - Verb derived from function name tokens
    - Noun phrase from remaining name tokens
    - Whether it's a method/async
    - Body description (loops, conditionals, calls, variables, raises, decorators)
    - Parameters with types and defaults
    - Return type (from annotation + inferred return expressions)
    - Raises section  (from AST walk)
    - Complexity note for complex/very_complex functions
    """
    tokens = _meaningful_tokens(ff.name)
    verb = _pick_verb(tokens)
    noun_tokens = [t for t in tokens if t not in _VERB_MAP]
    noun_phrase = " ".join(noun_tokens).strip()
    if not noun_phrase:
        noun_phrase = _humanize(ff.name)

    # Context prefix (method vs function, async)
    context_prefix = ""
    if ff.is_async:
        context_prefix = "Asynchronously "
    elif ff.is_method and ff.parent_class:
        pass  # method context implied by the class docstring

    summary = f"{context_prefix}{verb} {noun_phrase}."

    lines = ['"""', summary]

    # ── Body analysis section ────────────────────────────────────────────────
    body_sentences = _describe_body(ff, fc, source_code)
    # Filter out the async sentence since it's already in the summary
    body_sentences = [s for s in body_sentences if not s.startswith("Runs asynchronously")]
    if body_sentences:
        lines.append("")
        for sent in body_sentences:
            # Sanitize content before wrapping to prevent docstring breaks
            sent = _sanitize_docstring_content(sent)
            # Wrap long sentences at 80 chars within the docstring
            wrapped = textwrap.fill(sent, width=76, subsequent_indent="    ")
            # Sanitize again after wrapping in case wrapping creates issues
            wrapped = _sanitize_docstring_content(wrapped)
            lines.append(wrapped)

    # ── Args section ────────────────────────────────────────────────────────
    real_params = [p for p in ff.params if p.name not in ("self", "cls")]
    if real_params:
        lines.append("")
        lines.append("Args:")
        for p in real_params:
            ann = f" ({p.annotation})" if p.annotation else ""
            default = f", defaults to ``{p.default}``" if p.default is not None else ""
            desc = _sanitize_docstring_content(_humanize(p.name))
            lines.append(f"    {p.name}{ann}: {desc}{default}.")

    # ── Returns section ──────────────────────────────────────────────────────
    ret_ann = ff.return_annotation
    if ret_ann and ret_ann not in ("None", "none"):
        lines.append("")
        lines.append("Returns:")
        ret_ann = _sanitize_docstring_content(ret_ann)
        lines.append(f"    {ret_ann}: The {noun_phrase} result.")
    elif not ret_ann and source_code:
        # Try to infer from body return expressions
        inferred = extract_returned_types(
            source_code, ff.lineno, ff.lineno + ff.body_lines
        )
        if inferred and "None" not in inferred:
            ret_str = " or ".join(inferred[:3])
            ret_str = _sanitize_docstring_content(ret_str)
            lines.append("")
            lines.append("Returns:")
            lines.append(f"    {ret_str}: The {noun_phrase} result.")

    # ── Raises section ───────────────────────────────────────────────────────
    if source_code:
        raises = extract_raises(source_code, ff.lineno, ff.lineno + ff.body_lines)
        if raises:
            lines.append("")
            lines.append("Raises:")
            for exc in raises[:4]:
                exc = _sanitize_docstring_content(exc)
                lines.append(f"    {exc}: If an error occurs during {noun_phrase}.")

    # ── Security section ─────────────────────────────────────────────────────
    if fc and getattr(fc, 'security_issues', None):
        lines.append("")
        lines.append("Security Warnings:")
        for issue in fc.security_issues:
            issue = _sanitize_docstring_content(issue)
            lines.append(f"    - {issue}")

    lines.append('"""')
    return "\n".join(lines)


def _generate_class_docstring(
    cf: ClassFeature,
    source_code: str = "",
) -> str:
    """Generate a rich docstring for a class using full AST analysis."""
    tokens = _meaningful_tokens(cf.name)
    noun_phrase = " ".join(tokens).strip() or cf.name
    noun_phrase = noun_phrase[0].upper() + noun_phrase[1:] if noun_phrase else cf.name
    noun_phrase = _sanitize_docstring_content(noun_phrase)

    lines = ['"""', f"Represents a {noun_phrase}."]

    if cf.bases and cf.bases != ["object"]:
        bases = [_sanitize_docstring_content(b) for b in cf.bases]
        lines.append(f"Inherits from: {', '.join(bases)}.")

    # ── Attributes section ───────────────────────────────────────────────────
    attr_lines = _describe_class_attributes(cf)
    if attr_lines:
        lines.append("")
        lines.extend([_sanitize_docstring_content(line) for line in attr_lines])

    # ── Methods section ──────────────────────────────────────────────────────
    if cf.methods:
        public_methods = [m for m in cf.methods if not m.startswith("_")]
        if public_methods:
            lines.append("")
            lines.append("Methods:")
            for m in public_methods[:8]:
                m_tokens = _meaningful_tokens(m)
                m_verb = _pick_verb(m_tokens)
                m_noun = " ".join(t for t in m_tokens if t not in _VERB_MAP)
                desc = f"{m_verb} {m_noun}." if m_noun else f"{m_verb}."
                desc = _sanitize_docstring_content(desc)
                lines.append(f"    {m}(): {desc}")

    lines.append('"""')
    return "\n".join(lines)


def _generate_inline_comment(
    ff: FunctionFeature,
    fc: FunctionContext,
    source_code: str = "",
) -> Optional[str]:
    """
    Generate a block comment summarising a complex function body,
    placed above the function for quick scanning.
    """
    if fc.complexity_label == "simple":
        return None

    tokens = _meaningful_tokens(ff.name)
    verb = _pick_verb(tokens)
    noun = " ".join(t for t in tokens if t not in _VERB_MAP).strip() or ff.name
    cc = fc.cyclomatic_complexity

    parts = [f"# {verb} {noun}."]

    # Add a brief one-liner body summary
    body_parts: List[str] = []
    if ff.loops:
        body_parts.append(f"{ff.loops} loop(s)")
    if ff.conditionals:
        body_parts.append(f"{ff.conditionals} conditional(s)")
    if fc.calls_internal:
        body_parts.append(f"calls {', '.join(fc.calls_internal[:2])}")
    if body_parts:
        parts.append("# Body: " + ", ".join(body_parts) + ".")

    parts.append(
        f"# Cyclomatic complexity: {cc} ({fc.complexity_label})."
    )

    if source_code:
        raises = extract_raises(source_code, ff.lineno, ff.lineno + ff.body_lines)
        if raises:
            parts.append(f"# May raise: {', '.join(raises[:3])}.")

    if getattr(fc, 'security_issues', None):
        for issue in fc.security_issues:
            parts.append(f"# SECURITY WARNING: {issue}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API — Rule-Based
# ---------------------------------------------------------------------------

def generate_comments(
    module_features: ModuleFeatures,
    context_graph: ContextGraph,
    source_code: str = "",
    security_layer=None,
    user_prompt: Optional[str] = None,
) -> List[CommentItem]:
    """
    Generate AST-driven docstrings and inline comments for all undocumented
    functions and classes in a module.

    Args:
        module_features: Output of ast_extractor.extract_features().
        context_graph:   Output of context_analyzer.analyze_context().
        source_code:     Raw source string — used for body/raises analysis.
        security_layer:  Optional AgenticSecurityLayer instance.
        user_prompt:     Optional user prompt/instruction.

    Returns:
        List[CommentItem] sorted by line number.
    """
    if user_prompt and security_layer is not None:
        security_layer.validate_prompt(user_prompt)

    comments: List[CommentItem] = []

    # Build quick lookup: function name -> FunctionContext
    fc_map: Dict[str, FunctionContext] = {
        fc.name: fc for fc in context_graph.function_contexts
    }

    # --- Classes ---
    for cf in module_features.classes:
        if not cf.has_docstring:
            doc_text = _generate_class_docstring(cf, source_code)
            comments.append(CommentItem(
                node_id=cf.node_id,
                node_type="class",
                lineno=cf.lineno,
                col_offset=cf.col_offset,
                text=doc_text,
                kind="docstring",
                target_name=cf.name,
            ))

    # --- Functions ---
    for ff in module_features.functions:
        fc = fc_map.get(ff.name)

        if not ff.has_docstring:
            doc_text = _generate_function_docstring(ff, fc, source_code)
            comments.append(CommentItem(
                node_id=ff.node_id,
                node_type="function",
                lineno=ff.lineno,
                col_offset=ff.col_offset,
                text=doc_text,
                kind="docstring",
                target_name=ff.name,
            ))

        # Inline block comment for moderate/complex/very_complex functions
        if fc and fc.complexity_label != "simple":
            inline = _generate_inline_comment(ff, fc, source_code)
            if inline:
                comments.append(CommentItem(
                    node_id=ff.node_id + "_inline",
                    node_type="inline",
                    lineno=ff.lineno,
                    col_offset=ff.col_offset,
                    text=inline,
                    kind="inline",
                    target_name=ff.name,
                ))

    if security_layer is not None:
        for c in comments:
            score, matched_ids, matched = security_layer.compute_risk_score(c.text, vector_type="SOURCE_CODE_COMMENT")
            if score >= security_layer.sanitize_threshold:
                if c.kind == "inline":
                    c.text = f"# {security_layer.SANITIZED_PLACEHOLDER}"
                else:
                    c.text = f'"""{security_layer.SANITIZED_PLACEHOLDER}"""'

    # Sort by lineno so attacher processes in order
    comments.sort(key=lambda c: (c.lineno, c.kind))
    return comments


# ---------------------------------------------------------------------------
# Public API — ML-Enhanced (AST Features → T5 NLP Model)
# ---------------------------------------------------------------------------

def ml_generate_comments(
    module_features: ModuleFeatures,
    context_graph: ContextGraph,
    ast_model=None,
    source_code: str = "",
    strict_ml: bool = False,
    security_layer=None,
    user_prompt: Optional[str] = None,
) -> List[CommentItem]:
    """
    Generate comments using ASTCommentModel — T5 fine-tuned on structured
    AST features.

    For each undocumented function:
      1. Get FunctionFeature (loops, conditionals, params, decorators …)
         and FunctionContext (complexity, calls, variable types …).
      2. Extract raises from source_code via ast_body_extractor.
      3. Call  ASTCommentModel.generate(ff, fc, raises)
         — the model receives ONLY structured AST feature objects,
           never raw source code or the function name alone.
      4. In strict ML mode, fail fast if model is unavailable or generation fails.
         Otherwise, fall back to rule-based generation.

    Args:
        module_features: Output of ast_extractor.extract_features().
        context_graph:   Output of context_analyzer.analyze_context().
        ast_model:       A loaded ASTCommentModel instance (or None).
        source_code:     Raw source string — used only for raises extraction.
        strict_ml:       If True, require ML model and fail on ML generation errors.
        security_layer:  Optional AgenticSecurityLayer instance.
        user_prompt:     Optional user prompt/instruction.

    Returns:
        List[CommentItem] sorted by line number.
    """
    if user_prompt and security_layer is not None:
        security_layer.validate_prompt(user_prompt)

    if ast_model is None and strict_ml:
        raise RuntimeError(
            "AST+NLP ML mode requires a trained AST model. "
            "Run: python3 -m src.main --train"
        )

    if ast_model is None:
        return generate_comments(
            module_features,
            context_graph,
            source_code,
            security_layer=security_layer,
            user_prompt=user_prompt,
        )

    fc_map: Dict[str, FunctionContext] = {
        fc.name: fc for fc in context_graph.function_contexts
    }
    comments: List[CommentItem] = []

    # --- Note: Rule-based generation is strictly disabled in ML mode ---
    # Classes are skipped since the ML model only supports Functions
    for ff in module_features.functions:
        fc = fc_map.get(ff.name)

        if not ff.has_docstring:
            raises: List[str] = []
            if source_code:
                try:
                    end_line = ff.lineno + ff.body_lines
                    raises = extract_raises(source_code, ff.lineno, end_line)
                except Exception:
                    raises = []

            doc_text = None
            try:
                raw_text, confidence = ast_model.generate(ff, fc, raises)
                summary = _sanitize_docstring_content(raw_text.strip().strip('"""').strip())

                doc_text = build_full_docstring(
                    summary, ff, fc, source_code, raises
                )
            except Exception as exc:
                if strict_ml:
                    raise RuntimeError(
                        f"AST+NLP generation failed for function '{ff.name}': {exc}"
                    ) from exc
                else:
                    print(f"Warning: ML generation failed for '{ff.name}': {exc}")
                    continue

            if doc_text:
                comments.append(CommentItem(
                    node_id=ff.node_id,
                    node_type="function",
                    lineno=ff.lineno,
                    col_offset=ff.col_offset,
                    text=doc_text,
                    kind="docstring",
                    target_name=ff.name,
                ))

            # Inline block comment for security issues
            if fc and getattr(fc, 'security_issues', None):
                inline_parts = ["# SECURITY WARNING: " + issue for issue in fc.security_issues]
                comments.append(CommentItem(
                    node_id=ff.node_id + "_inline_security",
                    node_type="inline",
                    lineno=ff.lineno,
                    col_offset=ff.col_offset,
                    text="\n".join(inline_parts),
                    kind="inline",
                    target_name=ff.name,
                ))

    if security_layer is not None:
        for c in comments:
            score, matched_ids, matched = security_layer.compute_risk_score(c.text, vector_type="SOURCE_CODE_COMMENT")
            if score >= security_layer.sanitize_threshold:
                if c.kind == "inline":
                    c.text = f"# {security_layer.SANITIZED_PLACEHOLDER}"
                else:
                    c.text = f'"""{security_layer.SANITIZED_PLACEHOLDER}"""'

    comments.sort(key=lambda c: (c.lineno, c.kind))
    return comments


def build_full_docstring(
    summary: str,
    ff,
    fc,
    source_code: str = "",
    raises: Optional[List[str]] = None,
) -> str:
    """
    Assemble a complete Google-style docstring by combining:
      - An ML-generated summary line.
      - Args: / Returns: / Raises: sections derived from AST features.

    This ensures the structured, accurate sections always come from AST
    while the natural-language summary comes from the NLP model.
    """
    noun_phrase = _humanize(ff.name)
    real_params = [p for p in ff.params if p.name not in ("self", "cls")]

    tokens = _meaningful_tokens(ff.name)
    fallback_summary = f"{_pick_verb(tokens)} {noun_phrase}."
    summary = _sanitize_docstring_content(summary if summary else fallback_summary)
    lines = ['"""', summary]

    # Args
    if real_params:
        lines.append("")
        lines.append("Args:")
        for p in real_params:
            ann     = f" ({p.annotation})" if p.annotation else ""
            default = f", defaults to ``{p.default}``" if p.default is not None else ""
            desc = _sanitize_docstring_content(_humanize(p.name))
            lines.append(f"    {p.name}{ann}: {desc}{default}.")

    # Returns
    ret_ann = ff.return_annotation
    if ret_ann and ret_ann not in ("None", "none"):
        lines.append("")
        lines.append("Returns:")
        ret_ann = _sanitize_docstring_content(ret_ann)
        lines.append(f"    {ret_ann}: The {noun_phrase} result.")

    # Raises
    if raises:
        lines.append("")
        lines.append("Raises:")
        for exc in raises[:4]:
            exc = _sanitize_docstring_content(exc)
            lines.append(f"    {exc}: If an error occurs during {noun_phrase}.")

    lines.append('"""')
    return "\n".join(lines)


