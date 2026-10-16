"""
Main CLI — Week 9 ML/AI Integration Pipeline
==============================================
Full pipeline:
  parse → validate → extract → analyze → [ml_]generate → attach
  → build_ir → run_analysis

Usage:
    python -m src.main <file.py> [options]

Options:
    --output <out.py>    Write annotated source to this file
    --logs               Save JSON + text pipeline logs to logs/
    --show-features      Print extracted AST features as JSON
    --show-context       Print the context graph as JSON
    --ir                 Print the pretty-printed IR dump
    --analysis           Print the pattern-analysis report
    --ml                 Use ML-based comment generation (Week 9)
    --train              Train/retrain the ML models and exit
    --output-dir DIR     Output directory for models/reports (default: outputs)
"""
import sys
import os
import json
import argparse
import warnings
from typing import Optional, List, Dict, Any, Tuple

# Suppress SyntaxWarnings from ast.parse when evaluating legacy CodeSearchNet/Stdlib files
warnings.filterwarnings("ignore", category=SyntaxWarning)

from .parser_module import read_file, parse_code
from .validator import validate_ast
from .error_handler import format_error, ParserError
from .ast_extractor import extract_features, features_to_dict
from .context_analyzer import analyze_context, context_to_dict
from .comment_generator import generate_comments, ml_generate_comments
from .neurosymbolic.engine import neurosymbolic_generate_comments
from .comment_attacher import attach_comments
from .logger import PipelineLogger
from .ir import build_ir, pretty_print_ir, serialize_ir
from .analysis import build_cfg, run_dfa, detect_patterns
from .security_analyzer import run_security_analysis, SecurityReport
from .security import AgenticSecurityLayer, SecurityException


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="code-comment-gen",
        description="Code Comment Generation via AST with two engines: Rule-Based and AST+NLP ML",
    )
    p.add_argument("filepath", nargs="?", default=None,
                   help="Path to the Python source file to annotate")
    p.add_argument("--block-max-pairs", type=int, default=20000,
                        help="Maximum block-level (code -> inline comment) pairs to extract.")
    p.add_argument("--output", "-o", metavar="FILE",
                   help="Write annotated source to FILE (default: print to stdout)")
    p.add_argument("--logs", action="store_true",
                   help="Save pipeline logs (JSON + text) to logs/ directory")
    p.add_argument("--show-features", action="store_true",
                   help="Print extracted AST features as JSON")
    p.add_argument("--show-context", action="store_true",
                   help="Print semantic context graph as JSON")
    p.add_argument("--ir", action="store_true",
                   help="Print the pretty-printed IR dump (Week 8)")
    p.add_argument("--analysis", action="store_true",
                   help="Print the pattern-analysis report (Week 8)")
    p.add_argument("--ml", action="store_true",
                   help="Use AST+NLP ML-based comment generation")
    p.add_argument("--neurosymbolic", action="store_true",
                   help="Use neurosymbolic (confidence-gated ML + symbolic) generation")
    p.add_argument("--train", action="store_true",
                   help="Train/retrain ML models and save reports, then exit")
    p.add_argument("--epochs", type=int, default=4,
                   help="Training epochs for AST model (default: 4)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Training batch size for AST model (default: 8)")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Gradient accumulation steps (default: 1)")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate for AST model fine-tuning (default: 2e-4)")
    p.add_argument("--codesearchnet-max", type=int, default=20000,
                   help="Max CodeSearchNet samples for training (default: 20000)")
    p.add_argument("--max-stdlib-files", type=int, default=1500,
                   help="Max stdlib .py files for fallback/supplement (default: 1500)")
    p.add_argument("--no-codesearchnet", action="store_true",
                   help="Disable CodeSearchNet dataset source")
    p.add_argument("--no-stdlib", action="store_true",
                   help="Disable Python stdlib dataset source")
    p.add_argument("--output-dir", metavar="DIR", default="outputs",
                   help="Directory for ML model outputs (default: outputs)")
    p.add_argument("--prompt", type=str, default=None,
                   help="User prompt or instruction for comment generation")
    p.add_argument("--security-threshold", type=float, default=0.7,
                   help="Risk score threshold to block malicious prompts (default: 0.7)")
    p.add_argument("--sanitize-threshold", type=float, default=0.4,
                   help="Risk score threshold to sanitize comments/docstrings (default: 0.4)")
    p.add_argument("--show-security-logs", action="store_true",
                   help="Display Agentic AI Security Layer structured audit logs")
    return p


def run_pipeline(
    filepath: str,
    logger: PipelineLogger,
    ast_model=None,
    strict_ml: bool = False,
    engine: str = "rule_based",
    user_prompt: Optional[str] = None,
    security_layer: Optional[AgenticSecurityLayer] = None,
):
    """
    Execute the full pipeline with Agentic AI Security Layer protection.

    Args:
        filepath:       Path to Python source file.
        logger:         PipelineLogger instance.
        ast_model:      Optional ASTCommentModel for ML-based generation.
        strict_ml:      Require AST+NLP ML path when True (no rule fallback).
        engine:         One of "rule_based", "ml", "neurosymbolic".
        user_prompt:    Optional user prompt/instruction to guide comment generation.
        security_layer: Optional AgenticSecurityLayer instance.

    Returns:
        (annotated_source, comments, module_features, context_graph,
         attach_result, ir_module, analysis_report, security_report)
    """
    if security_layer is None:
        security_layer = AgenticSecurityLayer()

    # ── Stage 0: Security Ingestion & Policy Inspection ──────────────
    logger.begin_stage("security_ingestion")

    # 1. Intercept and block malicious user prompts before NLP processing
    if user_prompt:
        prompt_decision = security_layer.inspect_prompt(user_prompt)
        if prompt_decision.action == "BLOCK":
            logger.end_stage(summary={
                "status": "BLOCKED",
                "risk_score": prompt_decision.risk_score,
                "vector": prompt_decision.vector,
                "matched_patterns": prompt_decision.matched_patterns,
            })
            raise SecurityException(
                f"Execution blocked: Direct prompt injection / policy bypass detected in user prompt "
                f"(risk score: {prompt_decision.risk_score:.2f})",
                risk_score=prompt_decision.risk_score,
                vector=prompt_decision.vector,
                audit_entry=prompt_decision.to_dict(),
            )

    source_code = read_file(filepath)

    # 2. Wire AST parser inspection so code-embedded comments/docstrings are scanned
    # for indirect injection payloads, sanitizing them if they exceed the risk threshold
    sanitized_source, decisions = security_layer.scan_and_sanitize_source(source_code)
    sanitized_count = sum(1 for d in decisions if d.action == "SANITIZE")

    # 3. Check if the remaining source code itself is a raw direct prompt injection attack
    code_decision = security_layer.inspect_prompt(sanitized_source)
    if code_decision.action == "BLOCK":
        logger.end_stage(summary={
            "status": "BLOCKED",
            "risk_score": code_decision.risk_score,
            "vector": code_decision.vector,
            "matched_patterns": code_decision.matched_patterns,
        })
        raise SecurityException(
            f"Execution blocked: Malicious prompt injection payload detected in source input "
            f"(risk score: {code_decision.risk_score:.2f})",
            risk_score=code_decision.risk_score,
            vector=code_decision.vector,
            audit_entry=code_decision.to_dict(),
        )

    logger.end_stage(summary={
        "status": "ALLOW",
        "user_prompt_checked": bool(user_prompt),
        "sanitized_payloads": sanitized_count,
    })

    # ── Stage 1: Parse ───────────────────────────────────────────────
    logger.begin_stage("parse")
    ast_tree = parse_code(sanitized_source, security_layer=security_layer)
    logger.end_stage(summary={
        "total_lines": len(sanitized_source.splitlines()),
        "file": filepath,
        "sanitized_payloads": sanitized_count,
    })

    # ── Stage 2: Validate ────────────────────────────────────────────
    logger.begin_stage("validate")
    val_errors = validate_ast(ast_tree)
    val_warnings = [format_error(e) for e in val_errors]
    logger.end_stage(
        summary={"issues_found": len(val_errors)},
        warnings=val_warnings,
    )

    # ── Stage 3: Extract AST Features ────────────────────────────────
    logger.begin_stage("extract")
    module_features = extract_features(ast_tree, sanitized_source, filepath=filepath)
    logger.end_stage(summary={
        "functions": len(module_features.functions),
        "classes": len(module_features.classes),
        "imports": len(module_features.imports),
    })

    # ── Stage 4: Analyze Context ─────────────────────────────────────
    logger.begin_stage("analyze")
    context_graph = analyze_context(module_features, ast_tree, sanitized_source)
    complexity_dist = {}
    for fc in context_graph.function_contexts:
        lbl = fc.complexity_label
        complexity_dist[lbl] = complexity_dist.get(lbl, 0) + 1
    logger.end_stage(summary={
        "function_contexts": len(context_graph.function_contexts),
        "complexity_distribution": complexity_dist,
    })

    # ── Stage 5: Generate Comments ───────────────────────────────────
    logger.begin_stage("generate")
    if engine == "neurosymbolic" and ast_model is not None:
        comments = neurosymbolic_generate_comments(
            module_features, context_graph, ast_model,
            source_code=sanitized_source,
            strict_ml=strict_ml,
        )
        gen_engine = "neurosymbolic"
    elif ast_model is not None:
        comments = ml_generate_comments(
            module_features, context_graph, ast_model,
            source_code=sanitized_source,
            strict_ml=strict_ml,
        )
        gen_engine = "ast+nlp_ml"
    else:
        comments = generate_comments(module_features, context_graph,
                                     source_code=sanitized_source)
        gen_engine = "rule_based"
    counts = {"docstring": 0, "inline": 0}
    for c in comments:
        counts[c.kind] = counts.get(c.kind, 0) + 1
    logger.end_stage(summary={
        "total_comments": len(comments),
        "by_kind": counts,
        "engine": gen_engine,
    })
    logger.set_comments_generated(len(comments))

    # ── Stage 6: Attach Comments ─────────────────────────────────────
    logger.begin_stage("attach")
    attach_result = attach_comments(sanitized_source, comments)
    logger.end_stage(summary={"comments_attached": attach_result.comments_attached})

    # ── Stage 7: Build IR ─────────────────────────────────────────────
    logger.begin_stage("build_ir")
    ir_module = build_ir(module_features, context_graph)
    logger.end_stage(summary={
        "ir_functions": len(ir_module.functions),
        "total_blocks": sum(len(f.blocks) for f in ir_module.functions),
    })

    # ── Stage 8: Analysis (CFG + DFA + Patterns) ──────────────────────
    logger.begin_stage("analysis")
    dfa_results = []
    for ir_func in ir_module.functions:
        cfg = build_cfg(ir_func)
        dfa = run_dfa(cfg, ir_func)
        dfa_results.append(dfa)
    analysis_report = detect_patterns(ir_module, dfa_results)
    logger.end_stage(summary={
        "findings": len(analysis_report.findings),
        "summary": analysis_report.summary,
    })

    # ── Stage 9: Security Analysis ────────────────────────────────────────
    logger.begin_stage("security")
    security_report = run_security_analysis(module_features, context_graph, sanitized_source)
    logger.end_stage(summary={
        "total_issues": security_report.total_issues,
        "module_safe_pct": security_report.module_safe_pct,
        "by_severity": security_report.by_severity,
    })

    return (attach_result.annotated_source, comments, module_features,
            context_graph, attach_result, ir_module, analysis_report,
            security_report)


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve output_dir relative to this script's location if relative
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else \
                 os.path.join(project_root, args.output_dir)

    # ── --train : train models and exit ──────────────────────────────────
    if args.train:
        print(f"\n{'='*55}")
        print("  Training AST+NLP ML Model")
        print(f"{'='*55}")
        try:
            from .ml.trainer import train_and_evaluate
            result = train_and_evaluate(
                output_dir=output_dir,
                include_codesearchnet=not args.no_codesearchnet,
                include_stdlib=not args.no_stdlib,
                codesearchnet_max=args.codesearchnet_max,
                max_stdlib_files=args.max_stdlib_files,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                lr=args.lr,
                verbose=True,
            )
            tr = result["training_report"]
            ev = result["eval_report"]
            print(f"\n  Dataset : {tr['dataset_total']} samples  "
                  f"(train={tr['train_size']}, test={tr['test_size']})")
            data_profile = tr.get("data_profile", {})
            print(f"  Training mode      : "
                  f"{data_profile.get('input_mode', 'AST structured features')} -> "
                  f"{data_profile.get('target_mode', 'NLP docstring sentence')}")
            summary = ev.get("summary", {})
            print(f"  Best BLEU-4        : {summary.get('best_bleu4', 0):.4f}")
            print(f"  Best ROUGE-L       : {summary.get('best_rouge_l', 0):.4f}")
            print(f"  Best Exact Match   : {summary.get('best_exact_match', 0):.4f}")
            print(f"\n  Reports saved to   : {output_dir}/")
            print(f"{'='*55}\n")
        except Exception as exc:
            print(f"\n  Training failed: {exc}")
            raise
        return

    # ── Require filepath for pipeline runs ────────────────────────────────
    if args.filepath is None:
        parser.error("filepath is required unless --train is specified")

    filepath = args.filepath
    logger = PipelineLogger(input_file=filepath)

    print(f"\n{'='*55}")
    print("  Code Comment Generation — Rule-Based + AST+NLP ML + Neurosymbolic")
    print(f"{'='*55}")
    print(f"  Input : {filepath}")

    engine = "rule_based"
    if args.neurosymbolic:
        engine = "neurosymbolic"
        print("  Mode  : Neurosymbolic (confidence-gated ML + symbolic fusion)")
    elif args.ml:
        engine = "ml"
        print("  Mode  : AST+NLP ML generation")
    else:
        print("  Mode  : Rule-Based generation")

    ast_model = None
    if args.ml or args.neurosymbolic:
        try:
            from .ml.trainer import load_ast_model
            ast_model = load_ast_model(output_dir=output_dir)
            if ast_model:
                print(f"  ML models loaded from: {output_dir}/model/")
            else:
                print("  [error] No trained AST+NLP model found.")
                print("          Train first: python3 -m src.main --train")
                sys.exit(2)
        except Exception as exc:
            print(f"  [error] Could not load AST+NLP model: {exc}")
            print("          Train first: python3 -m src.main --train")
            sys.exit(2)

    security_layer = AgenticSecurityLayer(
        block_threshold=args.security_threshold,
        sanitize_threshold=args.sanitize_threshold,
    )

    try:
        annotated, comments, mf, cg, attach_result, ir_module, analysis_report, security_report = \
            run_pipeline(
                filepath,
                logger,
                ast_model=ast_model,
                strict_ml=args.ml or args.neurosymbolic,
                engine=engine,
                user_prompt=args.prompt,
                security_layer=security_layer,
            )
    except SecurityException as e:
        print(f"\n{'='*55}")
        print("  [SECURITY BLOCKED] Agentic AI Security Layer Alert")
        print(f"{'='*55}")
        print(f"  {e}")
        print(f"  Risk Score    : {e.risk_score:.2f}")
        print(f"  Threat Vector : {e.vector}")
        print("  Mitigation    : BLOCKED before NLP comment processing.")
        print(f"{'='*55}\n")
        security_layer.print_audit_logs()
        sys.exit(3)
    except ParserError as e:
        print(f"\n{format_error(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise

    # ── Show Features ─────────────────────────────────────────────
    if args.show_features:
        print("\n--- Extracted AST Features ---")
        print(json.dumps(features_to_dict(mf), indent=2))

    # ── Show Context ──────────────────────────────────────────────
    if args.show_context:
        print("\n--- Semantic Context Graph ---")
        print(json.dumps(context_to_dict(cg), indent=2))

    # ── Show IR dump ──────────────────────────────────────────────
    if args.ir:
        print("\n--- IR Dump ---")
        print(pretty_print_ir(ir_module))

    # ── Show Analysis report ──────────────────────────────────────
    if args.analysis:
        print("\n--- Analysis Report ---")
        if analysis_report.findings:
            for finding in analysis_report.findings:
                sev = finding.severity.upper()
                print(f"  [{sev}] {finding.pattern_id} @{finding.function_name}: {finding.message}")
        else:
            print("  No patterns detected.")
        print(f"  Summary: {analysis_report.summary}")

    # ── Show Summary ──────────────────────────────────────────────
    print(f"\n  Functions found : {len(mf.functions)}")
    print(f"  Classes found   : {len(mf.classes)}")
    print(f"  Comments generated: {len(comments)}")
    print(f"  IR functions built: {len(ir_module.functions)}")
    print(f"  Analysis findings : {len(analysis_report.findings)}")
    print(f"  Security issues   : {security_report.total_issues} "
          f"({security_report.module_safe_pct}% safe)")
    print()

    for c in comments:
        kind_tag = "[docstring]" if c.kind == "docstring" else "[inline]  "
        print(f"  {kind_tag} {c.node_type} '{c.target_name}' (line {c.lineno})")

    # ── Output annotated file ─────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(annotated)
        logger.set_output_file(args.output)
        print(f"\n  Annotated file written: {args.output}")
    else:
        print("\n--- Annotated Source ---")
        print(annotated)

    # ── Save Logs ─────────────────────────────────────────────────
    if args.logs:
        # Determine logs dir relative to the input file's directory
        base_dir = os.path.dirname(os.path.abspath(filepath))
        logs_dir = os.path.join(base_dir, "..", "logs")
        logs_dir = os.path.normpath(logs_dir)
        json_path, text_path = logger.save(logs_dir)
        print(f"\n  Logs saved:")
        print(f"    JSON : {json_path}")
        print(f"    Text : {text_path}")
        # Save analysis report JSON
        report_path = os.path.join(
            os.path.dirname(os.path.abspath(filepath)), "..", "outputs", "analysis_report.json"
        )
        report_path = os.path.normpath(report_path)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        import json as _json
        with open(report_path, 'w', encoding='utf-8') as rf:
            _json.dump(
                {
                    "source_file": analysis_report.source_file,
                    "summary": analysis_report.summary,
                    "findings": [
                        {
                            "pattern_id": f.pattern_id,
                            "severity": f.severity,
                            "function": f.function_name,
                            "message": f.message,
                            "lineno": f.lineno,
                        }
                        for f in analysis_report.findings
                    ],
                },
                rf,
                indent=2,
            )
        print(f"    Analysis: {report_path}")

    # ── Security Audit Logs ───────────────────────────────────────
    if args.show_security_logs:
        security_layer.print_audit_logs()

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
