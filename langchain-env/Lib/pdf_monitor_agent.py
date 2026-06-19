"""
pdf_monitor_agent.py
LangGraph agent for PDF monitoring, extraction, evaluation & finalization.

Separate agent (NOT connected to crawler_graph).

Nodes (in order):
  1. scan_folder       — check for new PDFs in monitoring folder
  2. load_ground_truth — load GT file using naming convention
  3. extract_dual_llm  — extract with Gemini + GPT (via run_pipeline)
  4. evaluate_both     — evaluate both extractions (via _ev_score)
  5. pick_winner       — compare F1 scores, select winner
  6. save_results      — save winner's TP rows to results_finalized.csv
  7. check_next        — loop back to scan_folder or END

State flows through all nodes. Each node returns only what it changes.
"""

import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, TypedDict, Annotated, Optional
import operator
import pandas as pd

from langgraph.graph import StateGraph, END

from DocToDB_eval_v2 import run_pipeline, _ev_score, _find_col, _norm_name

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s"
)
log = logging.getLogger(__name__)

# Folder structure
PDF_MONITOR_ROOT = Path("./pdf_monitor")
PDF_FOLDER = PDF_MONITOR_ROOT / "pdfs"
GT_FOLDER = PDF_MONITOR_ROOT / "ground_truth"
DB_FOLDER = PDF_MONITOR_ROOT / "results_db"

RESULTS_FINALIZED_CSV = DB_FOLDER / "results_finalized.csv"
PROCESSING_LOG_JSON = DB_FOLDER / "processing_log.json"

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_folders():
    """Create required folder structure."""
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)
    GT_FOLDER.mkdir(parents=True, exist_ok=True)
    DB_FOLDER.mkdir(parents=True, exist_ok=True)
    log.info(f"✓ Folders ready: {PDF_MONITOR_ROOT.resolve()}")


def load_processing_state() -> set:
    """Load which files have been processed."""
    if PROCESSING_LOG_JSON.exists():
        try:
            with open(PROCESSING_LOG_JSON, "r") as f:
                data = json.load(f)
                return set(data.get("processed_files", []))
        except Exception as e:
            log.warning(f"Failed to load processing state: {e}")
    return set()


def save_processing_state(processed_files: set):
    """Save which files have been processed."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "processed_files": list(processed_files)
    }
    with open(PROCESSING_LOG_JSON, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# STATE DEFINITION (like CrawlerState)
# ─────────────────────────────────────────────────────────────────────────────

class PDFMonitorState(TypedDict):
    """Shared state flowing through all nodes."""
    current_pdf:         Optional[Path]      # PDF being processed
    pdf_bytes:           bytes               # PDF file content
    gt_df:               object              # ground truth dataframe
    df_consensus:        object              # consensus extraction result
    df_gemini:           object              # Gemini extraction result
    df_gpt:              object              # GPT extraction result
    fallback_rows:       object              # consensus + source_verified fallback rows when GT missing
    eval_gemini:         Optional[Dict]      # Gemini evaluation result
    eval_gpt:            Optional[Dict]      # GPT evaluation result
    winner_eval:         Optional[Dict]      # winning LLM evaluation
    winner_name:         str                 # "Gemini", "GPT", or fallback label
    processed_files:     set                 # which PDFs already processed
    pending_pdfs:        List[Path]          # remaining PDFs to process
    errors:              Annotated[List[str], operator.add]  # error log


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: Scan folder for unprocessed PDFs
# ─────────────────────────────────────────────────────────────────────────────

def scan_folder(state: PDFMonitorState) -> dict:
    """
    Scan PDF folder for unprocessed PDFs.
    Returns next PDF to process, or empty to end.
    """
    log.info("\n[Node 1] Scanning PDF folder...")
    
    try:
        all_pdfs = sorted([f for f in PDF_FOLDER.glob("*.pdf") if f.is_file()])
        pending = [p for p in all_pdfs if p.name not in state["processed_files"]]
        
        if pending:
            next_pdf = pending[0]
            log.info(f"  Found {len(pending)} unprocessed PDF(s)")
            log.info(f"  Next: {next_pdf.name}")
            return {"pending_pdfs": pending, "current_pdf": next_pdf}
        else:
            log.info(f"  No unprocessed PDFs found (total: {len(all_pdfs)})")
            return {"pending_pdfs": [], "current_pdf": None}
    except Exception as e:
        log.error(f"  Error scanning folder: {e}")
        return {"errors": [f"Scan error: {e}"], "pending_pdfs": []}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: Load ground truth using naming convention
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(state: PDFMonitorState) -> dict:
    """
    Load ground truth for current PDF using naming convention.
    Pattern: {pdf_name}_gt.{ext} or fallback to ground_truth.{ext}
    """
    pdf_path = state["current_pdf"]
    log.info(f"\n[Node 2] Loading ground truth for {pdf_path.name}...")
    
    pdf_stem = pdf_path.stem
    gt_base = f"{pdf_stem}_gt"
    
    # Try {pdf_name}_gt.{ext}
    for ext in [".csv", ".xlsx", ".xls"]:
        gt_path = GT_FOLDER / f"{gt_base}{ext}"
        if gt_path.exists():
            try:
                if ext == ".csv":
                    gt_df = pd.read_csv(gt_path)
                else:
                    gt_df = pd.read_excel(gt_path)
                log.info(f"  ✓ Loaded: {gt_path.name} ({len(gt_df)} rows)")
                return {"gt_df": gt_df}
            except Exception as e:
                log.error(f"  Failed to load {gt_path.name}: {e}")
                return {"errors": [f"GT load error: {e}"], "gt_df": None}
    
    # Fallback to ground_truth.{ext}
    log.info(f"  ('{gt_base}' not found, trying fallback...)")
    for ext in [".csv", ".xlsx", ".xls"]:
        gt_path = GT_FOLDER / f"ground_truth{ext}"
        if gt_path.exists():
            try:
                if ext == ".csv":
                    gt_df = pd.read_csv(gt_path)
                else:
                    gt_df = pd.read_excel(gt_path)
                log.info(f"  ✓ Loaded fallback: {gt_path.name} ({len(gt_df)} rows)")
                return {"gt_df": gt_df}
            except Exception as e:
                log.error(f"  Failed to load fallback: {e}")
                return {"errors": [f"GT fallback error: {e}"], "gt_df": None}
    
    err = f"No GT found for {pdf_path.name} (tried {gt_base}*.* and ground_truth.*)"
    log.error(f"  ✗ {err}")
    return {"errors": [err], "gt_df": None}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: Extract with Gemini + GPT (direct call to run_pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def extract_dual_llm(state: PDFMonitorState) -> dict:
    """
    Extract properties from PDF using dual LLM pipeline.
    DIRECT CALL to run_pipeline() - no wrapper.
    Returns: (df_gemini, df_gpt, chunks, errors, meta)
    """
    pdf_path = state["current_pdf"]
    log.info(f"\n[Node 3] Extracting with Gemini + GPT from {pdf_path.name}...")
    
    try:
        pdf_bytes = pdf_path.read_bytes()
        
        # DIRECT CALL to run_pipeline - returns consensus, Gemini, GPT results
        df_consensus, df_gemini, df_gpt, chunks, errors, meta = run_pipeline(pdf_bytes)
        
        if errors:
            log.warning(f"  Extraction warnings: {errors}")
        
        if df_gemini is None or len(df_gemini) == 0:
            log.warning(f"  Gemini: 0 rows extracted")
            df_gemini = None
        else:
            log.info(f"  ✓ Gemini: {len(df_gemini)} rows")
        
        if df_gpt is None or len(df_gpt) == 0:
            log.warning(f"  GPT: 0 rows extracted")
            df_gpt = None
        else:
            log.info(f"  ✓ GPT: {len(df_gpt)} rows")
        
        if df_consensus is None or len(df_consensus) == 0:
            log.info(f"  Consensus: 0 rows")
        else:
            log.info(f"  ✓ Consensus: {len(df_consensus)} rows")
        
        if df_gemini is None and df_gpt is None:
            err = "Both extractions returned 0 rows"
            log.error(f"  ✗ {err}")
            return {"errors": [err], "df_gemini": None, "df_gpt": None}
        
        return {
            "pdf_bytes": pdf_bytes,
            "df_consensus": df_consensus,
            "df_gemini": df_gemini,
            "df_gpt": df_gpt,
            "errors": errors
        }
        
    except Exception as e:
        log.error(f"  Extraction failed: {e}")
        return {"errors": [f"Extraction error: {e}"], "df_gemini": None, "df_gpt": None}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4: Evaluate both extractions (direct call to _ev_score)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_both(state: PDFMonitorState) -> dict:
    """
    Evaluate Gemini + GPT extractions against ground truth.
    DIRECT CALL to _ev_score() - no wrapper.
    """
    gt_df = state["gt_df"]
    log.info(f"\n[Node 4] Evaluating extractions...")

    if gt_df is None:
        log.info("  Ground truth missing — skipping GT evaluation")
        fallback_rows = _combine_consensus_and_verified(state)
        log.info(f"  ✓ Fallback rows prepared: {len(fallback_rows)}")
        return {
            "eval_gemini": None,
            "eval_gpt": None,
            "fallback_rows": fallback_rows
        }
    
    eval_results = {}
    
    # Evaluate Gemini
    if state["df_gemini"] is not None and len(state["df_gemini"]) > 0:
        try:
            log.info(f"  Scoring Gemini...")
            nc = _find_col(state["df_gemini"], ["property_name", "property", "name"])
            vc = _find_col(state["df_gemini"], ["value", "val", "measured_value"])
            uc = _find_col(state["df_gemini"], ["unit", "units", "si_unit"])
            
            gnc = _find_col(gt_df, ["property_name", "property", "name"])
            gvc = _find_col(gt_df, ["value", "val", "measured_value"])
            guc = _find_col(gt_df, ["unit", "units", "si_unit"])
            
            if not (nc and vc and gnc and gvc):
                log.warning(f"    Could not detect all columns")
                eval_results["gemini"] = None
            else:
                # DIRECT CALL to _ev_score
                annotated_df, metrics = _ev_score(
                    model_df=state["df_gemini"],
                    gt_df=gt_df,
                    nc=nc, vc=vc, uc=uc,
                    gnc=gnc, gvc=gvc, guc=guc,
                    min_conf=1
                )
                
                eval_results["gemini"] = {
                    "model": "Gemini",
                    "f1": metrics.get("F1", 0.0),
                    "precision": metrics.get("Precision", 0.0),
                    "recall": metrics.get("Recall", 0.0),
                    "tp": metrics.get("TP", 0),
                    "fp": metrics.get("FP", 0),
                    "fn": metrics.get("FN", 0),
                    "metrics": metrics,
                    "annotated_df": annotated_df
                }
                log.info(f"    ✓ Gemini F1: {eval_results['gemini']['f1']:.4f}")
        except Exception as e:
            log.error(f"    Gemini evaluation failed: {e}")
            eval_results["gemini"] = None
    
    # Evaluate GPT
    if state["df_gpt"] is not None and len(state["df_gpt"]) > 0:
        try:
            log.info(f"  Scoring GPT...")
            nc = _find_col(state["df_gpt"], ["property_name", "property", "name"])
            vc = _find_col(state["df_gpt"], ["value", "val", "measured_value"])
            uc = _find_col(state["df_gpt"], ["unit", "units", "si_unit"])
            
            gnc = _find_col(gt_df, ["property_name", "property", "name"])
            gvc = _find_col(gt_df, ["value", "val", "measured_value"])
            guc = _find_col(gt_df, ["unit", "units", "si_unit"])
            
            if not (nc and vc and gnc and gvc):
                log.warning(f"    Could not detect all columns")
                eval_results["gpt"] = None
            else:
                # DIRECT CALL to _ev_score
                annotated_df, metrics = _ev_score(
                    model_df=state["df_gpt"],
                    gt_df=gt_df,
                    nc=nc, vc=vc, uc=uc,
                    gnc=gnc, gvc=gvc, guc=guc,
                    min_conf=1
                )
                
                eval_results["gpt"] = {
                    "model": "GPT",
                    "f1": metrics.get("F1", 0.0),
                    "precision": metrics.get("Precision", 0.0),
                    "recall": metrics.get("Recall", 0.0),
                    "tp": metrics.get("TP", 0),
                    "fp": metrics.get("FP", 0),
                    "fn": metrics.get("FN", 0),
                    "metrics": metrics,
                    "annotated_df": annotated_df
                }
                log.info(f"    ✓ GPT F1: {eval_results['gpt']['f1']:.4f}")
        except Exception as e:
            log.error(f"    GPT evaluation failed: {e}")
            eval_results["gpt"] = None
    
    if gt_df is None:
        log.info("  Ground truth missing — preparing fallback rows")
        fallback_rows = _combine_consensus_and_verified(state)
        log.info(f"  ✓ Fallback rows: {len(fallback_rows)}")
        return {
            "eval_gemini": None,
            "eval_gpt": None,
            "fallback_rows": fallback_rows
        }

    return {
        "eval_gemini": eval_results.get("gemini"),
        "eval_gpt": eval_results.get("gpt")
    }


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK HELPERS

def _canonical_dedup_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for name in ["material_name", "property_name", "value", "unit"]:
        if name in df.columns:
            cols.append(name)
    return cols


def _combine_consensus_and_verified(state: PDFMonitorState) -> pd.DataFrame:
    dfs = []
    if state.get("df_consensus") is not None and len(state["df_consensus"]) > 0:
        dfs.append(state["df_consensus"].copy())
    if state.get("df_gemini") is not None and len(state["df_gemini"]) > 0:
        gem_verified = state["df_gemini"][state["df_gemini"].get("source_verified", pd.Series(dtype=bool)) == True].copy()
        dfs.append(gem_verified)
    if state.get("df_gpt") is not None and len(state["df_gpt"]) > 0:
        gpt_verified = state["df_gpt"][state["df_gpt"].get("source_verified", pd.Series(dtype=bool)) == True].copy()
        dfs.append(gpt_verified)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    dedup_cols = _canonical_dedup_columns(combined)
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep="first")
    return combined.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5: Pick winner by comparing F1 scores
# ─────────────────────────────────────────────────────────────────────────────

def pick_winner(state: PDFMonitorState) -> dict:
    """
    Compare F1 scores, select LLM with higher F1.
    If GT is missing, route to fallback consensus+verified results.
    """
    log.info(f"\n[Node 5] Picking winner...")

    if state["gt_df"] is None:
        fallback_size = len(state.get("fallback_rows") or [])
        log.info(f"  No GT available — using fallback consensus+verified rows ({fallback_size})")
        return {
            "winner_name": "Consensus+Verified",
            "winner_eval": None
        }

    f1_gemini = state["eval_gemini"].get("f1", 0.0) if state["eval_gemini"] else 0.0
    f1_gpt = state["eval_gpt"].get("f1", 0.0) if state["eval_gpt"] else 0.0
    
    if f1_gemini > f1_gpt:
        log.info(f"  🏆 Gemini wins! (F1: {f1_gemini:.4f} vs GPT: {f1_gpt:.4f})")
        return {
            "winner_name": "Gemini",
            "winner_eval": state["eval_gemini"]
        }
    elif f1_gpt > f1_gemini:
        log.info(f"  🏆 GPT wins! (F1: {f1_gpt:.4f} vs Gemini: {f1_gemini:.4f})")
        return {
            "winner_name": "GPT",
            "winner_eval": state["eval_gpt"]
        }
    else:
        log.info(f"  🤝 Tie! Both F1: {f1_gemini:.4f}")
        return {
            "winner_name": "Gemini (tie)",
            "winner_eval": state["eval_gemini"]
        }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6: Save winner's TP results to CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_results(state: PDFMonitorState) -> dict:
    """
    Extract TP rows from winner, append to results_finalized.csv.
    """
    log.info(f"\n[Node 6] Saving results...")
    
    try:
        if state["gt_df"] is None:
            fallback_df = state.get("fallback_rows") or pd.DataFrame()
            if fallback_df.empty:
                log.warning(f"  ⚠ No fallback rows to save")
                return {}

            save_df = fallback_df.copy()
            save_df["pdf_filename"] = state["current_pdf"].name
            save_df["gt_filename"] = state["current_pdf"].stem + "_gt_missing"
            save_df["winning_llm"] = state["winner_name"]
            save_df["processing_timestamp"] = datetime.now().isoformat()
            save_df["f1_score"] = None
            save_df["precision"] = None
            save_df["recall"] = None
        else:
            annotated_df = state["winner_eval"]["annotated_df"]
            tp_rows = annotated_df[annotated_df["eval_result"] == "TP"].copy()
            
            if len(tp_rows) == 0:
                log.warning(f"  ⚠ No TP rows to save")
                return {}
            
            save_df = tp_rows
            save_df["pdf_filename"] = state["current_pdf"].name
            save_df["gt_filename"] = state["current_pdf"].stem + "_gt"
            save_df["winning_llm"] = state["winner_name"]
            save_df["processing_timestamp"] = datetime.now().isoformat()
            save_df["f1_score"] = state["winner_eval"]["f1"]
            save_df["precision"] = state["winner_eval"]["precision"]
            save_df["recall"] = state["winner_eval"]["recall"]

        # Append to CSV
        if RESULTS_FINALIZED_CSV.exists():
            existing_df = pd.read_csv(RESULTS_FINALIZED_CSV)
            combined_df = pd.concat([existing_df, save_df], ignore_index=True)
        else:
            combined_df = save_df
        
        combined_df.to_csv(RESULTS_FINALIZED_CSV, index=False)
        log.info(f"  ✓ Saved {len(save_df)} rows to {RESULTS_FINALIZED_CSV.name}")
        
        return {}
        
    except Exception as e:
        log.error(f"  ✗ Failed to save results: {e}")
        return {"errors": [f"Save error: {e}"]}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 7: Check for next PDF or END
# ─────────────────────────────────────────────────────────────────────────────

def check_next(state: PDFMonitorState) -> dict:
    """
    Mark current PDF as processed.
    If more PDFs pending, loop back to scan_folder.
    Otherwise END.
    """
    log.info(f"\n[Node 7] Checking for next PDF...")
    
    # Mark as processed
    state["processed_files"].add(state["current_pdf"].name)
    save_processing_state(state["processed_files"])
    log.info(f"  ✓ Marked as processed: {state['current_pdf'].name}")
    
    # Check if more PDFs
    remaining = len([p for p in state["pending_pdfs"] if p.name not in state["processed_files"]])
    log.info(f"  Remaining PDFs: {remaining}")
    
    return {"processed_files": state["processed_files"]}


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH (like build_graph in crawler_graph.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """Build LangGraph for PDF monitoring agent."""
    graph = StateGraph(PDFMonitorState)
    
    # Add nodes
    graph.add_node("scan_folder", scan_folder)
    graph.add_node("load_ground_truth", load_ground_truth)
    graph.add_node("extract_dual_llm", extract_dual_llm)
    graph.add_node("evaluate_both", evaluate_both)
    graph.add_node("pick_winner", pick_winner)
    graph.add_node("save_results", save_results)
    graph.add_node("check_next", check_next)
    
    # Add edges (linear flow + loop back)
    graph.set_entry_point("scan_folder")
    graph.add_edge("scan_folder", "load_ground_truth")
    graph.add_edge("load_ground_truth", "extract_dual_llm")
    graph.add_edge("extract_dual_llm", "evaluate_both")
    graph.add_edge("evaluate_both", "pick_winner")
    graph.add_edge("pick_winner", "save_results")
    graph.add_edge("save_results", "check_next")
    
    # Conditional edge: if more PDFs, loop back; else END
    def should_loop(state: PDFMonitorState) -> str:
        remaining = [p for p in state["pending_pdfs"] if p.name not in state["processed_files"]]
        if remaining:
            return "scan_folder"
        else:
            return END
    
    graph.add_conditional_edges("check_next", should_loop)
    
    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# RUN AGENT (like run_crawler in crawler_graph.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_monitor_agent():
    """Run the PDF monitoring agent."""
    log.info(f"\n{'='*80}")
    log.info(f"🚀 Starting PDF Monitor Agent")
    log.info(f"{'='*80}")
    log.info(f"  Naming Convention: {{pdf_name}}.pdf → {{pdf_name}}_gt.csv")
    log.info(f"  Folder: {PDF_MONITOR_ROOT.resolve()}")
    
    setup_folders()
    processed_files = load_processing_state()
    
    graph = build_graph()
    
    result = graph.invoke({
        "current_pdf": None,
        "pdf_bytes": b"",
        "gt_df": None,
        "df_gemini": None,
        "df_gpt": None,
        "eval_gemini": None,
        "eval_gpt": None,
        "winner_eval": None,
        "winner_name": "",
        "processed_files": processed_files,
        "pending_pdfs": [],
        "errors": [],
    })
    
    log.info(f"\n{'='*80}")
    log.info(f"✓ Agent execution complete!")
    log.info(f"{'='*80}\n")
    
    return result


if __name__ == "__main__":
    run_monitor_agent()
