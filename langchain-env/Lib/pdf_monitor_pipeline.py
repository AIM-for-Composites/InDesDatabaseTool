"""
PDF Monitor & Evaluation Pipeline
==================================
Autonomous automation for:
1. Monitor PDF folder for new files (one at a time)
2. Extract with Gemini + GPT in parallel
3. Run evaluation against ground truth
4. Compare F1 scores → pick winner
5. Save only winner's TP results to results_finalized.csv
6. Loop back to check next PDF

NO connection to crawler_graph. Fully standalone.
"""

import os
import json
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, List

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS FROM EXISTING MODULES
# ─────────────────────────────────────────────────────────────────────────────

from DocToDB_eval_v2 import _ev_score, _find_col, _norm_name, run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s"
)
log = logging.getLogger(__name__)

# Folder structure (user-configurable)
PDF_MONITOR_ROOT = Path("./pdf_monitor")  # New root folder for PDFs
PDF_FOLDER = PDF_MONITOR_ROOT / "pdfs"    # Subfolder with PDFs to process
GT_FOLDER = PDF_MONITOR_ROOT / "ground_truth"  # Ground truth CSVs
DB_FOLDER = PDF_MONITOR_ROOT / "results_db"    # Results database folder

# Results tracking
RESULTS_FINALIZED_CSV = DB_FOLDER / "results_finalized.csv"
PROCESSING_LOG_JSON = DB_FOLDER / "processing_log.json"

# Processing state
PROCESSED_FILES = set()

# Poll interval (seconds)
POLL_INTERVAL = 5


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_folders():
    """Create required folder structure."""
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)
    GT_FOLDER.mkdir(parents=True, exist_ok=True)
    DB_FOLDER.mkdir(parents=True, exist_ok=True)
    log.info(f"✓ Folders ready:")
    log.info(f"  PDF Monitor Root: {PDF_MONITOR_ROOT.resolve()}")
    log.info(f"  PDFs to process: {PDF_FOLDER.resolve()}")
    log.info(f"  Ground Truth:    {GT_FOLDER.resolve()}")
    log.info(f"  Results DB:      {DB_FOLDER.resolve()}")


def load_processing_state():
    """Load which files have been processed."""
    global PROCESSED_FILES
    if PROCESSING_LOG_JSON.exists():
        try:
            with open(PROCESSING_LOG_JSON, "r") as f:
                data = json.load(f)
                PROCESSED_FILES = set(data.get("processed_files", []))
                log.info(f"Loaded {len(PROCESSED_FILES)} previously processed files")
        except Exception as e:
            log.warning(f"Failed to load processing state: {e}")
    return PROCESSED_FILES


def save_processing_state():
    """Save which files have been processed."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "processed_files": list(PROCESSED_FILES)
    }
    with open(PROCESSING_LOG_JSON, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(pdf_filename: str) -> Optional[pd.DataFrame]:
    """
    Load ground truth CSV/Excel from GT folder.
    Naming convention: {pdf_name}_gt.{ext}
    
    Example:
    - PDF: sample1.pdf → looks for: sample1_gt.csv
    - PDF: material_test.pdf → looks for: material_test_gt.xlsx
    
    Fallback: If {pdf_name}_gt not found, tries generic ground_truth.{ext}
    """
    # Extract PDF name without extension (e.g., "sample1.pdf" → "sample1")
    pdf_stem = Path(pdf_filename).stem
    gt_base = f"{pdf_stem}_gt"
    
    # Try {pdf_name}_gt.{ext} first
    for ext in [".csv", ".xlsx", ".xls"]:
        gt_path = GT_FOLDER / f"{gt_base}{ext}"
        if gt_path.exists():
            try:
                if ext == ".csv":
                    gt_df = pd.read_csv(gt_path)
                else:
                    gt_df = pd.read_excel(gt_path)
                log.info(f"✓ Loaded ground truth: {gt_path.name} ({len(gt_df)} rows)")
                return gt_df
            except Exception as e:
                log.error(f"Failed to load GT {gt_path.name}: {e}")
                return None
    
    # Fallback to generic ground_truth.{ext}
    log.info(f"  ('{gt_base}' not found, trying fallback...)")
    for ext in [".csv", ".xlsx", ".xls"]:
        gt_path = GT_FOLDER / f"ground_truth{ext}"
        if gt_path.exists():
            try:
                if ext == ".csv":
                    gt_df = pd.read_csv(gt_path)
                else:
                    gt_df = pd.read_excel(gt_path)
                log.info(f"✓ Loaded fallback ground truth: {gt_path.name} ({len(gt_df)} rows)")
                return gt_df
            except Exception as e:
                log.error(f"Failed to load fallback GT: {e}")
                return None
    log.error(f"  Tried: {gt_base}.csv/.xlsx/.xls or ground_truth.csv/.xlsx/.xls")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT WITH DUAL LLM (Gemini + GPT)
# ─────────────────────────────────────────────────────────────────────────────

def extract_dual_llm(pdf_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Extract with both Gemini and GPT using run_pipeline (single call, dual LLM inside).
    Returns (df_consensus, df_gemini, df_gpt).

    run_pipeline internally:
    - Extracts chunks once
    - Runs Gemini + GPT in parallel
    - Produces a consensus DataFrame plus per-LLM DataFrames
    """
    log.info(f"\n[EXTRACT] Processing {pdf_path.name}...")
    
    try:
        pdf_bytes = pdf_path.read_bytes()
        
        # Single call returns consensus, Gemini + GPT results
        df_consensus, df_gemini, df_gpt, chunks, errors, meta = run_pipeline(pdf_bytes)
        
        if errors:
            log.warning(f"  ⚠ Extraction errors: {errors}")
        
        if df_gemini is None or len(df_gemini) == 0:
            log.warning(f"  ⚠ Gemini extracted 0 rows")
            df_gemini = None
        else:
            df_gemini["llm_model"] = "Gemini"
            log.info(f"✓ Gemini: {len(df_gemini)} properties extracted")
        
        if df_gpt is None or len(df_gpt) == 0:
            log.warning(f"  ⚠ GPT extracted 0 rows")
            df_gpt = None
        else:
            df_gpt["llm_model"] = "GPT"
            log.info(f"✓ GPT: {len(df_gpt)} properties extracted")
        
        if df_consensus is None or len(df_consensus) == 0:
            log.info(f"  Consensus: 0 rows")
        else:
            log.info(f"✓ Consensus: {len(df_consensus)} properties agreed by both LLMs")
        
        if df_gemini is None and df_gpt is None:
            log.error("✗ Both extractions returned 0 rows")
        
        return df_consensus, df_gemini, df_gpt
        
    except Exception as e:
        log.error(f"  ✗ Extraction failed: {e}")
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_extraction(
    model_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    model_name: str
) -> Dict:
    """
    Run evaluation against ground truth using _ev_score.
    Returns: {
        "model": model_name,
        "f1": float,
        "precision": float,
        "recall": float,
        "tp": int,
        "fp": int,
        "fn": int,
        "annotated_df": pd.DataFrame
    }
    """
    try:
        log.info(f"  [EVAL] Scoring {model_name} extraction...")
        
        # Detect column names in model and GT dataframes
        nc = _find_col(model_df, ["property_name", "property", "name"])
        vc = _find_col(model_df, ["value", "val", "measured_value"])
        uc = _find_col(model_df, ["unit", "units", "si_unit"])
        
        gnc = _find_col(gt_df, ["property_name", "property", "name"])
        gvc = _find_col(gt_df, ["value", "val", "measured_value"])
        guc = _find_col(gt_df, ["unit", "units", "si_unit"])
        
        if not (nc and vc and gnc and gvc):
            log.warning(f"    ⚠ Could not detect all required columns")
            log.warning(f"      Model: name={nc}, value={vc}, unit={uc}")
            log.warning(f"      GT: name={gnc}, value={gvc}, unit={guc}")
            return None
        
        # Run evaluation
        annotated_df, metrics = _ev_score(
            model_df=model_df,
            gt_df=gt_df,
            nc=nc, vc=vc, uc=uc,
            gnc=gnc, gvc=gvc, guc=guc,
            min_conf=1
        )
        
        result = {
            "model": model_name,
            "f1": metrics.get("F1", 0.0),
            "precision": metrics.get("Precision", 0.0),
            "recall": metrics.get("Recall", 0.0),
            "tp": metrics.get("TP", 0),
            "fp": metrics.get("FP", 0),
            "fn": metrics.get("FN", 0),
            "metrics": metrics,
            "annotated_df": annotated_df
        }
        
        log.info(f"    ✓ {model_name} F1 Score: {result['f1']:.4f}")
        log.info(f"      Precision: {result['precision']:.4f} | Recall: {result['recall']:.4f}")
        log.info(f"      TP: {result['tp']} | FP: {result['fp']} | FN: {result['fn']}")
        
        return result
        
    except Exception as e:
        log.error(f"    ✗ Evaluation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PICK WINNER & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def pick_winner(eval_gemini: Dict, eval_gpt: Dict) -> Tuple[Dict, str]:
    """
    Compare F1 scores, pick winner.
    Returns: (winning_eval_result, winner_name)
    """
    f1_gemini = eval_gemini.get("f1", 0.0) if eval_gemini else 0.0
    f1_gpt = eval_gpt.get("f1", 0.0) if eval_gpt else 0.0
    
    if f1_gemini > f1_gpt:
        log.info(f"\n[WINNER] 🏆 Gemini wins! (F1: {f1_gemini:.4f} vs GPT: {f1_gpt:.4f})")
        return eval_gemini, "Gemini"
    elif f1_gpt > f1_gemini:
        log.info(f"\n[WINNER] 🏆 GPT wins! (F1: {f1_gpt:.4f} vs Gemini: {f1_gemini:.4f})")
        return eval_gpt, "GPT"
    else:
        log.info(f"\n[WINNER] 🤝 Tie! Both F1: {f1_gemini:.4f}")
        return eval_gemini, "Gemini (tie)"


def save_winner_results(
    pdf_filename: str,
    eval_result: Dict,
    winner_name: str,
    gt_filename: str
):
    """
    Extract TP rows from winner's annotated DataFrame.
    Append to results_finalized.csv with metadata.
    """
    try:
        annotated_df = eval_result["annotated_df"]
        tp_rows = annotated_df[annotated_df["eval_result"] == "TP"].copy()
        
        if len(tp_rows) == 0:
            log.warning(f"  ⚠ No TP rows to save for {pdf_filename}")
            return
        
        # Add metadata
        tp_rows["pdf_filename"] = pdf_filename
        tp_rows["gt_filename"] = gt_filename
        tp_rows["winning_llm"] = winner_name
        tp_rows["processing_timestamp"] = datetime.now().isoformat()
        tp_rows["f1_score"] = eval_result["f1"]
        tp_rows["precision"] = eval_result["precision"]
        tp_rows["recall"] = eval_result["recall"]
        
        # Append to results CSV
        if RESULTS_FINALIZED_CSV.exists():
            existing_df = pd.read_csv(RESULTS_FINALIZED_CSV)
            combined_df = pd.concat([existing_df, tp_rows], ignore_index=True)
        else:
            combined_df = tp_rows
        
        combined_df.to_csv(RESULTS_FINALIZED_CSV, index=False)
        log.info(f"  ✓ Saved {len(tp_rows)} TP rows to {RESULTS_FINALIZED_CSV.name}")
        
    except Exception as e:
        log.error(f"  ✗ Failed to save results: {e}")


# -----------------------------------------------------------------------------
# Fallback helpers: combine consensus + source_verified rows when GT missing
# -----------------------------------------------------------------------------
def _canonical_dedup_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for name in ["material_name", "property_name", "value", "unit"]:
        if name in df.columns:
            cols.append(name)
    return cols


def _combine_consensus_and_verified(df_consensus: Optional[pd.DataFrame],
                                    df_gemini: Optional[pd.DataFrame],
                                    df_gpt: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return combined consensus + source_verified rows without duplicates."""
    dfs = []
    if df_consensus is not None and len(df_consensus) > 0:
        dfs.append(df_consensus.copy())
    if df_gemini is not None and len(df_gemini) > 0:
        gem_verified = df_gemini[df_gemini.get("source_verified", pd.Series(dtype=bool)) == True].copy() if "source_verified" in df_gemini.columns else pd.DataFrame()
        if not gem_verified.empty:
            dfs.append(gem_verified)
    if df_gpt is not None and len(df_gpt) > 0:
        gpt_verified = df_gpt[df_gpt.get("source_verified", pd.Series(dtype=bool)) == True].copy() if "source_verified" in df_gpt.columns else pd.DataFrame()
        if not gpt_verified.empty:
            dfs.append(gpt_verified)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    dedup_cols = _canonical_dedup_columns(combined)
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep="first")
    return combined.reset_index(drop=True)


def save_fallback_results(pdf_filename: str, df: pd.DataFrame, winner_name: str, gt_filename: str):
    """Save fallback DataFrame rows to results CSV with metadata."""
    try:
        save_df = df.copy()
        save_df["pdf_filename"] = pdf_filename
        save_df["gt_filename"] = gt_filename
        save_df["winning_llm"] = winner_name
        save_df["processing_timestamp"] = datetime.now().isoformat()
        save_df["f1_score"] = None
        save_df["precision"] = None
        save_df["recall"] = None

        if RESULTS_FINALIZED_CSV.exists():
            existing_df = pd.read_csv(RESULTS_FINALIZED_CSV)
            combined_df = pd.concat([existing_df, save_df], ignore_index=True)
        else:
            combined_df = save_df

        combined_df.to_csv(RESULTS_FINALIZED_CSV, index=False)
        log.info(f"  ✓ Saved {len(save_df)} fallback rows to {RESULTS_FINALIZED_CSV.name}")
    except Exception as e:
        log.error(f"  ✗ Failed to save fallback results: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MONITORING & PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def get_next_pdf() -> Optional[Path]:
    """
    Get next unprocessed PDF from PDF folder.
    Returns first PDF file that hasn't been processed yet.
    """
    try:
        pdfs = sorted([f for f in PDF_FOLDER.glob("*.pdf") if f.is_file()])
        for pdf in pdfs:
            if pdf.name not in PROCESSED_FILES:
                return pdf
    except Exception as e:
        log.error(f"Error scanning PDF folder: {e}")
    
    return None


def process_pdf(pdf_path: Path):
    """
    Process a single PDF:
    1. Load ground truth by PDF name (e.g., sample1.pdf → sample1_gt.csv)
    2. Extract with Gemini + GPT
    3. Evaluate both
    4. Pick winner
    5. Save TP rows
    6. Mark as processed
    """
    log.info(f"\n{'='*80}")
    log.info(f"📄 PROCESSING PDF: {pdf_path.name}")
    log.info(f"{'='*80}")
    
    # Extract with both LLMs (and consensus)
    df_consensus, df_gemini, df_gpt = extract_dual_llm(pdf_path)

    if df_gemini is None and df_gpt is None and (df_consensus is None or len(df_consensus) == 0):
        log.error("✗ Both extractions failed. Skipping PDF.")
        PROCESSED_FILES.add(pdf_path.name)
        save_processing_state()
        return False

    # Load ground truth using PDF name convention
    gt_df = load_ground_truth(pdf_path.name)

    # If no GT, use consensus + source_verified fallback
    if gt_df is None or len(gt_df) == 0:
        log.warning("✗ No ground truth loaded — using consensus+verified fallback if available")
        fallback_df = _combine_consensus_and_verified(df_consensus, df_gemini, df_gpt)
        if fallback_df is None or len(fallback_df) == 0:
            log.error("✗ No fallback rows found. Skipping PDF.")
            PROCESSED_FILES.add(pdf_path.name)
            save_processing_state()
            return False
        # Save fallback rows
        save_fallback_results(pdf_path.name, fallback_df, "Consensus+Verified", pdf_path.stem + "_gt_missing")
        PROCESSED_FILES.add(pdf_path.name)
        save_processing_state()
        log.info(f"\n✓ PDF processing complete (fallback saved)!\n")
        return True

    # (Normal path) Evaluate both
    log.info(f"\n[EVALUATE]")
    eval_results = []

    if df_gemini is not None:
        eval_gemini = evaluate_extraction(df_gemini, gt_df, "Gemini")
        if eval_gemini:
            eval_results.append(("Gemini", eval_gemini))

    if df_gpt is not None:
        eval_gpt = evaluate_extraction(df_gpt, gt_df, "GPT")
        if eval_gpt:
            eval_results.append(("GPT", eval_gpt))

    if len(eval_results) == 0:
        log.error("✗ No successful evaluations. Skipping PDF.")
        return False

    # Pick winner and save
    if len(eval_results) == 2:
        winner_eval, winner_name = pick_winner(eval_results[0][1], eval_results[1][1])
    else:
        winner_name, winner_eval = eval_results[0]
        log.info(f"\n[WINNER] Only {winner_name} succeeded.")

    log.info(f"\n[SAVE]")
    # Derive which GT was used (same naming convention as load_ground_truth)
    pdf_stem = pdf_path.stem
    save_winner_results(pdf_path.name, winner_eval, winner_name, pdf_stem)

    # Mark as processed
    PROCESSED_FILES.add(pdf_path.name)
    save_processing_state()

    log.info(f"\n✓ PDF processing complete!\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_monitor_loop():
    """
    Main loop: continuously check for new PDFs and process one at a time.
    Each PDF automatically finds its ground truth via naming convention.
    """
    log.info(f"\n🚀 Starting PDF Monitor Pipeline")
    log.info(f"   Naming Convention: {{pdf_name}}.pdf → {{pdf_name}}_gt.csv")
    log.info(f"   Poll Interval: {POLL_INTERVAL}s")
    log.info(f"   Press Ctrl+C to stop\n")
    
    try:
        while True:
            next_pdf = get_next_pdf()
            
            if next_pdf:
                process_pdf(next_pdf)
            else:
                log.info(f"[IDLE] No new PDFs found. Checking again in {POLL_INTERVAL}s...")
                time.sleep(POLL_INTERVAL)
    
    except KeyboardInterrupt:
        log.info(f"\n⏹ Pipeline stopped by user")
        save_processing_state()
    except Exception as e:
        log.error(f"\n✗ Pipeline error: {e}")
        save_processing_state()
        raise


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_folders()
    load_processing_state()
    
    # Run the monitoring loop
    # Naming convention: {pdf_name}.pdf → {pdf_name}_gt.csv
    run_monitor_loop()
