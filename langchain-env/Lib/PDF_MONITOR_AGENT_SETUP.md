# 🤖 PDF Monitor Agent - Setup & Usage Guide

**LangGraph Agent** (like `crawler_graph.py`) for autonomous PDF processing.

## Overview

Standalone agent with **7 LangGraph nodes**:
1. **scan_folder** - check for new PDFs
2. **load_ground_truth** - load GT using naming convention
3. **extract_dual_llm** - extract with Gemini + GPT (calls `run_pipeline` directly)
4. **evaluate_both** - evaluate both extractions (calls `_ev_score` directly)
5. **pick_winner** - compare F1 scores
6. **save_results** - save winner's TP rows to CSV
7. **check_next** - loop back or END

**✓ Fully isolated - no connection to crawler_graph**  
**✓ LangGraph StateGraph with proper nodes/edges**  
**✓ Direct imports: ONLY `run_pipeline` + `_ev_score` from existing code**

---

## Folder Structure

```
pdf_monitor/
├── pdfs/                    ← Drop PDFs here
├── ground_truth/            ← Ground truth files (naming convention)
└── results_db/              ← Results auto-saved here
    ├── results_finalized.csv
    └── processing_log.json
```

### Setup Command

```powershell
cd c:\Users\varam\langchain-env\Lib

# Create folders
mkdir pdf_monitor\pdfs
mkdir pdf_monitor\ground_truth
mkdir pdf_monitor\results_db
```

---

## How to Use

### Step 1: Naming Convention

Each PDF automatically finds its ground truth using this pattern:

```
PDF File              →    Ground Truth File
──────────────────────────────────────────────
sample1.pdf          →    sample1_gt.csv
material_test.pdf    →    material_test_gt.xlsx
batch_02.pdf         →    batch_02_gt.csv
```

**Fallback:** If `{pdf_name}_gt.csv` is NOT found, tries generic `ground_truth.csv`

### Step 2: Prepare Ground Truth

Create ground truth files in `pdf_monitor/ground_truth/` with naming convention:

**Example: `sample1_gt.csv`**
```csv
property_name,value,unit,material,confidence
Tensile Strength,250,MPa,Steel,3
Young's Modulus,210,GPa,Steel,3
Density,7850,kg/m³,Steel,3
```

**Supported formats:** `.csv`, `.xlsx`, `.xls`

### Step 3: Add PDFs

Drop PDF files into `pdf_monitor/pdfs/` with matching GT files:

```
pdf_monitor/
├── pdfs/
│   ├── sample1.pdf              ← matches sample1_gt.csv
│   ├── sample2.pdf              ← matches sample2_gt.csv
│   └── material_test.pdf        ← matches material_test_gt.xlsx
│
└── ground_truth/
    ├── sample1_gt.csv
    ├── sample2_gt.csv
    └── material_test_gt.xlsx
```

### Step 4: Run Agent

```powershell
cd c:\Users\varam\langchain-env\Lib
python pdf_monitor_agent.py
```

**Expected output:**
```
INFO │ ════════════════════════════════════════════════════════════════════════════════
INFO │ 🚀 Starting PDF Monitor Agent
INFO │ ════════════════════════════════════════════════════════════════════════════════
INFO │   Naming Convention: {pdf_name}.pdf → {pdf_name}_gt.csv
INFO │   Folder: C:\...\pdf_monitor

INFO │ [Node 1] Scanning PDF folder...
INFO │   Found 2 unprocessed PDF(s)
INFO │   Next: sample1.pdf

INFO │ [Node 2] Loading ground truth for sample1.pdf...
INFO │   ✓ Loaded: sample1_gt.csv (3 rows)

INFO │ [Node 3] Extracting with Gemini + GPT from sample1.pdf...
INFO │   ✓ Gemini: 45 rows
INFO │   ✓ GPT: 42 rows

INFO │ [Node 4] Evaluating extractions...
INFO │   Scoring Gemini...
INFO │     ✓ Gemini F1: 0.8542
INFO │   Scoring GPT...
INFO │     ✓ GPT F1: 0.7895

INFO │ [Node 5] Picking winner...
INFO │   🏆 Gemini wins! (F1: 0.8542 vs GPT: 0.7895)

INFO │ [Node 6] Saving results...
INFO │   ✓ Saved 25 TP rows to results_finalized.csv

INFO │ [Node 7] Checking for next PDF...
INFO │   ✓ Marked as processed: sample1.pdf
INFO │   Remaining PDFs: 1

INFO │ [Node 1] Scanning PDF folder...
INFO │   Found 1 unprocessed PDF(s)
INFO │   Next: sample2.pdf

... (repeats for sample2.pdf)

INFO │ [Node 1] Scanning PDF folder...
INFO │   No unprocessed PDFs found (total: 2)

INFO │ ════════════════════════════════════════════════════════════════════════════════
INFO │ ✓ Agent execution complete!
INFO │ ════════════════════════════════════════════════════════════════════════════════
```

### Step 5: Check Results

Results are saved to `pdf_monitor/results_db/results_finalized.csv`:

```csv
property_name,value,unit,eval_result,pdf_filename,winning_llm,f1_score,precision,recall,...
Tensile Strength,250,MPa,TP,sample1.pdf,Gemini,0.8542,0.875,0.8333,...
Young's Modulus,210,GPa,TP,sample1.pdf,Gemini,0.8542,0.875,0.8333,...
```

---

## Architecture (LangGraph Nodes)

```
┌─────────────┐
│ scan_folder │ ← Entry point
└──────┬──────┘
       ↓
┌──────────────────┐
│load_ground_truth │
└──────┬───────────┘
       ↓
┌────────────────┐
│extract_dual_llm│ ← Calls run_pipeline()
└──────┬─────────┘
       ↓
┌─────────────────┐
│ evaluate_both   │ ← Calls _ev_score()
└──────┬──────────┘
       ↓
┌───────────────┐
│ pick_winner   │
└──────┬────────┘
       ↓
┌─────────────────┐
│ save_results    │
└──────┬──────────┘
       ↓
┌──────────────┐
│ check_next   │
└──────┬───────┘
       │
   [Loop?]
   /      \
  YES     NO
  │       │
  ↓       ↓
scan  →  END
```

**Key differences from pipeline script:**
- ✅ `StateGraph` + 7 nodes (like crawler_graph)
- ✅ Direct calls to `run_pipeline()` + `_ev_score()` (no wrappers)
- ✅ Conditional looping via `should_loop()` function
- ✅ State flows through all nodes (`PDFMonitorState`)

---

## Configuration

Edit these variables in `pdf_monitor_agent.py`:

```python
# Folder paths
PDF_MONITOR_ROOT = Path("./pdf_monitor")
PDF_FOLDER = PDF_MONITOR_ROOT / "pdfs"
GT_FOLDER = PDF_MONITOR_ROOT / "ground_truth"
DB_FOLDER = PDF_MONITOR_ROOT / "results_db"

# Results file
RESULTS_FINALIZED_CSV = DB_FOLDER / "results_finalized.csv"
PROCESSING_LOG_JSON = DB_FOLDER / "processing_log.json"
```

**Naming Convention (automatic):**
- PDF: `sample1.pdf` → GT lookup: `sample1_gt.csv`, `sample1_gt.xlsx`, `sample1_gt.xls`
- Fallback: If not found, tries `ground_truth.csv`, `ground_truth.xlsx`, `ground_truth.xls`

---

## State Definition

`PDFMonitorState` (like `CrawlerState` in crawler_graph.py):

```python
class PDFMonitorState(TypedDict):
    current_pdf:       Optional[Path]      # PDF being processed
    pdf_bytes:         bytes               # PDF file content
    gt_df:             object              # ground truth dataframe
    df_gemini:         object              # Gemini extraction
    df_gpt:            object              # GPT extraction
    eval_gemini:       Optional[Dict]      # Gemini evaluation
    eval_gpt:          Optional[Dict]      # GPT evaluation
    winner_eval:       Optional[Dict]      # winning LLM eval
    winner_name:       str                 # "Gemini" or "GPT"
    processed_files:   set                 # which PDFs processed
    pending_pdfs:      List[Path]          # remaining PDFs
    errors:            Annotated[List[str], operator.add]
```

---

## Troubleshooting

### Issue: "No ground truth found for sample1.pdf"
**Solution:**
- Create matching GT file: `sample1_gt.csv` (exact name!)
- Place in `pdf_monitor/ground_truth/`
- Supported formats: `.csv`, `.xlsx`, `.xls`
- Fallback: Create `ground_truth.csv` for all PDFs

### Issue: "Both extractions returned 0 rows"
**Solution:**
- Check PDF isn't image-only
- Verify GEMINI_API_KEY + OPENAI_API_KEY in `.env`
- Check network connection

### Issue: Agent never processes PDFs
**Solution:**
- Verify PDF filenames in `pdf_monitor/pdfs/`
- Check folder permissions (can agent read/write?)
- Delete `processing_log.json` to reset state

---

## Code Structure

**Only imports from existing code:**
```python
from DocToDB_eval_v2 import run_pipeline, _ev_score, _find_col, _norm_name
```

**NEW code (7 nodes + agent orchestration):**
- Node functions (scan_folder, load_ground_truth, extract_dual_llm, etc.)
- PDFMonitorState TypedDict
- build_graph() - creates StateGraph
- run_monitor_agent() - executes agent

**NO wrappers - direct calls:**
- `extract_dual_llm()` → calls `run_pipeline()` directly
- `evaluate_both()` → calls `_ev_score()` directly

---

## Integration

**Separate from crawler_graph:**
```
Lib/
├── crawler_graph.py      ← Unchanged
├── pdf_monitor_agent.py  ← New LangGraph agent
├── DocToDB_eval_v2.py    ← Unchanged (provides run_pipeline, _ev_score)
└── ...
```

Run independently:
```powershell
python pdf_monitor_agent.py      # PDF agent
# (in another terminal)
python crawler_graph.py           # Crawler agent
```

Both can run simultaneously without interference.

---

## Next Steps

1. **Create folder structure** (see Step 1)
2. **Prepare ground truth** → `pdf_monitor/ground_truth/sample1_gt.csv`
3. **Add test PDF** → `pdf_monitor/pdfs/sample1.pdf`
4. **Run agent** → `python pdf_monitor_agent.py`
5. **Check results** → `pdf_monitor/results_db/results_finalized.csv`

---

**Questions?** Check console output for detailed error messages. All operations logged with timestamps.
