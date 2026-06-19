# 🤖 PDF Monitor Pipeline - Setup & Usage Guide

## Overview
Standalone automation pipeline that:
1. **Monitors** a PDF folder for new files (one at a time)
2. **Extracts** properties using Gemini + GPT in parallel
3. **Evaluates** both extractions against ground truth
4. **Compares** F1 scores → picks winner
5. **Finalizes** only winner's TP results to `results_finalized.csv`
6. **Loops** continuously for next PDF

**✓ Fully isolated - does NOT connect to crawler_graph**

---

## Folder Structure

Create this folder structure in your workspace:

```
pdf_monitor/
├── pdfs/                    ← Drop PDFs here
├── ground_truth/            ← Ground truth CSV/Excel files
└── results_db/              ← Results saved here
    ├── results_finalized.csv
    └── processing_log.json
```

### Setup Command

```powershell
# Create the folders
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
─────────────────────────────────────────────
sample1.pdf          →    sample1_gt.csv
material_test.pdf    →    material_test_gt.xlsx
batch_02.pdf         →    batch_02_gt.csv
```

**Fallback:** If `{pdf_name}_gt.csv` is NOT found, the pipeline tries generic `ground_truth.csv`

### Step 2: Prepare Ground Truth

Create ground truth files in `pdf_monitor/ground_truth/` with naming convention:

**Examples:**

`sample1_gt.csv`:
```
property_name,value,unit,material,confidence
Tensile Strength,250,MPa,Steel,3
Young's Modulus,210,GPa,Steel,3
```

`material_test_gt.xlsx`:
```
property_name  | value  | unit   | material | confidence
───────────────┼────────┼────────┼──────────┼───────────
Tensile Strength| 250   | MPa    | Steel    | 3
Young's Modulus | 210   | GPa    | Steel    | 3
```

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

### Step 4: Run Pipeline

```powershell
cd c:\Users\varam\langchain-env\Lib

python pdf_monitor_pipeline.py
```

**Expected output:**
```
INFO │ ✓ Folders ready:
INFO │   PDF Monitor Root: C:\...\pdf_monitor
INFO │   PDFs to process: C:\...\pdf_monitor\pdfs
INFO │   Ground Truth:    C:\...\pdf_monitor\ground_truth
INFO │   Results DB:      C:\...\pdf_monitor\results_db

INFO │ 🚀 Starting PDF Monitor Pipeline
INFO │    Naming Convention: {pdf_name}.pdf → {pdf_name}_gt.csv
INFO │    Poll Interval: 5s
INFO │    Press Ctrl+C to stop

INFO │ ════════════════════════════════════════════════════════════════════════════════
INFO │ 📄 PROCESSING PDF: sample1.pdf
INFO │ ════════════════════════════════════════════════════════════════════════════════

INFO │ ✓ Loaded ground truth: sample1_gt.csv (3 rows)

INFO │ [EXTRACT] Processing sample1.pdf...
INFO │ ✓ Gemini: 45 properties extracted
INFO │ ✓ GPT: 42 properties extracted

INFO │ [EVALUATE]
INFO │   [EVAL] Scoring Gemini extraction...
INFO │     ✓ Gemini F1 Score: 0.8542
INFO │       Precision: 0.8750 | Recall: 0.8333
INFO │       TP: 25 | FP: 4 | FN: 5

INFO │   [EVAL] Scoring GPT extraction...
INFO │     ✓ GPT F1 Score: 0.7895
INFO │       Precision: 0.7600 | Recall: 0.8000
INFO │       TP: 20 | FP: 6 | FN: 5

INFO │ [WINNER] 🏆 Gemini wins! (F1: 0.8542 vs GPT: 0.7895)

INFO │ [SAVE]
INFO │   ✓ Saved 25 TP rows to results_finalized.csv

INFO │ ✓ PDF processing complete!

INFO │ [IDLE] No new PDFs found. Checking again in 5s...
```

### Step 4: Check Results

Results are saved to `pdf_monitor/results_db/results_finalized.csv`:

```csv
property_name,value,unit,matched_gt_property,matched_gt_value,matched_gt_unit,eval_result,match_score,pdf_filename,gt_filename,winning_llm,processing_timestamp,f1_score,precision,recall
Tensile Strength,250,MPa,Tensile Strength,250,MPa,TP,0.99,sample1.pdf,ground_truth,Gemini,2026-06-16T10:45:32.123456,0.8542,0.875,0.8333
Young's Modulus,210,GPa,Young's Modulus,210,GPa,TP,0.98,sample1.pdf,ground_truth,Gemini,2026-06-16T10:45:32.123456,0.8542,0.875,0.8333
...
```

---

## Configuration

Edit these variables in `pdf_monitor_pipeline.py`:

```python
# Folder paths
PDF_MONITOR_ROOT = Path("./pdf_monitor")
PDF_FOLDER = PDF_MONITOR_ROOT / "pdfs"
GT_FOLDER = PDF_MONITOR_ROOT / "ground_truth"
DB_FOLDER = PDF_MONITOR_ROOT / "results_db"

# Results file
RESULTS_FINALIZED_CSV = DB_FOLDER / "results_finalized.csv"

# Poll interval (check for new PDFs every N seconds)
POLL_INTERVAL = 5  # Change to 30 for slower polling, 1 for faster
```

**Naming Convention (NOT configurable):**
- PDF: `sample1.pdf` → GT lookup: `sample1_gt.csv`, `sample1_gt.xlsx`, `sample1_gt.xls`
- Fallback: If not found, tries `ground_truth.csv`, `ground_truth.xlsx`, `ground_truth.xls`

---

## How It Works

### Processing Flow

1. **Initialization**
   - Creates folder structure
   - Loads previously processed file list from `processing_log.json`

2. **Monitoring Loop**
   - Scans `pdfs/` folder for new `.pdf` files
   - Skips files already in `processing_log.json`
   - Processes one PDF at a time (sequential)

3. **Ground Truth Loading (Naming Convention)**
   ```
   PDF: sample1.pdf
   ↓
   Try: sample1_gt.csv → sample1_gt.xlsx → sample1_gt.xls
   ↓
   If not found, try fallback: ground_truth.csv → ground_truth.xlsx → ground_truth.xls
   ↓
   If neither exists: skip PDF with warning
   ```
   ```
   For each LLM output:
   ├── Detect column names (property, value, unit)
   ├── Call _ev_score() with ground truth
   ├── Get bipartite matching TP/FP/FN
   └── Calculate Precision, Recall, F1
   ```

5. **Winner Selection**
   - Compare F1 scores
   - Pick LLM with higher F1
   - If tie: Gemini wins by default

6. **Finalization**
   - Extract TP rows only from winner
   - Add metadata (LLM name, F1 score, timestamps)
   - Append to `results_finalized.csv`
   - Mark PDF as processed

7. **Loop Back**
   - If new PDFs exist: process next
   - If no PDFs: idle/wait for new ones

---

## Monitoring & State

### Processing Log (`processing_log.json`)

Tracks which PDFs have been processed:

```json
{
  "timestamp": "2026-06-16T10:45:32.123456",
  "processed_files": [
    "sample1.pdf",
    "sample2.pdf"
  ]
}
```

**Note:** Delete this file to re-process all PDFs.

---

## Troubleshooting

### Issue: "No ground truth found for sample1.pdf"
**Solution:**
- Create a ground truth file matching the PDF name: `sample1_gt.csv` (or `.xlsx`)
- Place it in `pdf_monitor/ground_truth/`
- Naming must be exact: `{pdf_name}_gt.{ext}`
- Fallback: Create `ground_truth.csv` to use for all PDFs

### Issue: "Ground truth file not found"
**Solution:**
- Check naming convention: `sample1.pdf` → `sample1_gt.csv` ✓
- Supported formats: `.csv`, `.xlsx`, `.xls`
- Verify file is in `pdf_monitor/ground_truth/` folder
- Check for typos in filename

### Issue: "Both extractions returned 0 rows"
**Solution:**
- PDF might be image-only or corrupted
- Check if Gemini/GPT APIs are available
- Check ChromaDB initialization

### Issue: "Could not detect all required columns"
**Solution:**
- Column names in GT must match hints in code:
  - Name: `property_name`, `property`, `prop_name`, `name`
  - Value: `value`, `val`, `measured_value`, `result`
  - Unit: `unit`, `units`, `si_unit`, `uom`
- Edit `_find_col()` hints if using custom column names

### Issue: Pipeline stops / No output
**Solution:**
- Check API keys are set (GEMINI_API_KEY, OPENAI_API_KEY in `.env`)
- Verify network connection
- Check logs for detailed error messages

---

## Advanced Usage

### Re-process All PDFs

```powershell
# Delete processing log to reset (pipeline will re-process all PDFs)
rm pdf_monitor\results_db\processing_log.json

# Restart pipeline
python pdf_monitor_pipeline.py
```

### Add New PDFs While Running

- Simply drop new PDFs into `pdf_monitor/pdfs/`
- Pipeline will automatically detect and process them
- Existing PDFs in `processing_log.json` are skipped

---

## Performance Notes

- **Extraction time:** ~5-15 minutes per PDF (depends on size)
- **Gemini + GPT run in parallel:** Total time = max(gemini_time, gpt_time)
- **Rate limiting:** 4s delay between batches (Gemini), respects API quotas
- **Polling:** Every 5s checks for new PDFs (configurable)

---

## Integration with Existing Code

- ✅ **Imports from existing modules:**
  - `run_pipeline()` from `DocToDB_eval_v2.py` (dual LLM extraction)
  - `_ev_score()` from `DocToDB_eval_v2.py` (evaluation)
  - `_find_col()` from `DocToDB_eval_v2.py` (column detection)

- ✅ **Does NOT modify:**
  - `crawler_graph.py` (untouched)
  - `Extraction.py` (untouched)
  - `Database.py` (untouched)

- ✅ **Fully standalone:**
  - Can run independently from crawler
  - No interference with existing pipelines
  - Can be toggled on/off anytime

---

## Next Steps

1. **Create folder structure** (already done in setup)
2. **Create ground truth file** → `pdf_monitor/ground_truth/sample1_gt.csv`
3. **Add test PDF** → `pdf_monitor/pdfs/sample1.pdf`
4. **Run pipeline** → `python pdf_monitor_pipeline.py`
5. **Check results** → `pdf_monitor/results_db/results_finalized.csv`

**Key:** PDF filename and GT filename must match!
- ✅ `sample1.pdf` + `sample1_gt.csv` → Will work
- ❌ `sample1.pdf` + `different_gt.csv` → Will be skipped (unless using fallback `ground_truth.csv`)

---

**Questions?** Check the log output for detailed error messages. All operations are logged to console with timestamps.
