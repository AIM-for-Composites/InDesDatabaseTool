"""relevance.py
Two-stage relevance filter:
  Stage 1: keyword fast-path (free, instant)
  Stage 2: Gemini (free tier available)
"""
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import List, Dict

client = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY", "AIzaSyBzyMFKEqcjsWpR-OGAY42T250o1O39v3Y")
)
STRONG_INCLUDE = [
    "composite", "fiber reinforced", "fibre reinforced",
    "carbon fiber", "carbon fibre", "glass fiber", "glass fibre",
    "epoxy", "polymer matrix", "tensile strength", "flexural strength",
    "young's modulus", "interlaminar", "laminate", "nanocomposite",
    "aramid", "basalt fiber", "hybrid composite", "woven composite",
]

STRONG_EXCLUDE = [
    "dental composite", "geological", "music", "economics",
    "politics", "social science", "history", "literature",
    "cryptography", "blockchain", "finance",
]

PROMPT = """You are a materials science expert.
Decide if this paper is relevant to:
- Composite materials (fiber-reinforced, particle-reinforced, hybrid)
- Fibers (carbon, glass, natural, aramid) used in structural applications
- Polymers used as matrix materials (epoxy, polyester, thermoplastics)

Papers must report measurable material properties or processing methods.

Reply ONLY with valid JSON — no extra text:
{"relevant": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}"""


def filter_papers(papers: List[Dict]) -> List[Dict]:
    """Run relevance check on all papers. Returns papers with is_relevant set."""
    results = []
    llm_count = 0

    for paper in papers:
        text = f"{paper.get('title','')} {paper.get('abstract','')}".lower()

        # Stage 1a: fast exclude
        if any(kw in text for kw in STRONG_EXCLUDE):
            paper["is_relevant"] = 0
            paper["_stage"] = "fast_exclude"
            results.append(paper)
            continue

        # Stage 1b: fast include (2+ strong keywords)
        hits = [kw for kw in STRONG_INCLUDE if kw in text]
        if len(hits) >= 2:
            paper["is_relevant"] = 1
            paper["_stage"] = "fast_include"
            results.append(paper)
            continue

        # Stage 2: uncertain — ask Claude
        paper = _claude_check(paper)
        paper["_stage"] = "claude"
        llm_count += 1
        results.append(paper)

    relevant = sum(1 for p in results if p["is_relevant"])
    print(f"  [Relevance] {relevant}/{len(results)} relevant | Claude calls: {llm_count}")
    return results


def _claude_check(paper: Dict) -> Dict:
    text = f"Title: {paper.get('title','')}\nAbstract: {paper.get('abstract','')[:600]}"
    try:
        msg = client.invoke([HumanMessage(content=f"{PROMPT}\n\n{text}")])
        result = json.loads(msg.content.strip())
        paper["is_relevant"] = 1 if result.get("relevant") else 0
        paper["_reason"] = result.get("reason", "")
    except Exception as e:
        print(f"  [Relevance] Gemini check failed: {e} — keeping paper")
        paper["is_relevant"] = 1  # keep on failure; validate agent will catch it
    return paper