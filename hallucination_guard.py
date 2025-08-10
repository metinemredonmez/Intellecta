# hallucination_guard.py
import json
import re
from typing import List, Dict, Tuple

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# ---- Utilities ----

def build_vstore_from_text(text: str) -> FAISS:
    """Create a FAISS store from a single transcript string."""
    docs = [Document(page_content=text, metadata={"source": "meeting"})]
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def bullets_from_text(text: str) -> List[str]:
    """Extract bullet-like lines from an LLM output."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("-", "*", "•")):
            line = line.lstrip("-*• ").strip()
        lines.append(line)
    # remove trivial single words
    return [l for l in lines if len(l.split()) > 2]

def sentences_from_paragraph(text: str) -> List[str]:
    """Naive sentence splitter for QA answers."""
    # Split by ., ?, ! but keep content
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip().split()) > 2]

def topk_support(vstore: FAISS, claim: str, k: int = 2) -> List[Tuple[str, float]]:
    """Return top-k (passage, score) for a claim using similarity_search_with_score.
    Note: FAISS returns distance; LangChain normalizes to similarity scores in [0,1] for cosine/IP backends.
    """
    results = vstore.similarity_search_with_score(claim, k=k)
    # results: List[(Document, score)] where higher means closer (implementation detail may vary).
    support = []
    for doc, score in results:
        # Normalize to [0,1] conservative (cap if library returns raw distances).
        # If score > 1 we cap to 1; if negative we floor to 0.
        sim = max(0.0, min(1.0, float(score)))  # best-effort normalization
        support.append((doc.page_content, sim))
    return support

def safe_json_parse(txt: str) -> Dict:
    try:
        return json.loads(txt)
    except Exception:
        # try to find JSON block
        m = re.search(r'\{.*\}', txt, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

# ---- LLM fact-check ----

FACT_CHECK_SYSTEM = (
    "You are a strict fact-checker. Given a single CLAIM and supporting CONTEXT "
    "passages from a meeting transcript, decide if the claim is grounded in the context. "
    "Respond ONLY as compact JSON with keys: "
    '{"grounded": true|false, "confidence": float (0..1), "reason": "short reason"}'
)

def llm_fact_check(llm: ChatOpenAI, claim: str, context_passages: List[str]) -> Dict:
    joined = "\n\n---\n\n".join(context_passages) if context_passages else ""
    user = f"CLAIM:\n{claim}\n\nCONTEXT:\n{joined}"
    resp = llm.invoke(
        [
            {"role": "system", "content": FACT_CHECK_SYSTEM},
            {"role": "user", "content": user},
        ]
    )
    data = safe_json_parse(resp.content or "{}")
    # fallback if model didn't follow schema
    grounded = bool(data.get("grounded", False))
    conf = float(data.get("confidence", 0.0))
    reason = str(data.get("reason", "")).strip() or "n/a"
    return {"grounded": grounded, "confidence": max(0.0, min(1.0, conf)), "reason": reason}

# ---- Main checker ----

def check_claims(
    transcript: str,
    claims: List[str],
    similarity_threshold: float = 0.35,
    min_overall_confidence: float = 0.5,
) -> List[Dict]:
    """Return per-claim evaluation dicts:
    {
      'claim': str,
      'support_passages': [str, ...],
      'similarity_scores': [float, ...],
      'similarity_ok': bool,
      'llm_grounded': bool,
      'llm_conf': float,
      'overall_confidence': float,
      'reason': str
    }
    """
    if not transcript or not claims:
        return []

    vstore = build_vstore_from_text(transcript)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    results = []
    for claim in claims:
        # 1) Similarity-based grounding
        sup = topk_support(vstore, claim, k=2)
        passages = [s for s, _ in sup]
        sims = [sc for _, sc in sup]
        sim_ok = any(sc >= similarity_threshold for sc in sims)

        # 2) LLM fact-check
        fc = llm_fact_check(llm, claim, passages[:2])
        grounded = fc["grounded"]
        llm_conf = fc["confidence"]

        # 3) Aggregate confidence (conservative)
        overall = 0.6 * (max(sims) if sims else 0.0) + 0.4 * llm_conf

        results.append(
            {
                "claim": claim,
                "support_passages": passages,
                "similarity_scores": sims,
                "similarity_ok": bool(sim_ok),
                "llm_grounded": bool(grounded),
                "llm_conf": float(llm_conf),
                "overall_confidence": float(overall),
                "reason": fc["reason"],
                "flagged": bool((not sim_ok) or (not grounded) or (overall < min_overall_confidence)),
            }
        )
    return results

def extract_claims_for_guard(summary_text: str, action_text: str, qa_text: str) -> Dict[str, List[str]]:
    """Collect claims from different outputs."""
    summary_claims = bullets_from_text(summary_text)
    action_claims = bullets_from_text(action_text)
    qa_claims = sentences_from_paragraph(qa_text)
    return {
        "summary": summary_claims,
        "actions": action_claims,
        "qa": qa_claims,
    }
