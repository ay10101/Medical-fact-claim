"""
Verdict and Explanation Engine Module
Determines scientific verification verdict, assesses health risk levels,
and formats rich educational explanations with:
  - Full article abstracts in blockquotes
  - Doctor/researcher credentials
  - Source credibility tier
  - Direct article links
"""

from typing import List, Dict, Any

# Dangerous medical advice indicators that elevate risk to Very High
DANGEROUS_PATTERNS = [
    "stop chemo", "skip chemo", "stop insulin", "no insulin", "avoid vaccine",
    "do not vaccinate", "don't vaccinate", "dawai band", "chemo mat lo",
    "insulin mat lo", "teeka mat lagao", "toxic", "bleach", "gomutra cure"
]

HIGH_SIMILARITY_THRESHOLD = 0.35
MODERATE_SIMILARITY_THRESHOLD = 0.30


def assess_verdict_and_risk(claim: str, retrieved_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates verdict, evidence status, and risk level based on retrieved evidence.
    """
    if not retrieved_evidence or len(retrieved_evidence) == 0:
        return {
            "verdict": "No relevant evidence",
            "evidence_status": "No matching clinical documents found",
            "risk": "Medium",
            "confidence": 0.0,
            "stance": "none"
        }

    top_evidence = retrieved_evidence[0]
    top_score = top_evidence.get("similarity_score", 0.0)
    top_stance = top_evidence.get("stance", "").strip().lower()
    top_topic = top_evidence.get("claim_topic", "").strip().lower()
    correlation_score = float(top_evidence.get("correlation_score", 0.0))

    # Conflict detection: same topic, opposing stances, close scores
    has_conflict = any(
        item.get("claim_topic", "").strip().lower() == top_topic
        and item.get("stance", "").strip().lower() != top_stance
        and (top_score - item.get("similarity_score", 0.0)) <= 0.12
        and item.get("similarity_score", 0.0) >= MODERATE_SIMILARITY_THRESHOLD
        for item in retrieved_evidence[1:]
    )

    # If the top retrieved article has no concept correlation with the claim and semantic score is marginal,
    # it is an irrelevant spurious retrieval (e.g. Chyawanprash for a cricket query).
    if correlation_score == 0.0 and top_score < 0.38:
        verdict = "No relevant evidence"
        evidence_status = "No relevant clinical evidence found in trusted repository for this condition"
        risk = "Medium"
    elif top_score < MODERATE_SIMILARITY_THRESHOLD:
        verdict = "No relevant evidence"
        evidence_status = "No relevant clinical evidence found in trusted repository"
        risk = "Medium"
    elif has_conflict:
        verdict = "Uncertain / Insufficient evidence"
        evidence_status = "Conflicting clinical evidence found among top sources"
        risk = "Medium"
    elif top_score >= HIGH_SIMILARITY_THRESHOLD:
        if top_stance == "refutes":
            verdict = "Refuted"
            evidence_status = "Directly refuted by reputable clinical evidence"
            claim_lower = claim.lower()
            is_dangerous = any(pat in claim_lower for pat in DANGEROUS_PATTERNS)
            risk = "Very High" if is_dangerous else "High"
        elif top_stance == "supports":
            verdict = "Supported"
            evidence_status = "Corroborated by verified medical sources"
            risk = "Low"
        else:
            verdict = "Uncertain / Insufficient evidence"
            evidence_status = "Evidence stance inconclusive"
            risk = "Medium"
    else:
        verdict = "Uncertain / Insufficient evidence"
        evidence_status = "Moderate similarity; insufficient clinical evidence"
        risk = "Medium"

    return {
        "verdict": verdict,
        "evidence_status": evidence_status,
        "risk": risk,
        "confidence": top_score,
        "stance": top_stance
    }


def _format_abstract_block(abstract: str, max_chars: int = 1000) -> str:
    """
    Formats a full article abstract into a clean quoted blockquote block.
    Truncates gracefully and always wraps in apostrophes as requested.
    """
    if not abstract or len(abstract.strip()) < 20:
        return ""
    text = abstract.strip()
    if len(text) > max_chars:
        # Cut at last full sentence within limit
        cut = text[:max_chars]
        last_period = max(cut.rfind(". "), cut.rfind(".\n"))
        if last_period > max_chars * 0.6:
            text = cut[:last_period + 1] + " [...]"
        else:
            text = cut + " [...]"
    # Wrap in apostrophes (as user requested)
    return f'"{text}"'


def generate_explanation(
    original_claim: str,
    detected_language: str,
    is_health: bool,
    verdict_data: Dict[str, Any],
    retrieved_evidence: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generates rich educational explanation with full article abstract in quotes,
    doctor/researcher credentials, source tier, and direct article links.
    """
    verdict = verdict_data["verdict"]
    risk = verdict_data["risk"]
    evidence_status = verdict_data["evidence_status"]

    if not is_health:
        summary_text = (
            "The claim was classified as **NOT health-related**. "
            "Our medical verification engine only processes health, medical, or clinical statements. "
            "No clinical retrieval was performed."
        )
        return {
            "original_claim": original_claim,
            "detected_language": detected_language,
            "is_health_related": False,
            "verdict": "Non-Health Claim",
            "evidence_status": "Not Applicable",
            "risk": "None",
            "summary_text": summary_text,
            "retrieved_evidence": [],
            "disclaimer": "This tool is for educational purposes only and does not constitute medical advice."
        }

    # Verdict-specific framing
    if verdict == "Supported":
        action_phrase = "**supports**"
        condition_header = "✅ **How and When this Claim is True (Clinical Context)**:"
    elif verdict == "Refuted":
        action_phrase = "**directly refutes**"
        condition_header = "❌ **Why this Claim is Refuted (Not Scientifically Valid)**:"
    else:
        action_phrase = "**does not conclusively substantiate**"
        condition_header = "⚠️ **Why this Claim is NOT PROVEN YET (Scientific Evidence Lacking)**:"

    top_evidence = retrieved_evidence[0] if retrieved_evidence else {}

    # Source metadata
    evaluators = top_evidence.get("authors", "Medical Fact-Checking Team")
    credentials = top_evidence.get("author_credentials", "Peer-Reviewed Medical Researchers")
    source_tier = top_evidence.get("source_tier", "Verified Medical Source")
    rag_source = top_evidence.get("source", "Clinical Evidence Base")
    journal = top_evidence.get("journal", "")
    article_link = top_evidence.get("url", "#")
    article_title = top_evidence.get("article_title", top_evidence.get("claim_topic", "Clinical Evidence Review"))
    citation = top_evidence.get("citation", "")
    pub_date = top_evidence.get("date", "")

    # Full abstract in apostrophes
    full_abstract = top_evidence.get("full_abstract", top_evidence.get("evidence_text", ""))
    abstract_block = _format_abstract_block(full_abstract)

    # Build rich explanation
    summary_lines = [
        f"### 📋 Scientific Verdict: **{verdict}** (Risk Level: `{risk}`)",
        f"",
        f"The retrieved clinical evidence {action_phrase} the claim based on investigations in **{rag_source}**.",
        f"",
        f"{condition_header}",
    ]

    if abstract_block:
        summary_lines += [
            f"> {abstract_block}",
            f">",
        ]

    summary_lines += [
        f"",
        f"---",
        f"",
        f"**🧑‍⚕️ Verified / Authored by**: **{evaluators}**",
        f"**📋 Author Credentials**: {credentials}",
        f"**{source_tier}**",
    ]

    if journal:
        summary_lines.append(f"**📰 Published In**: {journal}" + (f" ({pub_date})" if pub_date else ""))

    if citation:
        summary_lines.append(f"**📚 Full Citation**: *{citation}*")

    summary_lines += [
        f"**🔗 Source Article**: [{article_title}]({article_link})",
        f"**📊 RAG Semantic Similarity Score**: `{top_evidence.get('similarity_score', 'N/A')}`",
        f"",
        f"> ⚠️ *This verification is retrieved from peer-reviewed evidence and is for educational purposes only.*",
        f"> *Always consult a licensed medical professional for personal clinical advice.*",
    ]

    summary_text = "\n".join(summary_lines)

    return {
        "original_claim": original_claim,
        "detected_language": detected_language,
        "is_health_related": True,
        "verdict": verdict,
        "evidence_status": evidence_status,
        "risk": risk,
        "summary_text": summary_text,
        "retrieved_evidence": retrieved_evidence,
        "disclaimer": (
            "This tool is for educational purposes and is NOT medical advice. "
            "Consult a certified medical professional for diagnosis and treatment."
        )
    }
