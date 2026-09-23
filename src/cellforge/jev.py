"""cellforge — JEV oracle helper.

The Joint Embedding Validator (JEV) is used to gate canon-worthy promotion
of design decisions. cellforge's core inversion should pass JEV evaluation
before being committed as canonical design.

Usage:
    from cellforge.jev import jev_oracle
    result = jev_oracle("cellforge dispatcher cell IS the playhead",
                        questions={'canon_score': '...'})
"""
import json
import os
import urllib.request

JEV_URL = "https://api.typesafe.ai/v1/systemone"


def jev_oracle(state: str, questions: dict, model: str = "jev-latest") -> dict:
    """Query JEV with a state and questions, return parsed answer dict.

    Example:
        result = jev_oracle(
            "cellforge is a cellular-substrate ML training system. "
            "The dispatcher cell IS the playhead.",
            questions={
                'canon_score': {
                    'type': 'score',
                    'instructions': 'How canon-worthy is this design?',
                    'criteria': ['Spam', 'Derivative', 'Solid canon', 'Brilliant paradigm']
                },
                'is_canon': {
                    'type': 'noul',
                    'instructions': 'Is this canon?'
                }
            }
        )
    """
    key = os.environ.get("TYPESAFEAI_KEY", "")
    if not key:
        return {"error": "TYPESAFEAI_KEY not set", "answers": {}}
    body = json.dumps({
        "model": model,
        "state": state,
        "questions": questions,
    }).encode()
    try:
        req = urllib.request.Request(
            JEV_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "answers": {}}


def canon_gate(state: str, threshold: float = 0.7) -> dict:
    """Standard canon gate: query JEV for is_canon + canon_score. Promote if both pass.

    Returns: {'gate_passed': bool, 'score': float, 'is_canon': float, 'details': dict}
    """
    result = jev_oracle(state, questions={
        "canon_score": {
            "type": "score",
            "instructions": "How canon-worthy is this?",
            "criteria": ["Spam", "Derivative", "Solid canon", "Brilliant paradigm"],
        },
        "is_canon": {
            "type": "noul",
            "instructions": "Is this canon?",
        },
    })
    answers = result.get("answers", {})
    canon_score = answers.get("canon_score", {})
    is_canon = answers.get("is_canon", {})
    score_value = canon_score.get("score", 0) / 3.0  # normalize to [0,1]
    is_canon_value = is_canon.get("noul", 0)
    return {
        "gate_passed": score_value >= threshold and is_canon_value >= threshold,
        "canon_score": score_value,
        "is_canon": is_canon_value,
        "details": result,
    }
