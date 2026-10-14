"""Group raw GitHub label names into rough "families".

Used only for reporting (to show how differently repos name the same idea).
The model itself always learns each repo's raw label names.
"""

import re

FAMILY_KEYWORDS = {
    "bug": ["bug", "defect", "crash", "regression", "broken"],
    # "feature request", not bare "request" (that would also catch "pull request").
    "feature": ["feature", "enhancement", "improvement", "proposal", "feature request"],
    # No bare "help": "help wanted" means "needs a contributor", not a question.
    "question": ["question", "support", "usage", "howto", "how to"],
    "docs": ["doc", "docs", "documentation"],
    "duplicate": ["duplicate", "dupe"],
    "wontfix": ["wontfix", "won't fix", "invalid", "not planned"],
}


def normalize(label: str) -> str:
    """Lowercase, drop emoji/punctuation, collapse separators."""
    text = label.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def family(label: str) -> str:
    text = normalize(label)
    tokens = set(text.split())
    for fam, words in FAMILY_KEYWORDS.items():
        for word in words:
            if " " in word:
                if word in text:
                    return fam
            elif word in tokens:
                return fam
    return "other"
