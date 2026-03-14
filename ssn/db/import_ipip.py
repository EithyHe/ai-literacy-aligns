"""Import IPIP data (items_master_ipip.csv) into the unified hierarchy DB.

Maps 36 instruments to frameworks, domains, and constructs using
predefined hierarchy mappings based on the IPIP documentation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from ssn.config import IPIP_MASTER_CSV
from ssn.db.schema import (
    bulk_upsert_items,
    init_db,
    insert_cross_framework_mapping,
    link_domain_construct,
    upsert_construct,
    upsert_domain,
    upsert_framework,
    get_connection,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hierarchy mapping tables
# ---------------------------------------------------------------------------

FRAMEWORK_DEFS: list[dict] = [
    {"framework_id": "IPIP-NEO", "name": "IPIP-NEO", "version": "Johnson 2014",
     "source_url": "https://ipip.ori.org/newNEOKey.htm", "license": "Public Domain",
     "citation": "Johnson, J.A. (2014). Measuring thirty facets of the FFM with a 120-item public domain inventory."},
    {"framework_id": "HEXACO", "name": "HEXACO-PI (IPIP)", "version": "IPIP implementation",
     "source_url": "https://ipip.ori.org/newHEXACOKey.htm", "license": "Public Domain", "citation": ""},
    {"framework_id": "AB5C", "name": "Abridged Big Five Circumplex", "version": "IPIP",
     "source_url": "https://ipip.ori.org/newAB5CKey.htm", "license": "Public Domain", "citation": ""},
    {"framework_id": "VIA", "name": "VIA Character Strengths (IPIP)", "version": "IPIP implementation",
     "source_url": "https://ipip.ori.org/newVIAKey.htm", "license": "Public Domain", "citation": "Peterson & Seligman (2004)"},
    {"framework_id": "BFAS", "name": "Big Five Aspect Scales", "version": "DeYoung et al. 2007",
     "source_url": "https://ipip.ori.org/newBFASKey.htm", "license": "Public Domain", "citation": ""},
    {"framework_id": "16PF", "name": "16 Personality Factors (IPIP)", "version": "IPIP",
     "source_url": "https://ipip.ori.org/new16PFKey.htm", "license": "Public Domain", "citation": ""},
    {"framework_id": "6FPQ", "name": "Six Factor Personality Questionnaire (IPIP)", "version": "IPIP",
     "source_url": "https://ipip.ori.org/new6FPQKey.htm", "license": "Public Domain", "citation": ""},
    {"framework_id": "TCI", "name": "Temperament and Character Inventory (IPIP)", "version": "IPIP",
     "source_url": "https://ipip.ori.org/newTCIKey.htm", "license": "Public Domain", "citation": ""},
    {"framework_id": "CPI", "name": "California Psychological Inventory (IPIP)", "version": "IPIP",
     "source_url": "https://ipip.ori.org/", "license": "Public Domain", "citation": ""},
    {"framework_id": "MPQ", "name": "Multidimensional Personality Questionnaire (IPIP)", "version": "IPIP",
     "source_url": "https://ipip.ori.org/", "license": "Public Domain", "citation": ""},
    {"framework_id": "JPI", "name": "Jackson Personality Inventory (IPIP)", "version": "IPIP",
     "source_url": "https://ipip.ori.org/", "license": "Public Domain", "citation": ""},
    {"framework_id": "HPI", "name": "Hogan Personality Inventory (IPIP)", "version": "IPIP",
     "source_url": "https://ipip.ori.org/", "license": "Public Domain", "citation": ""},
    {"framework_id": "CAT-PD", "name": "CAT-PD (IPIP)", "version": "IPIP",
     "source_url": "https://ipip.ori.org/", "license": "Public Domain", "citation": ""},
    {"framework_id": "ORVIS", "name": "Oregon Vocational Interest Scales", "version": "IPIP",
     "source_url": "https://ipip.ori.org/newORVISKey.htm", "license": "Public Domain", "citation": ""},
    {"framework_id": "ORAIS", "name": "Oregon Avocational Interest Scales", "version": "IPIP",
     "source_url": "https://ipip.ori.org/", "license": "Public Domain", "citation": ""},
    {"framework_id": "IPIP-IPC", "name": "IPIP Interpersonal Circumplex", "version": "IPIP",
     "source_url": "https://ipip.ori.org/", "license": "Public Domain", "citation": ""},
]

# Instrument -> framework_id mapping
INSTRUMENT_TO_FRAMEWORK: dict[str, str] = {
    "NEO": "IPIP-NEO", "NEO5-20": "IPIP-NEO",
    "HEXACO_PI": "HEXACO",
    "AB5C": "AB5C",
    "VIA": "VIA",
    "BFAS": "BFAS", "BFAS-20": "BFAS",
    "16PF": "16PF",
    "6FPQ": "6FPQ",
    "7FACTOR": "IPIP-NEO",
    "TCI": "TCI",
    "CPI": "CPI",
    "MPQ": "MPQ",
    "JPI": "JPI",
    "HPI": "HPI", "HPI-HIC": "HPI",
    "CAT-PD": "CAT-PD",
    "ORVIS": "ORVIS",
    "ORAIS": "ORAIS",
    "IPIP-IPC": "IPIP-IPC",
}

# NEO Big Five domain mapping (facet -> domain)
NEO_DOMAIN_MAP: dict[str, str] = {
    "Anxiety": "Neuroticism", "Anger": "Neuroticism", "Depression": "Neuroticism",
    "Self-Consciousness": "Neuroticism", "Immoderation": "Neuroticism",
    "Vulnerability": "Neuroticism",
    "Friendliness": "Extraversion", "Gregariousness": "Extraversion",
    "Assertiveness": "Extraversion", "Activity Level": "Extraversion",
    "Excitement-Seeking": "Extraversion", "Cheerfulness": "Extraversion",
    "Imagination": "Openness", "Artistic Interests": "Openness",
    "Emotionality": "Openness", "Adventurousness": "Openness",
    "Intellect": "Openness", "Liberalism": "Openness",
    "Trust": "Agreeableness", "Morality": "Agreeableness",
    "Altruism": "Agreeableness", "Cooperation": "Agreeableness",
    "Modesty": "Agreeableness", "Sympathy": "Agreeableness",
    "Self-Efficacy": "Conscientiousness", "Orderliness": "Conscientiousness",
    "Dutifulness": "Conscientiousness", "Achievement-Striving": "Conscientiousness",
    "Self-Discipline": "Conscientiousness", "Cautiousness": "Conscientiousness",
    "Achievement-striving": "Conscientiousness",
}

HEXACO_DOMAIN_MAP: dict[str, str] = {
    "Sincerity": "Honesty-Humility", "Fairness": "Honesty-Humility",
    "Greed Avoidance": "Honesty-Humility", "Greed-Avoidance": "Honesty-Humility",
    "Modesty": "Honesty-Humility",
    "Fearfulness": "Emotionality", "Anxiety": "Emotionality",
    "Dependence": "Emotionality", "Sentimentality": "Emotionality",
    "Social Self-Esteem": "eXtraversion", "Social Boldness": "eXtraversion",
    "Sociability": "eXtraversion", "Liveliness": "eXtraversion",
    "Social self-esteem": "eXtraversion", "Social boldness": "eXtraversion",
    "Forgiveness": "Agreeableness", "Gentleness": "Agreeableness",
    "Flexibility": "Agreeableness", "Patience": "Agreeableness",
    "Organization": "Conscientiousness", "Diligence": "Conscientiousness",
    "Perfectionism": "Conscientiousness", "Prudence": "Conscientiousness",
    "Aesthetic Appreciation": "Openness", "Inquisitiveness": "Openness",
    "Creativity": "Openness", "Unconventionality": "Openness",
    "Aesthetic appreciation": "Openness",
    "Altruism": "Interstitial",
}

VIA_DOMAIN_MAP: dict[str, str] = {
    "Creativity": "Wisdom", "Curiosity": "Wisdom", "Judgment": "Wisdom",
    "Love of Learning": "Wisdom", "Perspective": "Wisdom",
    "Bravery": "Courage", "Perseverance": "Courage", "Honesty": "Courage", "Zest": "Courage",
    "Love": "Humanity", "Kindness": "Humanity", "Social Intelligence": "Humanity",
    "Teamwork": "Justice", "Fairness": "Justice", "Leadership": "Justice",
    "Forgiveness": "Temperance", "Humility": "Temperance",
    "Prudence": "Temperance", "Self-Regulation": "Temperance",
    "Appreciation of Beauty": "Transcendence", "Gratitude": "Transcendence",
    "Hope": "Transcendence", "Humor": "Transcendence", "Spirituality": "Transcendence",
    "Self-regulation": "Temperance", "Social intelligence": "Humanity",
    "Love of learning": "Wisdom", "Appreciation of beauty": "Transcendence",
}

BFAS_DOMAIN_MAP: dict[str, str] = {
    "Volatility": "Neuroticism", "Withdrawal": "Neuroticism",
    "Enthusiasm": "Extraversion", "Assertiveness": "Extraversion",
    "Openness": "Openness/Intellect", "Intellect": "Openness/Intellect",
    "Compassion": "Agreeableness", "Politeness": "Agreeableness",
    "Industriousness": "Conscientiousness", "Orderliness": "Conscientiousness",
}


def _slug(s: str) -> str:
    """Create a URL-safe slug from a string."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _build_domain_id(framework_id: str, domain_name: str) -> str:
    return f"{_slug(framework_id)}__{_slug(domain_name)}"


def _build_construct_id(framework_id: str, construct_name: str) -> str:
    return f"{_slug(framework_id)}__{_slug(construct_name)}"


def _get_domain_for_construct(framework_id: str, construct_name: str) -> str | None:
    """Look up the domain for a given construct based on predefined mappings."""
    maps = {
        "IPIP-NEO": NEO_DOMAIN_MAP,
        "HEXACO": HEXACO_DOMAIN_MAP,
        "VIA": VIA_DOMAIN_MAP,
        "BFAS": BFAS_DOMAIN_MAP,
    }
    domain_map = maps.get(framework_id, {})
    return domain_map.get(construct_name) or domain_map.get(construct_name.title())


def _original_label_for(framework_id: str) -> tuple[str, str]:
    """Return (domain_label, construct_label) based on framework convention."""
    labels = {
        "VIA": ("Virtue", "Character Strength"),
        "BFAS": ("Factor", "Aspect"),
        "16PF": ("Global Factor", "Primary Factor"),
    }
    return labels.get(framework_id, ("Domain", "Facet"))


def import_ipip(csv_path: str | Path | None = None, reset: bool = False) -> dict:
    """Import IPIP data into the SQLite database.

    Returns a dict with import statistics.
    """
    csv_path = Path(csv_path) if csv_path else IPIP_MASTER_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"IPIP master file not found: {csv_path}")

    if reset:
        from ssn.db.schema import reset_db
        reset_db()
    else:
        init_db()

    df = pd.read_csv(csv_path, encoding="utf-8")
    logger.info("Loaded %d items from %s", len(df), csv_path)

    # Ensure required columns
    for col in ("item_id", "instrument", "text", "construct_reported"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Step 1: Register frameworks
    registered_fw = set()
    for fw_def in FRAMEWORK_DEFS:
        upsert_framework(fw_def)
        registered_fw.add(fw_def["framework_id"])

    # For instruments without explicit framework definitions, create generic ones
    instruments = df["instrument"].unique()
    for inst in instruments:
        fw_id = INSTRUMENT_TO_FRAMEWORK.get(inst, inst)
        if fw_id not in registered_fw:
            upsert_framework({
                "framework_id": fw_id,
                "name": f"{inst} (IPIP)",
                "version": "IPIP",
                "source_url": "https://ipip.ori.org/",
                "license": "Public Domain",
                "citation": "",
            })
            registered_fw.add(fw_id)

    # Step 2: Register all constructs with framework_id; create domains + domain_construct_map
    # only for the 4 frameworks with real theoretical domains (IPIP-NEO, HEXACO, VIA, BFAS).
    domain_set: set[str] = set()
    construct_set: set[str] = set()

    for inst in instruments:
        fw_id = INSTRUMENT_TO_FRAMEWORK.get(inst, inst)
        domain_label, construct_label = _original_label_for(fw_id)
        inst_df = df[df["instrument"] == inst]
        constructs = inst_df["construct_reported"].dropna().unique()

        for c_name in constructs:
            c_id = _build_construct_id(fw_id, c_name)
            n_items = len(inst_df[inst_df["construct_reported"] == c_name])
            if c_id not in construct_set:
                upsert_construct({
                    "construct_id": c_id,
                    "name": c_name,
                    "original_label": construct_label,
                    "description": None,
                    "scale_name": f"{inst} {c_name}",
                    "item_count": n_items,
                    "scale_source_url": None,
                    "framework_id": fw_id,
                })
                construct_set.add(c_id)

            domain_name = _get_domain_for_construct(fw_id, c_name)
            if domain_name:
                d_id = _build_domain_id(fw_id, domain_name)
                if d_id not in domain_set:
                    upsert_domain({
                        "domain_id": d_id,
                        "framework_id": fw_id,
                        "name": domain_name,
                        "original_label": domain_label,
                        "description": None,
                    })
                    domain_set.add(d_id)
                link_domain_construct(d_id, c_id, "primary")

    # Step 3: Insert items
    item_records: list[dict] = []
    for _, row in df.iterrows():
        inst = row["instrument"]
        fw_id = INSTRUMENT_TO_FRAMEWORK.get(inst, inst)
        c_name = row.get("construct_reported", "")
        c_id = _build_construct_id(fw_id, c_name) if pd.notna(c_name) and c_name else None

        direction = "-" if row.get("is_reverse_keyed") is True or str(row.get("key", "1")).strip() == "-1" else "+"
        text = str(row.get("text", "")).strip()
        text_norm = str(row.get("text_norm", "")).strip() if pd.notna(row.get("text_norm")) else text.lower()

        raw_alpha = row.get("alpha")
        try:
            alpha_val = float(raw_alpha) if pd.notna(raw_alpha) else None
        except (ValueError, TypeError):
            alpha_val = None

        item_records.append({
            "item_id": row["item_id"],
            "construct_id": c_id,
            "text": text,
            "text_norm": text_norm if text_norm else text.lower(),
            "direction": direction,
            "alpha": alpha_val,
            "instrument": inst,
            "framework_id": fw_id,
        })

    bulk_upsert_items(item_records)

    # Step 4: Seed cross-framework domain alignments (PRD FR-DS-3)
    _seed_cross_framework_map(domain_set)

    stats = {
        "frameworks": len(registered_fw),
        "domains": len(domain_set),
        "constructs": len(construct_set),
        "items": len(item_records),
    }
    logger.info("Import complete: %s", stats)
    return stats


def _seed_cross_framework_map(domain_set: set[str]) -> None:
    """Insert known cross-framework domain alignments (NEO–HEXACO, NEO–BFAS)."""
    # Only insert if the corresponding domains exist (e.g. after full IPIP import)
    def _d(fw: str, name: str) -> str:
        return f"{_slug(fw)}__{_slug(name)}"

    mappings: list[tuple[str, str, str, str, str, str]] = [
        # NEO <-> HEXACO (Big Five / Six Factor overlap)
        ("Extraversion", "IPIP-NEO", _d("IPIP-NEO", "Extraversion"), "HEXACO", _d("HEXACO", "eXtraversion"), "equivalent"),
        ("Openness", "IPIP-NEO", _d("IPIP-NEO", "Openness"), "HEXACO", _d("HEXACO", "Openness"), "equivalent"),
        ("Agreeableness", "IPIP-NEO", _d("IPIP-NEO", "Agreeableness"), "HEXACO", _d("HEXACO", "Agreeableness"), "equivalent"),
        ("Conscientiousness", "IPIP-NEO", _d("IPIP-NEO", "Conscientiousness"), "HEXACO", _d("HEXACO", "Conscientiousness"), "equivalent"),
        ("Neuroticism/Emotionality", "IPIP-NEO", _d("IPIP-NEO", "Neuroticism"), "HEXACO", _d("HEXACO", "Emotionality"), "approximate"),
        # NEO <-> BFAS (Big Five Aspect Scales)
        ("Neuroticism", "IPIP-NEO", _d("IPIP-NEO", "Neuroticism"), "BFAS", _d("BFAS", "Neuroticism"), "equivalent"),
        ("Extraversion", "IPIP-NEO", _d("IPIP-NEO", "Extraversion"), "BFAS", _d("BFAS", "Extraversion"), "equivalent"),
        ("Openness/Intellect", "IPIP-NEO", _d("IPIP-NEO", "Openness"), "BFAS", _d("BFAS", "Openness/Intellect"), "equivalent"),
        ("Agreeableness", "IPIP-NEO", _d("IPIP-NEO", "Agreeableness"), "BFAS", _d("BFAS", "Agreeableness"), "equivalent"),
        ("Conscientiousness", "IPIP-NEO", _d("IPIP-NEO", "Conscientiousness"), "BFAS", _d("BFAS", "Conscientiousness"), "equivalent"),
    ]
    for theme, fw_a, d_a, fw_b, d_b, mtype in mappings:
        if d_a in domain_set and d_b in domain_set:
            try:
                insert_cross_framework_mapping(theme, fw_a, d_a, fw_b, d_b, mtype)
            except Exception as e:
                logger.debug("Skip cross-framework mapping %s: %s", (theme, fw_a, fw_b), e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stats = import_ipip(reset=True)
    print(f"Import stats: {stats}")
