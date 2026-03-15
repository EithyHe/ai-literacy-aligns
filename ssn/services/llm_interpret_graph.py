"""SSN-specific LLM interpretation for Leiden community (domain) naming.

Uses SSN DB and embedding service only. Builds representative items per
Leiden community from construct-level community mapping, then calls OpenAI
to assign short domain labels and rationales. For use in the pipeline that
reassigns domains via Leiden and persists them to the DB.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ssn.config import OPENAI_API_KEY
from ssn.db.schema import get_all_constructs, get_items_by_construct
from ssn.services.embedding_service import load_item_embeddings

logger = logging.getLogger(__name__)

# Optional: load .env from project root and ssn/.env
try:
    import dotenv

    _project_root = Path(__file__).resolve().parent.parent.parent
    _env_candidates = [
        _project_root / ".env",
        _project_root / "ssn" / ".env",
    ]
    for _env in _env_candidates:
        if _env.exists():
            dotenv.load_dotenv(_env, override=False)
except ImportError:
    pass


def _has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY)


def _format_prompt(cluster_name: str, df_cluster: pd.DataFrame, max_items: int = 15) -> str:
    """Format representative items for the LLM prompt."""
    lines = []
    for _, r in df_cluster.head(max_items).iterrows():
        txt = str(r.get("text") or "")
        cst = str(r.get("construct_reported") or r.get("construct_name") or "")
        centrality = float(r.get("centrality", 0))
        lines.append(f"- [Centrality:{centrality:.3f}] {cst} :: {txt[:160]}")
    return f"Community: {cluster_name}\n" + "\n".join(lines)


def _extract_cluster_number(cluster_name: Any) -> int:
    """Extract numeric part from cluster id for sorting (e.g. 0 from 'leiden_0' or 0 from 0)."""
    if isinstance(cluster_name, (int, float)):
        return int(cluster_name)
    if isinstance(cluster_name, bytes):
        cluster_name = cluster_name.decode("utf-8")
    s = str(cluster_name)
    numbers = re.findall(r"\d+", s)
    return int(numbers[0]) if numbers else 999999


def build_top_items_from_leiden_communities(
    communities: dict[str, int],
    top_k: int = 10,
) -> pd.DataFrame:
    """Build a representative-items DataFrame per Leiden community for LLM input.

    Uses SSN DB and item embeddings. For each community, items from constructs
    in that community get an intra-community centrality (mean cosine similarity
    to other items in the same community); top_k items per community are kept.

    Args:
        communities: Mapping construct_id -> community_id (int).
        top_k: Max number of representative items per community.

    Returns:
        DataFrame with columns: cluster, centrality, item_id, text, construct_reported.
        Compatible with run_ssn_leiden_domain_interpretation(..., top_items_df=...).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    construct_names = {c["construct_id"]: c.get("name", c["construct_id"]) for c in get_all_constructs()}
    embeddings, item_ids_ordered = load_item_embeddings()
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids_ordered)}

    # Build item_id -> (construct_id, community_id, text)
    item_info: list[tuple[str, str, int, str]] = []
    for construct_id, comm_id in communities.items():
        for it in get_items_by_construct(construct_id):
            iid = str(it.get("item_id", ""))
            if iid not in item_id_to_idx:
                continue
            text = it.get("text") or ""
            item_info.append((iid, construct_id, comm_id, text))

    if not item_info:
        logger.warning("No items found for any Leiden community")
        return pd.DataFrame(columns=["cluster", "centrality", "item_id", "text", "construct_reported"])

    # Per-community: item indices and centrality
    unique_communities = sorted(set(cid for (_, _, cid, _) in item_info), key=lambda x: (x,))
    rows: list[dict[str, Any]] = []

    for comm_id in unique_communities:
        comm_items = [(iid, cid, text) for (iid, cid, c, text) in item_info if c == comm_id]
        if not comm_items:
            continue
        iids = [x[0] for x in comm_items]
        indices = [item_id_to_idx[iid] for iid in iids]
        sub_emb = embeddings[indices]
        if sub_emb.shape[0] == 1:
            centralities = np.array([0.0])
        else:
            sim = cosine_similarity(sub_emb, sub_emb)
            np.fill_diagonal(sim, 0)
            n_other = sim.shape[0] - 1
            centralities = (sim.sum(axis=1) / n_other) if n_other > 0 else np.zeros(sim.shape[0])

        order = np.argsort(-centralities)[:top_k]
        for pos in order:
            iid = iids[pos]
            cid = comm_items[pos][1]
            text = comm_items[pos][2]
            rows.append({
                "cluster": comm_id,
                "centrality": float(centralities[pos]),
                "item_id": iid,
                "text": text,
                "construct_reported": construct_names.get(cid, cid),
            })

    df = pd.DataFrame(rows)
    logger.info(f"Built top items: {len(df)} rows, {len(unique_communities)} communities")
    return df


def run_ssn_leiden_domain_interpretation(
    communities: dict[str, int] | None = None,
    top_items_df: pd.DataFrame | None = None,
    out_csv: str | Path | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    top_k: int = 10,
) -> pd.DataFrame:
    """Name Leiden communities (domains) using LLM from SSN data.

    Either pass Leiden result as construct_id -> community_id, or pass a
    pre-built representative items DataFrame. When communities is provided,
    top_items_df is built via build_top_items_from_leiden_communities.

    Args:
        communities: Optional. construct_id -> community_id (int). Ignored if top_items_df is provided.
        top_items_df: Optional. DataFrame with cluster, centrality, item_id, text, construct_reported.
        out_csv: Optional. Path to save cluster, label, rationale, prompt_hint.
        model: OpenAI model name.
        temperature: Sampling temperature.
        top_k: Representative items per community when building from communities.

    Returns:
        DataFrame with columns cluster, label, rationale, prompt_hint.
    """
    if top_items_df is not None:
        df = top_items_df
    elif communities:
        df = build_top_items_from_leiden_communities(communities, top_k=top_k)
    else:
        raise ValueError("Provide either communities or top_items_df")

    if df.empty:
        logger.warning("No clusters to interpret")
        result = pd.DataFrame(columns=["cluster", "label", "rationale", "prompt_hint"])
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out_csv, index=False, encoding="utf-8")
        return result

    clusters = sorted(df["cluster"].unique(), key=_extract_cluster_number)
    records: list[dict[str, Any]] = []

    use_api = _has_openai()
    try:
        import requests as _requests
    except ImportError:
        _requests = None
        if use_api:
            logger.warning("requests not installed; skipping LLM API calls")
        use_api = False
    else:
        use_api = use_api and _requests is not None

    for i, cluster_name in enumerate(clusters, 1):
        cluster_data = df[df["cluster"] == cluster_name]
        n_items = len(cluster_data)
        logger.info("Interpreting community %s (%s/%s, %s items)", cluster_name, i, len(clusters), n_items)

        if use_api:
            api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            api_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            prompt = (
                "You are an expert in personality psychology, psychometrics, and construct mapping. "
                "This analysis uses Leiden community detection on a construct network built from "
                "semantic scale items (e.g. personality/psychological instruments). "
                "Each community groups constructs that are semantically cohesive. "
                "\n\n"
                "Given the most representative items from this community (ranked by intra-community "
                "semantic similarity), assign a short, precise domain label (2–5 words) that captures "
                "the underlying psychological dimension. The label should: "
                "(1) reflect the construct theme rather than surface wording, "
                "(2) be generalizable across instruments, "
                "(3) be distinct from other domains. "
                "Use formal terminology where appropriate (e.g. Extraversion, Conscientiousness, "
                "Emotional Stability, Openness, Agreeableness, or more specific facets). "
                "\n\n"
                "Provide a one-sentence rationale. "
                "Return JSON with fields: label (string, 2–5 words), rationale (string, one sentence).\n\n"
                + _format_prompt(cluster_name, cluster_data)
            )
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            try:
                resp = _requests.post(
                    f"{api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                label = parsed.get("label", "")
                rationale = parsed.get("rationale", "")
                records.append({
                    "cluster": cluster_name,
                    "label": label,
                    "rationale": rationale,
                    "prompt_hint": "",
                })
                logger.info("  %s: %s", cluster_name, label)
            except Exception as e:
                logger.exception("LLM call failed for %s: %s", cluster_name, e)
                records.append({
                    "cluster": cluster_name,
                    "label": "",
                    "rationale": "",
                    "prompt_hint": _format_prompt(cluster_name, cluster_data),
                })
        else:
                records.append({
                    "cluster": cluster_name,
                    "label": "",
                    "rationale": "",
                    "prompt_hint": _format_prompt(cluster_name, cluster_data),
                })
    if not use_api and records:
        logger.warning("No OpenAI API key or requests; interpretations have empty labels")

    result_df = pd.DataFrame(records, columns=["cluster", "label", "rationale", "prompt_hint"])
    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info("Saved SSN Leiden domain interpretations to %s", out_csv)
    return result_df
