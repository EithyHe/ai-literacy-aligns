"""SSN-specific LLM interpretation for cluster / Leiden community naming.

Execution strategy
------------------
1. Cache check  — look up each cluster's cache key (SHA-256 of sorted constructs +
                  model + PROMPT_VERSION).  Cache hits skip all API calls.
2. Anthropic Batch API — all cache-misses are submitted as a single batch request
                  (50 % cheaper than individual Messages API calls).  The function
                  polls until the batch completes or a timeout is reached.
3. Serial fallback — if the batch API is unavailable or a subset of results fails,
                  individual Anthropic calls are made per cluster.
4. OpenAI fallback — if Anthropic is not configured, falls back to the OpenAI
                  chat-completions endpoint with retry-on-429 logic.

Cache persistence
-----------------
Results are written to:
  data/processed/visualization_cache/llm_interpret_cache.json

The cache is keyed by SHA-256(sorted-constructs | model | PROMPT_VERSION).
Bumping _PROMPT_VERSION below invalidates all prior entries automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ssn.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, PROCESSED_DIR
from ssn.db.schema import get_all_constructs, get_items_by_construct
from ssn.services.embedding_service import load_item_embeddings

logger = logging.getLogger(__name__)

# ── Prompt versioning ─────────────────────────────────────────────────────────
# Bump this string whenever the prompt text changes; all cached results built
# with the old version will be treated as misses and re-fetched.
_PROMPT_VERSION = "v1"

_PROMPT_PREFIX = (
    "You are an expert in personality psychology, psychometrics, and construct mapping. "
    "This analysis uses community detection on a construct network built from "
    "semantic scale items (e.g. personality/psychological instruments). "
    "Each community groups constructs that are semantically cohesive.\n\n"
    "Given the most representative items from this community (ranked by intra-community "
    "semantic similarity), assign a short, precise domain label (2–5 words) that captures "
    "the underlying psychological dimension. The label should:\n"
    "(1) reflect the construct theme rather than surface wording,\n"
    "(2) be generalizable across instruments,\n"
    "(3) be distinct from other domains.\n"
    "Use formal terminology where appropriate (e.g. Extraversion, Conscientiousness, "
    "Emotional Stability, Openness, Agreeableness, or more specific facets).\n\n"
    "Provide a one-sentence rationale.\n"
    "Return JSON only with exactly two fields: "
    "\"label\" (string, 2–5 words) and \"rationale\" (string, one sentence).\n\n"
)

_CACHE_PATH = PROCESSED_DIR / "visualization_cache" / "llm_interpret_cache.json"

_BATCH_POLL_INTERVAL = 5   # seconds between batch status polls
_BATCH_TIMEOUT = 300       # seconds before giving up on a batch

# Optional: load .env from project root and ssn/.env
try:
    import dotenv
    _project_root = Path(__file__).resolve().parent.parent.parent
    for _env in [_project_root / ".env", _project_root / "ssn" / ".env"]:
        if _env.exists():
            dotenv.load_dotenv(_env, override=False)
except ImportError:
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY)


def _has_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY)


def _anthropic_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY


def _extract_cluster_number(cluster_name: Any) -> int:
    if isinstance(cluster_name, (int, float)):
        return int(cluster_name)
    if isinstance(cluster_name, bytes):
        cluster_name = cluster_name.decode("utf-8")
    numbers = re.findall(r"\d+", str(cluster_name))
    return int(numbers[0]) if numbers else 999999


def _format_prompt(cluster_name: str, df_cluster: pd.DataFrame, max_items: int = 15) -> str:
    lines = []
    for _, r in df_cluster.head(max_items).iterrows():
        txt = str(r.get("text") or "")
        cst = str(r.get("construct_reported") or r.get("construct_name") or "")
        centrality = float(r.get("centrality", 0))
        lines.append(f"- [Centrality:{centrality:.3f}] {cst} :: {txt[:160]}")
    return f"Community: {cluster_name}\n" + "\n".join(lines)


def _parse_llm_json(text: str) -> dict[str, str]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]).rstrip("`").strip()
    return json.loads(text)


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_key(construct_items: list[str], model: str) -> str:
    """Stable SHA-256 key based on sorted unique construct names + model + prompt version."""
    canonical = "|".join(sorted(set(construct_items))) + f"|{model}|{_PROMPT_VERSION}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _load_cache() -> dict[str, Any]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(_CACHE_PATH.read_text("utf-8"))
        if not isinstance(data, dict):
            return {}
        # Invalidate if prompt version changed
        if data.get("prompt_version") != _PROMPT_VERSION:
            logger.info("Prompt version changed (%s → %s); discarding stale cache",
                        data.get("prompt_version"), _PROMPT_VERSION)
            return {}
        return data.get("entries", {})
    except Exception as e:
        logger.warning("Could not read LLM interpret cache: %s", e)
        return {}


def _save_cache(entries: dict[str, Any]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(
        json.dumps({"prompt_version": _PROMPT_VERSION, "entries": entries},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Anthropic helpers ─────────────────────────────────────────────────────────

def _anthropic_batch_interpret(
    cluster_prompts: dict[str, str],
    model: str,
    temperature: float,
) -> dict[str, dict[str, str]]:
    """Submit all prompts as one Anthropic Message Batch; poll until done.

    Returns a dict mapping cluster_name -> {label, rationale} for successful results.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=_anthropic_key())
    requests = [
        {
            "custom_id": f"cluster-{cname}",
            "params": {
                "model": model,
                "max_tokens": 256,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        for cname, prompt in cluster_prompts.items()
    ]

    logger.info("Submitting Anthropic batch with %d requests (model=%s)", len(requests), model)
    batch = client.messages.batches.create(requests=requests)
    logger.info("Batch created: id=%s", batch.id)

    deadline = time.monotonic() + _BATCH_TIMEOUT
    while batch.processing_status == "in_progress":
        if time.monotonic() > deadline:
            logger.warning("Batch %s timed out after %ss", batch.id, _BATCH_TIMEOUT)
            break
        time.sleep(_BATCH_POLL_INTERVAL)
        batch = client.messages.batches.retrieve(batch.id)
        logger.debug("Batch %s status: %s", batch.id, batch.processing_status)

    results: dict[str, dict[str, str]] = {}
    if batch.processing_status != "ended":
        logger.warning("Batch did not reach 'ended' state (status=%s)", batch.processing_status)
        return results

    for item in client.messages.batches.results(batch.id):
        cname = str(item.custom_id).removeprefix("cluster-")
        if item.result.type != "succeeded":
            logger.warning("Batch item %s failed: type=%s", cname, item.result.type)
            continue
        try:
            parsed = _parse_llm_json(item.result.message.content[0].text)
            results[cname] = {
                "label": str(parsed.get("label") or "").strip(),
                "rationale": str(parsed.get("rationale") or "").strip(),
            }
            logger.info("  [Batch] %s: %s", cname, results[cname]["label"])
        except Exception as e:
            logger.warning("Could not parse batch result for %s: %s", cname, e)

    return results


def _anthropic_serial_interpret(
    cluster_prompts: dict[str, str],
    model: str,
    temperature: float,
) -> dict[str, dict[str, str]]:
    """Call Anthropic one cluster at a time (fallback when batch is unavailable)."""
    import anthropic

    client = anthropic.Anthropic(api_key=_anthropic_key())
    results: dict[str, dict[str, str]] = {}

    for cname, prompt in cluster_prompts.items():
        try:
            message = client.messages.create(
                model=model,
                max_tokens=256,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = _parse_llm_json(message.content[0].text)
            results[cname] = {
                "label": str(parsed.get("label") or "").strip(),
                "rationale": str(parsed.get("rationale") or "").strip(),
            }
            logger.info("  [Serial] %s: %s", cname, results[cname]["label"])
        except Exception as e:
            logger.warning("Anthropic serial call failed for %s: %s", cname, e)
    return results


# ── Core interpretation function ──────────────────────────────────────────────

def _interpret_uncached(
    cluster_prompts: dict[str, str],
    model: str,
    temperature: float,
) -> dict[str, dict[str, str]]:
    """Resolve labels for all uncached clusters using the best available backend."""
    if not cluster_prompts:
        return {}

    if _has_anthropic():
        # Try batch first (50 % cost reduction)
        try:
            results = _anthropic_batch_interpret(cluster_prompts, model, temperature)
            # Serial fallback for any items the batch missed
            missed = {cn: p for cn, p in cluster_prompts.items() if cn not in results}
            if missed:
                logger.info("Batch missed %d clusters; retrying serially", len(missed))
                results.update(_anthropic_serial_interpret(missed, model, temperature))
            return results
        except Exception as e:
            logger.warning("Anthropic batch failed (%s); trying serial Anthropic", e)
            try:
                return _anthropic_serial_interpret(cluster_prompts, model, temperature)
            except Exception as e2:
                logger.warning("Serial Anthropic also failed (%s); trying OpenAI", e2)

    if _has_openai():
        try:
            import requests as _requests
        except ImportError:
            logger.warning("requests not installed; cannot call OpenAI")
            return {}

        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        api_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        results: dict[str, dict[str, str]] = {}

        for i, (cname, prompt) in enumerate(cluster_prompts.items(), 1):
            if i > 1:
                time.sleep(1.5)
            label, rationale = "", ""
            for attempt in range(4):
                try:
                    resp = _requests.post(
                        f"{api_base}/chat/completions",
                        headers=headers,
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": temperature,
                            "response_format": {"type": "json_object"},
                        },
                        timeout=60,
                    )
                    if resp.status_code == 429:
                        wait = 2 ** attempt * 5
                        logger.warning("OpenAI rate limited on %s, retrying in %ss", cname, wait)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
                    label = str(parsed.get("label") or "").strip()
                    rationale = str(parsed.get("rationale") or "").strip()
                    logger.info("  [OpenAI] %s: %s", cname, label)
                    break
                except Exception as e:
                    logger.exception("OpenAI call failed for %s (attempt %d): %s", cname, attempt + 1, e)
                    if attempt < 3:
                        time.sleep(2 ** attempt * 3)
            results[cname] = {"label": label, "rationale": rationale}
        return results

    logger.warning("No LLM API key configured; returning empty labels")
    return {}


# ── Public API ────────────────────────────────────────────────────────────────

def build_top_items_from_leiden_communities(
    communities: dict[str, int],
    top_k: int = 10,
) -> pd.DataFrame:
    """Build representative-items DataFrame per Leiden community for LLM input."""
    from sklearn.metrics.pairwise import cosine_similarity

    construct_names = {c["construct_id"]: c.get("name", c["construct_id"]) for c in get_all_constructs()}
    embeddings, item_ids_ordered = load_item_embeddings()
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids_ordered)}

    item_info: list[tuple[str, str, int, str]] = []
    for construct_id, comm_id in communities.items():
        for it in get_items_by_construct(construct_id):
            iid = str(it.get("item_id", ""))
            if iid not in item_id_to_idx:
                continue
            item_info.append((iid, construct_id, comm_id, it.get("text") or ""))

    if not item_info:
        logger.warning("No items found for any Leiden community")
        return pd.DataFrame(columns=["cluster", "centrality", "item_id", "text", "construct_reported"])

    unique_communities = sorted(set(cid for (_, _, cid, _) in item_info))
    rows: list[dict[str, Any]] = []

    for comm_id in unique_communities:
        comm_items = [(iid, cid, text) for (iid, cid, c, text) in item_info if c == comm_id]
        iids = [x[0] for x in comm_items]
        indices = [item_id_to_idx[iid] for iid in iids]
        sub_emb = embeddings[indices]
        if sub_emb.shape[0] == 1:
            centralities = np.array([0.0])
        else:
            sim = cosine_similarity(sub_emb, sub_emb)
            np.fill_diagonal(sim, 0)
            n_other = sim.shape[0] - 1
            centralities = sim.sum(axis=1) / n_other if n_other > 0 else np.zeros(sim.shape[0])

        for pos in np.argsort(-centralities)[:top_k]:
            rows.append({
                "cluster": comm_id,
                "centrality": float(centralities[pos]),
                "item_id": iids[pos],
                "text": comm_items[pos][2],
                "construct_reported": construct_names.get(comm_items[pos][1], comm_items[pos][1]),
            })

    df = pd.DataFrame(rows)
    logger.info("Built top items: %d rows, %d communities", len(df), len(unique_communities))
    return df


def run_ssn_leiden_domain_interpretation(
    communities: dict[str, int] | None = None,
    top_items_df: pd.DataFrame | None = None,
    out_csv: str | Path | None = None,
    model: str = "claude-haiku-4-5-20251001",
    temperature: float = 0.2,
    top_k: int = 10,
) -> pd.DataFrame:
    """Name clusters/communities using LLM with cache + batch optimisation.

    Execution order:
      1. Cache lookup (disk-persisted, keyed on cluster contents + model + prompt version)
      2. Anthropic Message Batches API for all cache-misses (50 % cost vs. serial)
      3. Serial Anthropic fallback for any items not returned by the batch
      4. OpenAI chat-completions fallback if Anthropic is unavailable

    Args:
        communities:  construct_id → community_id mapping (ignored when top_items_df given).
        top_items_df: Pre-built representative items DataFrame.
        out_csv:      Optional CSV path to persist results.
        model:        LLM model name (Anthropic or OpenAI model ID).
        temperature:  Sampling temperature.
        top_k:        Representative items per community when building from communities.

    Returns:
        DataFrame with columns: cluster, label, rationale, prompt_hint.
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
    cache = _load_cache()

    # ── Phase 1: cache lookup ─────────────────────────────────────────────────
    resolved: dict[str, dict[str, str]] = {}
    uncached_prompts: dict[str, str] = {}
    uncached_keys: dict[str, str] = {}   # cluster_name -> cache_key

    for cname in clusters:
        cluster_data = df[df["cluster"] == cname]
        construct_items = cluster_data["construct_reported"].dropna().tolist()
        key = _cache_key(construct_items, model)
        if key in cache:
            entry = cache[key]
            resolved[str(cname)] = {"label": entry.get("label", ""), "rationale": entry.get("rationale", "")}
            logger.info("Cache hit  cluster=%s  label=%s", cname, entry.get("label"))
        else:
            prompt = _PROMPT_PREFIX + _format_prompt(cname, cluster_data)
            uncached_prompts[str(cname)] = prompt
            uncached_keys[str(cname)] = key

    logger.info("%d/%d clusters from cache; %d need LLM calls",
                len(resolved), len(clusters), len(uncached_prompts))

    # ── Phase 2: LLM calls for uncached clusters ──────────────────────────────
    if uncached_prompts:
        fresh = _interpret_uncached(uncached_prompts, model, temperature)
        now = datetime.now(timezone.utc).isoformat()
        for cname, result in fresh.items():
            resolved[cname] = result
            if result.get("label"):  # only cache successful results
                cache[uncached_keys[cname]] = {**result, "model": model, "cached_at": now}
        _save_cache(cache)

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    records: list[dict[str, Any]] = []
    for cname in clusters:
        r = resolved.get(str(cname), {})
        label = r.get("label", "")
        records.append({
            "cluster": cname,
            "label": label,
            "rationale": r.get("rationale", ""),
            "prompt_hint": "" if label else _format_prompt(cname, df[df["cluster"] == cname]),
        })

    result_df = pd.DataFrame(records, columns=["cluster", "label", "rationale", "prompt_hint"])
    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info("Saved interpretations to %s", out_csv)
    return result_df
