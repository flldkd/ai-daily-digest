"""Filter, deduplicate, score and split news items into three independent tracks."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .utils import (
    CONFIG_DIR,
    load_seen_urls,
    load_yaml,
    save_seen_urls,
)

logger = logging.getLogger(__name__)

_SOURCE_WEIGHT = {"critical": 1.0, "high": 0.8, "medium": 0.5}


def _keyword_score(item: dict, kw_cfg: dict) -> float:
    text = (item["title"] + " " + item["summary"][:200]).lower()
    score = 0.0
    for kw in kw_cfg.get("critical", []):
        if kw.lower() in text:
            score += 10
    for kw in kw_cfg.get("high", []):
        if kw.lower() in text:
            score += 5
    for kw in kw_cfg.get("medium", []):
        if kw.lower() in text:
            score += 2
    return score


def _recency_bonus(published: datetime) -> float:
    now = datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    diff_hours = (now - published).total_seconds() / 3600
    if diff_hours < 3:
        return 5
    if diff_hours < 6:
        return 3
    if diff_hours < 12:
        return 1
    return 0


def _personal_score(item: dict, interests: list[str]) -> float:
    """Score based purely on personal interest relevance (+8 per matched phrase)."""
    text = (item["title"] + " " + item["summary"][:200]).lower()
    score = 0.0
    for phrase in interests:
        if phrase.lower() in text:
            score += 8
    return score


def filter_and_score_tracks(
    raw_items: list[dict],
    user_cfg: dict,
    lookback_hours: int,
) -> dict[str, list[dict]]:
    """
    Filter all items then split into three independent parallel tracks:

    - industry:      non-academic sources, ranked by keyword×source_weight + recency
    - impact_papers: academic sources (tag="academic"), ranked by general keyword score
    - domain_papers: academic sources, ranked by personal_interests relevance score

    A paper can appear in both impact_papers and domain_papers if it qualifies for both.
    Returns {"industry": [...], "impact_papers": [...], "domain_papers": [...]}.
    """
    kw_cfg = load_yaml(CONFIG_DIR / "keywords.yaml").get("keywords", {})
    filt_cfg = user_cfg.get("filter", {})
    tracks_cfg = user_cfg.get("tracks", {})
    block_kws: list[str] = [k.lower() for k in (filt_cfg.get("block_keywords") or [])]
    interests: list[str] = user_cfg.get("personal_interests") or []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    seen_urls = load_seen_urls()

    # Stage 1: time window, dedup, block-keyword filter
    candidates: list[dict] = []
    for item in raw_items:
        published = item["published"]
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)

        if published < cutoff:
            continue
        if item["url"] in seen_urls:
            continue

        text_lower = (item["title"] + " " + item["summary"]).lower()
        if any(bk in text_lower for bk in block_kws):
            logger.debug("Blocked by keyword: %s", item["title"])
            continue

        candidates.append(item)

    # Stage 2: split by source type
    # Academic sources have "academic" in their tags (arXiv, etc.)
    academic = [i for i in candidates if "academic" in (i.get("source_tags") or [])]
    industry_raw = [i for i in candidates if "academic" not in (i.get("source_tags") or [])]

    # Stage 3: per-track scoring and selection
    ind_cfg = tracks_cfg.get("industry", {})
    ind_min: float = ind_cfg.get("min_score", 0)
    ind_max: int = ind_cfg.get("max_items", 8)

    imp_cfg = tracks_cfg.get("impact_papers", {})
    imp_min: float = imp_cfg.get("min_score", 0)
    imp_max: int = imp_cfg.get("max_items", 5)

    dom_cfg = tracks_cfg.get("domain_papers", {})
    dom_min: float = dom_cfg.get("min_personal_score", 8)
    dom_max: int = dom_cfg.get("max_items", 5)

    # Industry track: keyword relevance weighted by source priority + recency
    for item in industry_raw:
        kw = _keyword_score(item, kw_cfg)
        sw = _SOURCE_WEIGHT.get(item.get("source_priority", "medium"), 0.5)
        item["score"] = kw * sw + _recency_bonus(item["published"])

    industry_track = sorted(
        [i for i in industry_raw if i["score"] >= ind_min],
        key=lambda x: x["score"],
        reverse=True,
    )[:ind_max]

    # Academic items: compute both impact and personal scores up front
    for item in academic:
        kw = _keyword_score(item, kw_cfg)
        sw = _SOURCE_WEIGHT.get(item.get("source_priority", "medium"), 0.5)
        item["impact_score"] = kw * sw + _recency_bonus(item["published"])
        item["personal_score"] = _personal_score(item, interests)
        item["score"] = item["impact_score"]  # used by summarizer / archive

    # Impact papers track: high general keyword quality
    impact_track = sorted(
        [i for i in academic if i["impact_score"] >= imp_min],
        key=lambda x: x["impact_score"],
        reverse=True,
    )[:imp_max]

    # Domain papers track: personal relevance (independent of impact score)
    domain_track = sorted(
        [i for i in academic if i["personal_score"] >= dom_min],
        key=lambda x: x["personal_score"],
        reverse=True,
    )[:dom_max]

    # Persist all selected URLs for dedup in future runs
    all_selected = industry_track + impact_track + domain_track
    save_seen_urls([item["url"] for item in all_selected])

    logger.info(
        "Tracks — industry: %d, impact_papers: %d, domain_papers: %d",
        len(industry_track),
        len(impact_track),
        len(domain_track),
    )

    return {
        "industry": industry_track,
        "impact_papers": impact_track,
        "domain_papers": domain_track,
    }
