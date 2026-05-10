"""
geo_ranking.py
==============
Pure-Python geospatial ranking for the /api/nearest endpoint.

Key design choices:
  * No external geo libs — haversine is ~10 lines, zero deps.
  * Composite score weights are env-tunable via Settings.
  * NeedType → capability check is a simple dict lookup against the
    Gold extraction JSON — no LLM call required, sub-100ms at 46k rows.
  * Works offline: uses only the Gold parquet, not the FAISS index.

Composite score formula:
    score = W_dist * dist_decay + W_trust * trust_score + W_cap * cap_score

Where:
    dist_decay = exp(-dist_km / DECAY_KM)      # 0..1, soft 50km radius
    trust_score = Gold.trust_score             # 0..1, from rules engine
    cap_score   = capability_match(extraction, need_type)   # 0..1
"""

from __future__ import annotations

import json
import math
import logging
from enum import Enum
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DECAY_KM = 15.0        # Distance half-life for exponential decay
MAX_RADIUS_KM = 100.0  # Hard cutoff — don't return facilities > 100 km away
W_DIST = 0.35
W_TRUST = 0.35
W_CAP = 0.30


# ---------------------------------------------------------------------------
# NeedType
# ---------------------------------------------------------------------------


class NeedType(str, Enum):
    EMERGENCY = "emergency"
    ICU = "icu"
    MATERNITY = "maternity"
    SURGERY = "surgery"
    DIALYSIS = "dialysis"
    PEDIATRIC = "pediatric"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Capability matching
# ---------------------------------------------------------------------------

_PRESENT = {"confirmed", "claimed"}
_FULL_STAFF = {"full_time", "on_call"}


def _cap_score_emergency(ext: dict[str, Any]) -> float:
    em = ext.get("emergency") or {}
    score = 0.0
    if em.get("emergency_care") in _PRESENT:
        score += 0.5
    if em.get("is_24_7"):
        score += 0.3
    if em.get("ambulance") in _PRESENT:
        score += 0.2
    return score


def _cap_score_icu(ext: dict[str, Any]) -> float:
    icu = ext.get("icu") or {}
    score = 0.0
    if icu.get("present") == "confirmed":
        score += 0.6
    elif icu.get("present") == "claimed":
        score += 0.35
    if icu.get("functional_status") == "functional":
        score += 0.3
    if icu.get("bed_count") and int(icu.get("bed_count") or 0) > 0:
        score += 0.1
    return score


def _cap_score_maternity(ext: dict[str, Any]) -> float:
    sur = ext.get("surgery") or {}
    em = ext.get("emergency") or {}
    score = 0.0
    if sur.get("caesarean") in _PRESENT:
        score += 0.6
    if em.get("emergency_care") in _PRESENT:
        score += 0.2
    specs = ext.get("specialties_extracted") or []
    if any(s for s in specs if "mat" in s.lower() or "obstet" in s.lower() or "gyn" in s.lower()):
        score += 0.2
    return score


def _cap_score_surgery(ext: dict[str, Any]) -> float:
    sur = ext.get("surgery") or {}
    staff = ext.get("staff") or {}
    score = 0.0
    if sur.get("general_surgery") in _PRESENT:
        score += 0.5
    if staff.get("anesthesiologist") in _FULL_STAFF:
        score += 0.3
    if staff.get("surgeon") in _FULL_STAFF:
        score += 0.2
    return score


def _cap_score_dialysis(ext: dict[str, Any]) -> float:
    dial = ext.get("dialysis") or {}
    score = 0.0
    if dial.get("present") == "confirmed":
        score += 0.7
    elif dial.get("present") == "claimed":
        score += 0.4
    mc = dial.get("machine_count")
    if mc and int(mc or 0) > 0:
        score += 0.3
    return score


def _cap_score_pediatric(ext: dict[str, Any]) -> float:
    specs = ext.get("specialties_extracted") or []
    icu = ext.get("icu") or {}
    score = 0.0
    if any(s for s in specs if "paed" in s.lower() or "pedi" in s.lower() or "child" in s.lower()):
        score += 0.7
    if icu.get("neonatal_icu") in _PRESENT:
        score += 0.3
    return score


def _cap_score_general(ext: dict[str, Any]) -> float:
    """Baseline: any hospital with emergency or GP coverage."""
    em = ext.get("emergency") or {}
    staff = ext.get("staff") or {}
    score = 0.0
    if em.get("emergency_care") in _PRESENT:
        score += 0.5
    if staff.get("general_physician") in _FULL_STAFF:
        score += 0.5
    return score


_CAP_SCORERS = {
    NeedType.EMERGENCY: _cap_score_emergency,
    NeedType.ICU: _cap_score_icu,
    NeedType.MATERNITY: _cap_score_maternity,
    NeedType.SURGERY: _cap_score_surgery,
    NeedType.DIALYSIS: _cap_score_dialysis,
    NeedType.PEDIATRIC: _cap_score_pediatric,
    NeedType.GENERAL: _cap_score_general,
}


def capability_match(extraction_json: str | dict | None, need_type: NeedType) -> float:
    """Return a 0..1 capability match score for a given NeedType."""
    if extraction_json is None:
        return 0.0
    if isinstance(extraction_json, str):
        try:
            ext = json.loads(extraction_json)
        except (ValueError, TypeError):
            return 0.0
    else:
        ext = extraction_json

    scorer = _CAP_SCORERS.get(need_type, _cap_score_general)
    try:
        raw = scorer(ext)
    except Exception:  # noqa: BLE001
        return 0.0
    return round(min(1.0, max(0.0, raw)), 3)


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------


def composite_score(
    dist_km: float,
    trust_score: float,
    cap_score: float,
    *,
    w_dist: float = W_DIST,
    w_trust: float = W_TRUST,
    w_cap: float = W_CAP,
) -> float:
    """Weighted composite score combining distance, trust, and capability."""
    dist_decay = math.exp(-dist_km / DECAY_KM)
    return round(w_dist * dist_decay + w_trust * trust_score + w_cap * cap_score, 4)


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------


def rank_by_proximity(
    gold_df: pd.DataFrame,
    user_lat: float,
    user_lon: float,
    need_type: NeedType = NeedType.GENERAL,
    top_k: int = 5,
    radius_km: float = MAX_RADIUS_KM,
    min_trust_score: float = 0.0,
    functional_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Rank Gold facilities by composite score for a given user location and need.

    Parameters
    ----------
    gold_df : pd.DataFrame
        Gold parquet loaded via DuckDB or read_parquet.
    user_lat, user_lon : float
        User's GPS coordinates.
    need_type : NeedType
        The medical need to route for.
    top_k : int
        Number of results to return.
    radius_km : float
        Hard cutoff radius.
    min_trust_score : float
        Minimum trust score filter.
    functional_only : bool
        If True, skip facilities not explicitly marked Functional (Nigeria dataset).

    Returns
    -------
    List of result dicts sorted by composite_score descending.
    """
    df = gold_df.copy()

    # Drop rows without coordinates
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()

    # Functional filter (Nigeria-specific func_stats column if present)
    if functional_only and "functional_status" in df.columns:
        df = df[df["functional_status"].str.lower().isin(["functional", ""])]

    # Trust score filter
    if min_trust_score > 0:
        df = df[df["trust_score"] >= min_trust_score]

    if df.empty:
        return []

    # Compute haversine distance for all rows (vectorised)
    lat_rad = df["latitude"].apply(math.radians)
    lon_rad = df["longitude"].apply(math.radians)
    u_lat_r = math.radians(user_lat)
    u_lon_r = math.radians(user_lon)

    dphi = lat_rad - u_lat_r
    dlambda = lon_rad - u_lon_r
    a = (dphi / 2).apply(math.sin) ** 2 + (
        math.cos(u_lat_r) * lat_rad.apply(math.cos) * (dlambda / 2).apply(math.sin) ** 2
    )
    df["distance_km"] = (2 * 6_371.0 * a.apply(lambda x: math.asin(math.sqrt(x)))).round(2)

    # Radius cutoff
    df = df[df["distance_km"] <= radius_km]
    if df.empty:
        return []

    # Capability scores
    df["cap_score"] = df["extraction_json"].apply(
        lambda x: capability_match(x, need_type)
    )

    # Composite score
    df["composite_score"] = df.apply(
        lambda r: composite_score(
            r["distance_km"],
            float(r.get("trust_score", 0)),
            r["cap_score"],
        ),
        axis=1,
    )

    # Sort and take top_k
    df = df.sort_values("composite_score", ascending=False).head(top_k)

    results = []
    for rank_idx, (_, row) in enumerate(df.iterrows(), start=1):
        trust_flags: list = []
        try:
            flags_raw = row.get("trust_flags_json")
            if flags_raw and not (isinstance(flags_raw, float) and math.isnan(flags_raw)):
                trust_flags = json.loads(flags_raw)
        except (ValueError, TypeError):
            pass

        extraction: dict = {}
        try:
            ext_raw = row.get("extraction_json")
            if ext_raw:
                extraction = json.loads(ext_raw)
        except (ValueError, TypeError):
            pass

        matched_capabilities = _describe_capabilities(extraction, need_type)

        # Estimate travel time: rough 30 km/h average for Nigerian roads
        travel_min = round(row["distance_km"] / 30.0 * 60)

        results.append(
            {
                "rank": rank_idx,
                "facility_id": str(row["facility_id"]),
                "facility_name": str(row.get("name") or row.get("facility_id")),
                "composite_score": float(row["composite_score"]),
                "distance_km": float(row["distance_km"]),
                "estimated_travel_min": travel_min,
                "trust_score": float(row.get("trust_score", 0)),
                "trust_grade": _trust_grade(float(row.get("trust_score", 0))),
                "cap_score": float(row["cap_score"]),
                "matched_capabilities": matched_capabilities,
                "facility_type": str(row.get("facility_type") or "unknown"),
                "category": str(row.get("category") or ""),
                "functional_status": str(row.get("functional_status") or ""),
                "facility_meta": {
                    "name": str(row.get("name") or ""),
                    "city": str(row.get("address_city") or ""),
                    "state": str(row.get("address_state") or ""),
                    "lga": str(row.get("address_city") or ""),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "operator_type": str(row.get("operator_type") or ""),
                    "trust_score": float(row.get("trust_score", 0)),
                    "trust_grade": _trust_grade(float(row.get("trust_score", 0))),
                    "trust_flags": trust_flags,
                    "accessibility_note": str(row.get("accessibility_note") or ""),
                },
            }
        )

    return results


def _trust_grade(score: float) -> str:
    if score >= 0.85:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.55:
        return "C"
    if score >= 0.40:
        return "D"
    return "F"


def _describe_capabilities(extraction: dict, need_type: NeedType) -> list[str]:
    """Return human-readable capability matches for display in the app."""
    caps = []
    em = extraction.get("emergency") or {}
    icu = extraction.get("icu") or {}
    sur = extraction.get("surgery") or {}
    dial = extraction.get("dialysis") or {}
    staff = extraction.get("staff") or {}
    specs = extraction.get("specialties_extracted") or []

    if em.get("emergency_care") in _PRESENT:
        caps.append("Emergency Care")
    if em.get("is_24_7"):
        caps.append("24/7 Service")
    if em.get("ambulance") in _PRESENT:
        caps.append("Ambulance")
    if icu.get("present") in _PRESENT:
        caps.append(f"ICU ({icu.get('bed_count') or '?'} beds)")
    if sur.get("general_surgery") in _PRESENT:
        caps.append("Surgery")
    if sur.get("caesarean") in _PRESENT:
        caps.append("Maternity / C-Section")
    if dial.get("present") in _PRESENT:
        caps.append("Dialysis")
    if staff.get("anesthesiologist") in _FULL_STAFF:
        caps.append("Anesthesiologist on staff")
    for s in specs[:3]:
        caps.append(s)

    return caps[:6]  # cap at 6 for mobile display


__all__ = [
    "NeedType",
    "haversine",
    "capability_match",
    "composite_score",
    "rank_by_proximity",
]
