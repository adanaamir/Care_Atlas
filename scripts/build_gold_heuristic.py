"""
scripts/build_gold_heuristic.py
================================
FAST-PATH: Generates Silver + Gold parquets directly from the Nigeria
Bronze parquet using heuristic capability inference — NO LLM calls.

Why this exists:
  The /api/nearest endpoint only needs the Gold parquet (trust_score +
  extraction_json + lat/lon). For a hackathon MVP, we can infer reasonable
  capabilities from the facility category field and produce a Gold table
  in seconds instead of hours.

  For the full LLM-extracted Silver → Gold pipeline, run:
      sehat extract   (needs an OpenAI-compatible key)
      sehat trust

This script produces Gold directly and is safe to re-run.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Make sure src/ is on the path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("PYTHONUTF8", "1")

from sehat.config import get_settings                     # noqa: E402
from sehat.pipeline.trust_score import apply_trust_rules, compute_confidence, build_embedding_text  # noqa: E402
from sehat.schemas import FacilityType                    # noqa: E402
from sehat.storage import parquet_exists, read_parquet, write_parquet  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category → capability extraction (heuristic, no LLM)
# ---------------------------------------------------------------------------

_AVAIL_CONFIRMED = "confirmed"
_AVAIL_CLAIMED = "claimed"
_AVAIL_NOT_PRESENT = "not_present"
_AVAIL_UNCERTAIN = "uncertain"
_FUNC_FUNCTIONAL = "functional"
_FUNC_UNKNOWN = "unknown"
_STAFF_FULL_TIME = "full_time"
_STAFF_ON_CALL = "on_call"
_STAFF_UNKNOWN = "unknown"


def _infer_extraction(category: str, func_stats: str, name: str) -> dict:
    """Build a FacilityExtraction-compatible dict from Nigeria category field."""
    cat = str(category or "").lower()
    func = str(func_stats or "").lower()
    name_l = str(name or "").lower()

    is_functional = func in ("functional", "")
    is_not_functional = func == "not functional"

    # ICU
    if "teaching" in cat or "federal medical" in cat:
        icu = {
            "present": _AVAIL_CONFIRMED,
            "functional_status": _FUNC_FUNCTIONAL if is_functional else _FUNC_UNKNOWN,
            "bed_count": 20,
            "neonatal_icu": _AVAIL_CLAIMED,
            "source_text": f"Category: {category}",
        }
    elif "specialist" in cat or "general hospital" in cat or "district hospital" in cat:
        icu = {
            "present": _AVAIL_CLAIMED,
            "functional_status": _FUNC_FUNCTIONAL if is_functional else _FUNC_UNKNOWN,
            "bed_count": None,
            "neonatal_icu": _AVAIL_UNCERTAIN,
            "source_text": f"Category: {category}",
        }
    elif "cottage" in cat or "comprehensive" in cat or "medical center" in cat:
        icu = {
            "present": _AVAIL_UNCERTAIN,
            "functional_status": _FUNC_UNKNOWN,
            "bed_count": None,
            "neonatal_icu": _AVAIL_UNCERTAIN,
            "source_text": f"Category: {category}",
        }
    else:
        icu = {"present": _AVAIL_NOT_PRESENT, "functional_status": _FUNC_UNKNOWN,
               "bed_count": None, "neonatal_icu": _AVAIL_NOT_PRESENT, "source_text": ""}

    # Ventilator
    if "teaching" in cat or "federal medical" in cat:
        ventilator = {"present": _AVAIL_CONFIRMED, "count": 5, "reliability_note": None, "source_text": ""}
    elif "specialist" in cat or "general hospital" in cat:
        ventilator = {"present": _AVAIL_CLAIMED, "count": None, "reliability_note": None, "source_text": ""}
    else:
        ventilator = {"present": _AVAIL_UNCERTAIN, "count": None, "reliability_note": None, "source_text": ""}

    # Staff
    if "teaching" in cat or "federal medical" in cat:
        staff = {
            "anesthesiologist": _STAFF_FULL_TIME,
            "surgeon": _STAFF_FULL_TIME,
            "general_physician": _STAFF_FULL_TIME,
            "specialist_types": ["Surgery", "Internal Medicine", "Obstetrics", "Paediatrics", "Anaesthesiology"],
            "total_doctor_count": 50,
            "source_text": "",
        }
    elif "specialist" in cat:
        staff = {
            "anesthesiologist": _STAFF_ON_CALL,
            "surgeon": _STAFF_FULL_TIME,
            "general_physician": _STAFF_FULL_TIME,
            "specialist_types": ["Surgery", "Internal Medicine"],
            "total_doctor_count": 15,
            "source_text": "",
        }
    elif "general hospital" in cat or "district hospital" in cat or "cottage" in cat:
        staff = {
            "anesthesiologist": _STAFF_ON_CALL,
            "surgeon": _STAFF_ON_CALL,
            "general_physician": _STAFF_FULL_TIME,
            "specialist_types": ["General Practice"],
            "total_doctor_count": 5,
            "source_text": "",
        }
    elif "comprehensive" in cat or "medical center" in cat:
        staff = {
            "anesthesiologist": _STAFF_UNKNOWN,
            "surgeon": _STAFF_UNKNOWN,
            "general_physician": _STAFF_FULL_TIME,
            "specialist_types": [],
            "total_doctor_count": 2,
            "source_text": "",
        }
    elif "maternity" in cat:
        staff = {
            "anesthesiologist": _STAFF_UNKNOWN,
            "surgeon": _STAFF_UNKNOWN,
            "general_physician": _STAFF_ON_CALL,
            "specialist_types": ["Obstetrics", "Midwifery"],
            "total_doctor_count": 2,
            "source_text": "",
        }
    else:
        staff = {
            "anesthesiologist": _STAFF_UNKNOWN,
            "surgeon": _STAFF_UNKNOWN,
            "general_physician": _STAFF_UNKNOWN,
            "specialist_types": [],
            "total_doctor_count": None,
            "source_text": "",
        }

    # Emergency
    if "teaching" in cat or "federal medical" in cat or "specialist" in cat:
        emergency = {
            "emergency_care": _AVAIL_CONFIRMED,
            "is_24_7": True,
            "ambulance": _AVAIL_CONFIRMED,
            "trauma_capability": _AVAIL_CLAIMED,
            "source_text": "",
        }
    elif "general hospital" in cat or "district hospital" in cat or "cottage" in cat or "comprehensive" in cat:
        emergency = {
            "emergency_care": _AVAIL_CLAIMED,
            "is_24_7": False,
            "ambulance": _AVAIL_UNCERTAIN,
            "trauma_capability": _AVAIL_UNCERTAIN,
            "source_text": "",
        }
    elif "maternity" in cat:
        emergency = {
            "emergency_care": _AVAIL_CLAIMED,
            "is_24_7": False,
            "ambulance": _AVAIL_NOT_PRESENT,
            "trauma_capability": _AVAIL_NOT_PRESENT,
            "source_text": "",
        }
    else:
        emergency = {
            "emergency_care": _AVAIL_UNCERTAIN,
            "is_24_7": False,
            "ambulance": _AVAIL_NOT_PRESENT,
            "trauma_capability": _AVAIL_NOT_PRESENT,
            "source_text": "",
        }

    # Surgery
    if "teaching" in cat or "federal medical" in cat or "specialist" in cat:
        surgery = {
            "general_surgery": _AVAIL_CONFIRMED,
            "appendectomy": _AVAIL_CONFIRMED,
            "caesarean": _AVAIL_CONFIRMED,
            "orthopedic": _AVAIL_CLAIMED,
            "cardiac": _AVAIL_CLAIMED if ("teaching" in cat or "federal medical" in cat) else _AVAIL_UNCERTAIN,
            "source_text": "",
        }
    elif "general hospital" in cat or "district hospital" in cat or "cottage" in cat:
        surgery = {
            "general_surgery": _AVAIL_CLAIMED,
            "appendectomy": _AVAIL_CLAIMED,
            "caesarean": _AVAIL_CLAIMED,
            "orthopedic": _AVAIL_UNCERTAIN,
            "cardiac": _AVAIL_NOT_PRESENT,
            "source_text": "",
        }
    elif "maternity" in cat or "comprehensive" in cat:
        surgery = {
            "general_surgery": _AVAIL_UNCERTAIN,
            "appendectomy": _AVAIL_NOT_PRESENT,
            "caesarean": _AVAIL_CONFIRMED if "maternity" in cat else _AVAIL_CLAIMED,
            "orthopedic": _AVAIL_NOT_PRESENT,
            "cardiac": _AVAIL_NOT_PRESENT,
            "source_text": "",
        }
    else:
        surgery = {
            "general_surgery": _AVAIL_NOT_PRESENT,
            "appendectomy": _AVAIL_NOT_PRESENT,
            "caesarean": _AVAIL_NOT_PRESENT,
            "orthopedic": _AVAIL_NOT_PRESENT,
            "cardiac": _AVAIL_NOT_PRESENT,
            "source_text": "",
        }

    # Dialysis
    if "teaching" in cat or "federal medical" in cat:
        dialysis = {"present": _AVAIL_CONFIRMED, "machine_count": 4, "source_text": ""}
    elif "specialist" in cat:
        dialysis = {"present": _AVAIL_CLAIMED, "machine_count": None, "source_text": ""}
    else:
        dialysis = {"present": _AVAIL_NOT_PRESENT, "machine_count": None, "source_text": ""}

    # Specialties
    specialties = []
    if "teaching" in cat or "federal medical" in cat:
        specialties = ["Surgery", "Internal Medicine", "Obstetrics & Gynaecology",
                       "Paediatrics", "Anaesthesiology", "Radiology", "Pathology"]
    elif "specialist" in cat:
        specialties = ["Specialist care", "Surgery"]
    elif "maternity" in cat:
        specialties = ["Obstetrics", "Midwifery", "Postnatal care"]
    elif "general hospital" in cat or "district hospital" in cat or "cottage" in cat:
        specialties = ["General Medicine", "Surgery", "Maternity"]
    elif "dispensary" in cat or "primary" in cat:
        specialties = ["Primary care", "Immunisation", "Antenatal"]

    return {
        "facility_id": "",  # filled in by caller
        "icu": icu,
        "ventilator": ventilator,
        "staff": staff,
        "emergency": emergency,
        "surgery": surgery,
        "dialysis": dialysis,
        "specialties_extracted": specialties,
        "extraction_notes": f"Heuristic inference from category: {category}",
        "raw_text_used": f"Category: {category}. Functional: {func_stats}.",
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_gold_heuristic() -> None:
    s = get_settings()

    # Post-ingest Bronze (standard sehat columns, has facility_id)
    if not parquet_exists(s.bronze_path):
        log.error("Bronze parquet missing. Run: python scripts/load_nigeria.py && python -m sehat.cli ingest")
        sys.exit(1)

    # Raw Nigeria Bronze (has category, functional_status — needed for heuristics)
    nigeria_raw_path = REPO_ROOT / "data" / "nigeria_bronze.parquet"
    if not nigeria_raw_path.exists():
        log.error("Nigeria raw bronze missing at %s. Run: python scripts/load_nigeria.py", nigeria_raw_path)
        sys.exit(1)

    ingest_bronze = read_parquet(s.bronze_path)          # post-ingest, has facility_id
    nigeria_raw = pd.read_parquet(nigeria_raw_path)      # raw, has category/functional_status

    log.info("Loaded post-ingest Bronze: %d rows", len(ingest_bronze))
    log.info("Loaded Nigeria raw Bronze: %d rows", len(nigeria_raw))

    # Re-calculate facility_id on nigeria_raw using the exact logic from ingest.py
    import hashlib
    def _hash_id(*parts: object) -> str:
        joined = "|".join("" if p is None else str(p).strip().lower() for p in parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    nigeria_raw["facility_id"] = nigeria_raw.apply(
        lambda r: _hash_id(r["name"], r.get("address_city"), r.get("address_zipOrPostcode")),
        axis=1,
    )

    # Merge on facility_id to get both standard columns AND Nigeria-specific ones
    nigeria_cols = ["facility_id", "category", "functional_status", "accessibility_note"]
    # Drop duplicates on facility_id in nigeria_raw to avoid cross-join expansion
    nigeria_raw_dedup = nigeria_raw.drop_duplicates(subset=["facility_id"])
    bronze = ingest_bronze.merge(
        nigeria_raw_dedup[nigeria_cols],
        on="facility_id",
        how="left",
        suffixes=("", "_nga"),
    )
    # Fill any missing category/status
    bronze["category"] = bronze.get("category", "").fillna("")
    bronze["functional_status"] = bronze.get("functional_status", "").fillna("")
    log.info("Merged Bronze: %d rows", len(bronze))


    # Build fake Silver (extraction_json for each row)
    silver_records = []
    for _, row in bronze.iterrows():
        ext = _infer_extraction(
            category=row.get("category", ""),
            func_stats=row.get("functional_status", ""),
            name=row.get("name", ""),
        )
        ext["facility_id"] = str(row["facility_id"])
        silver_records.append({
            "facility_id": row["facility_id"],
            "extraction_json": json.dumps(ext, ensure_ascii=False),
        })

    silver_df = pd.DataFrame(silver_records)
    write_parquet(silver_df, s.silver_path, overwrite=True)
    log.info("Silver written: %d rows -> %s", len(silver_df), s.silver_path)

    # Merge Silver + Bronze → apply trust rules → build Gold
    merged = silver_df.merge(bronze, on="facility_id", how="inner", suffixes=("", "_bronze"))
    log.info("Merged: %d rows", len(merged))

    gold_records = []
    for _, row in merged.iterrows():
        try:
            extraction = json.loads(row["extraction_json"])
        except (ValueError, TypeError):
            continue

        ftype_raw = row.get("facilityTypeId")
        facility_type = FacilityType.normalise(ftype_raw if isinstance(ftype_raw, str) else None)

        ctl_raw = row.get("composite_text_length")
        ctl = int(ctl_raw) if pd.notna(ctl_raw) else 60  # default to non-trivial

        # Use functional_status to adjust ctl (non-functional = less trust)
        func = str(row.get("functional_status") or "").lower()
        if func == "not functional":
            ctl = 0  # triggers EMPTY_DESCRIPTION flag → lower trust

        trust_score, flags = apply_trust_rules(
            extraction,
            facility_type=facility_type,
            number_doctors=None,
            composite_text_length=ctl,
        )

        # Boost trust for verified categories
        cat = str(row.get("category") or "").lower()
        if "teaching" in cat or "federal medical" in cat:
            trust_score = min(1.0, trust_score * 1.25)
        elif "specialist" in cat:
            trust_score = min(1.0, trust_score * 1.10)

        # Non-functional penalty
        if func == "not functional":
            trust_score = max(0.05, trust_score * 0.40)
        elif func == "partially functional":
            trust_score = max(0.05, trust_score * 0.70)

        trust_score = round(trust_score, 3)

        confidence = compute_confidence(extraction, flags=flags, composite_text_length=ctl)
        embedding_text = build_embedding_text(
            name=str(row.get("name") or ""),
            city=row.get("address_city"),
            state=row.get("address_stateOrRegion"),
            pin_code=row.get("address_zipOrPostcode"),
            facility_type=facility_type,
            extraction=extraction,
            trust_score=trust_score,
        )

        gold_records.append({
            "facility_id": row["facility_id"],
            "name": str(row.get("name") or ""),
            "address_city": str(row.get("address_city") or ""),
            "address_state": str(row.get("address_stateOrRegion") or ""),
            "address_zip": str(row.get("address_zipOrPostcode") or ""),
            "latitude": float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
            "longitude": float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
            "facility_type": facility_type.value,
            "operator_type": str(row.get("operatorTypeId") or ""),
            "category": str(row.get("category") or ""),
            "functional_status": str(row.get("functional_status") or ""),
            "accessibility_note": str(row.get("accessibility_note") or ""),
            "extraction_json": json.dumps(extraction, ensure_ascii=False),
            "trust_score": trust_score,
            "trust_flags_json": json.dumps([f.model_dump(mode="json") for f in flags], ensure_ascii=False),
            "confidence_json": confidence.model_dump_json(),
            "correction_iterations": 0,
            "embedding_text": embedding_text,
            "extraction_version": "heuristic-1.0",
        })

    gold_df = pd.DataFrame(gold_records)
    write_parquet(gold_df, s.gold_path, overwrite=True)

    avg_trust = float(gold_df["trust_score"].mean())
    high_trust_pct = float((gold_df["trust_score"] >= 0.7).mean())
    log.info("Gold written: %d rows | avg trust=%.3f | high-trust %.1f%%",
             len(gold_df), avg_trust, high_trust_pct * 100)
    log.info("Gold path: %s", s.gold_path)
    print(f"\nGold table built: {len(gold_df):,} facilities")
    print(f"Average trust score: {avg_trust:.3f}")
    print(f"High-trust (>= 0.70): {high_trust_pct:.1%}")
    print(f"Output: {s.gold_path}")
    print("\nNext: python -m sehat.cli index  (builds FAISS for /api/query)")
    print("      python -m sehat.cli deserts")
    print("      python -m sehat.cli serve")


if __name__ == "__main__":
    build_gold_heuristic()
