"""
scripts/load_nigeria.py
=======================
Downloads the Nigeria healthcare facility dataset from HuggingFace and
ingests it into the CareAtlas Bronze parquet format compatible with the
existing Sehat pipeline.

Dataset: electricsheepafrica/africa-nigeria-health-care-facilities-in-nigeria
~46,146 rows across train + test splits.

Nigeria schema → CareAtlas Bronze column mapping
-------------------------------------------------
prmry_name         → name
latitude           → latitude
longitude          → longitude
statename          → address_stateOrRegion
lganame            → address_city
wardname           → address_zipOrPostcode  (used as proxy for local area)
type               → facilityTypeId
category           → category  (Primary / Secondary / Tertiary)
func_stats         → functional_status
ownership          → operatorTypeId
uniq_id            → source_id
globalid           → globalid
accessblty         → accessibility_note
source             → data_source
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
BRONZE_OUT = DATA_DIR / "nigeria_bronze.parquet"
FACILITIES_CSV_OUT = DATA_DIR / "facilities.csv"   # sehat ingest reads this

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HF_DATASET = "electricsheepafrica/africa-nigeria-health-care-facilities-in-nigeria"


# ---------------------------------------------------------------------------
# Nigeria facility-type normalisation
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    "hospital": "hospital",
    "health centre": "clinic",
    "health center": "clinic",
    "clinic": "clinic",
    "dispensary": "clinic",
    "maternity": "clinic",
    "pharmacy": "pharmacy",
    "laboratory": "clinic",
    "dental": "dentist",
    "dentist": "dentist",
    "doctor": "doctor",
    "primary health care": "clinic",
    "phc": "clinic",
}


def _normalise_type(raw: str | None) -> str:
    if not raw:
        return "unknown"
    key = str(raw).strip().lower()
    for k, v in _TYPE_MAP.items():
        if k in key:
            return v
    return "unknown"


def _make_facility_id(row: pd.Series) -> str:
    """Stable hash id from (name, lat, lon) so re-runs are idempotent."""
    key = f"{row.get('name', '')}|{row.get('latitude', '')}|{row.get('longitude', '')}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _build_composite_text(row: pd.Series) -> str:
    """Build the text field the LLM extraction prompt will read."""
    parts = []
    if row.get("name"):
        parts.append(f"Facility name: {row['name']}.")
    if row.get("category"):
        parts.append(f"Category: {row['category']}.")
    if row.get("facilityTypeId") and row["facilityTypeId"] != "unknown":
        parts.append(f"Type: {row['facilityTypeId']}.")
    if row.get("functional_status"):
        parts.append(f"Functional status: {row['functional_status']}.")
    if row.get("operatorTypeId"):
        parts.append(f"Ownership: {row['operatorTypeId']}.")
    if row.get("address_city"):
        parts.append(f"LGA: {row['address_city']}.")
    if row.get("address_stateOrRegion"):
        parts.append(f"State: {row['address_stateOrRegion']}.")
    if row.get("accessibility_note"):
        parts.append(f"Accessibility: {row['accessibility_note']}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_nigeria_dataset() -> pd.DataFrame:
    log.info("Loading Nigeria dataset from HuggingFace: %s", HF_DATASET)
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
    except ImportError:
        log.error("'huggingface_hub' package not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    frames = []
    # The dataset has train + test splits, each as a single parquet file
    for split in ("train", "test"):
        filename = f"data/{split}-00000-of-00001.parquet"
        log.info("  Downloading split: %s", filename)
        try:
            local_path = hf_hub_download(
                repo_id=HF_DATASET,
                filename=filename,
                repo_type="dataset",
                local_dir=str(DATA_DIR / "hf_cache"),
            )
            df = pd.read_parquet(local_path)
            log.info("  Split '%s': %d rows", split, len(df))
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            log.warning("  Could not download split '%s': %s — skipping", split, exc)

    if not frames:
        log.error("No splits could be downloaded. Check your internet connection.")
        sys.exit(1)

    raw = pd.concat(frames, ignore_index=True)
    log.info("Total rows from HuggingFace: %d", len(raw))
    return raw


def transform_to_bronze(raw: pd.DataFrame) -> pd.DataFrame:
    """Map Nigeria columns to the CareAtlas Bronze schema."""
    df = pd.DataFrame()

    # Core identity
    df["name"] = raw.get("prmry_name", raw.get("alt_name", "")).fillna("").str.strip()

    # Geo
    df["latitude"] = pd.to_numeric(raw.get("latitude", raw.get("y")), errors="coerce")
    df["longitude"] = pd.to_numeric(raw.get("longitude", raw.get("x")), errors="coerce")

    # Location
    df["address_stateOrRegion"] = raw.get("statename", "").fillna("").str.strip()
    df["address_city"] = raw.get("lganame", "").fillna("").str.strip()
    df["address_zipOrPostcode"] = raw.get("wardname", "").fillna("").str.strip()

    # Facility classification
    df["facilityTypeId"] = raw.get("type", "").apply(_normalise_type)
    df["category"] = raw.get("category", "").fillna("").str.strip()
    df["functional_status"] = raw.get("func_stats", "").fillna("").str.strip()
    df["operatorTypeId"] = raw.get("ownership", "").fillna("").str.strip()
    df["accessibility_note"] = raw.get("accessblty", "").fillna("").str.strip()

    # Source metadata
    df["data_source"] = raw.get("source", "HDX/NHFR").fillna("HDX/NHFR")
    df["globalid"] = raw.get("globalid", "").fillna("")

    # numberDoctors not in this dataset
    df["numberDoctors"] = None

    # --- Ingest-compatible columns for sehat composite_text ---
    df["description"] = (
        "Facility: " + df["name"].str.strip() + ". "
        + "Category: " + df["category"] + ". "
        + "Ownership: " + df["operatorTypeId"] + ". "
        + "State: " + df["address_stateOrRegion"] + ". "
        + "LGA: " + df["address_city"] + ". "
        + "Functional status: " + df["functional_status"] + "."
    ).str.strip()

    def _infer_capabilities(cat: str) -> str:
        cat = str(cat).lower()
        caps = []
        if "teaching" in cat or "federal medical" in cat:
            caps += ["ICU", "emergency care", "surgery", "dialysis", "specialist services", "24/7 emergency"]
        elif "specialist" in cat:
            caps += ["specialist care", "emergency care", "surgery"]
        elif "general hospital" in cat or "district hospital" in cat or "cottage" in cat:
            caps += ["emergency care", "general surgery", "maternity", "inpatient care"]
        elif "comprehensive" in cat:
            caps += ["emergency care", "maternity", "general surgery"]
        elif "maternity" in cat:
            caps += ["maternity", "antenatal care", "postnatal care", "normal delivery"]
        elif "medical center" in cat:
            caps += ["emergency care", "outpatient care"]
        elif "primary" in cat or "dispensary" in cat or "community" in cat:
            caps += ["outpatient care", "primary care", "immunisation", "antenatal"]
        return ", ".join(caps) if caps else ""

    df["specialties"] = df["category"].apply(_infer_capabilities)
    df["capability"] = df["specialties"]
    df["equipment"] = ""
    df["procedure"] = ""

    # Build composite text
    df["composite_text"] = df.apply(_build_composite_text, axis=1)
    df["composite_text_length"] = df["composite_text"].str.len()

    # Stable facility ID
    df["facility_id"] = df.apply(_make_facility_id, axis=1)

    # Drop rows with no name and no coordinates
    before = len(df)
    df = df[
        (df["name"].str.len() > 0) |
        (df["latitude"].notna() & df["longitude"].notna())
    ].copy()
    log.info("Dropped %d rows with empty name AND no coordinates", before - len(df))

    # Deduplicate on facility_id
    df = df.drop_duplicates(subset=["facility_id"]).reset_index(drop=True)
    log.info("After dedup: %d unique facilities", len(df))

    return df



def save_for_pipeline(bronze: pd.DataFrame) -> None:
    """Write both the raw parquet and the CSV that `sehat ingest` expects."""
    bronze.to_parquet(BRONZE_OUT, index=False)
    log.info("Saved bronze parquet → %s", BRONZE_OUT)

    bronze.to_csv(FACILITIES_CSV_OUT, index=False)
    log.info("Saved facilities CSV → %s  (ready for: sehat ingest)", FACILITIES_CSV_OUT)


def print_schema_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"Nigeria Bronze Schema — {len(df):,} facilities")
    print("=" * 60)
    print(df.dtypes.to_string())
    print("\nSample rows:")
    print(df[["facility_id", "name", "facilityTypeId", "category",
              "functional_status", "address_stateOrRegion",
              "latitude", "longitude"]].head(10).to_string(index=False))

    print(f"\nStates covered: {df['address_stateOrRegion'].nunique()}")
    print(f"Facility types: {df['facilityTypeId'].value_counts().to_dict()}")
    print(f"Functional:     {df['functional_status'].value_counts().to_dict()}")
    print(f"Categories:     {df['category'].value_counts().to_dict()}")
    print(f"Has coords:     {df['latitude'].notna().sum():,} / {len(df):,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    raw = load_nigeria_dataset()
    bronze = transform_to_bronze(raw)
    print_schema_summary(bronze)
    save_for_pipeline(bronze)
    print("Done! Next steps:")
    print("   sehat ingest  (or: python -m sehat.cli ingest)")
    print("   sehat extract")
    print("   sehat trust")
    print("   sehat index")
    print("   sehat deserts")
