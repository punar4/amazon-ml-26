import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TRAIN = ROOT / "dataset" / "train"
TEST = ROOT / "dataset" / "test"


def load_tsv(path):
    print(f"\nLoading: {path}")
    df = pd.read_csv(path, sep="\t")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


print("=" * 70)
print("AMAZON ML CHALLENGE 2026 - DATASET AUDIT")
print("=" * 70)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_s1 = load_tsv(TRAIN / "train_source1.tsv")
train_s2 = load_tsv(TRAIN / "train_source2.tsv")
train_s3 = load_tsv(TRAIN / "train_source3.tsv")
ground_truth = load_tsv(TRAIN / "train_ground_truth.tsv")

# ============================================================
# LOAD TEST DATA
# ============================================================

test_s1 = load_tsv(TEST / "test_source1.tsv")
test_s2 = load_tsv(TEST / "test_source2.tsv")
test_s3 = load_tsv(TEST / "test_source3.tsv")

# ============================================================
# FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print("\nSOURCE 1:")
print(train_s1.head())

print("\nSOURCE 2:")
print(train_s2.head())

print("\nSOURCE 3:")
print(train_s3.head())

print("\nGROUND TRUTH:")
print(ground_truth.head())

# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

for name, df in {
    "train_source1": train_s1,
    "train_source2": train_s2,
    "train_source3": train_s3,
    "ground_truth": ground_truth,
}.items():
    print(f"\n{name}")
    print(df.isna().sum())

# ============================================================
# GROUND TRUTH MATCH DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("GROUND TRUTH MATCH DISTRIBUTION")
print("=" * 70)


def count_matches(value):
    """
    Count how many Source 2 / Source 3 records
    are listed for one Source 1 entity.

    Empty or missing value = 0 matches.
    """
    if pd.isna(value) or str(value).strip() == "":
        return 0

    return len(str(value).split(","))


match_counts = ground_truth["matched_entity_ids"].apply(count_matches)

print("\nNumber of Source 1 entities by number of matches:")
print(match_counts.value_counts().sort_index())

print("\nSummary:")
print(f"Total Source 1 entities: {len(match_counts):,}")
print(f"No-match entities: {(match_counts == 0).sum():,}")
print(f"Entities with at least 1 match: {(match_counts > 0).sum():,}")

print("\nAverage number of matches:")
print(f"{match_counts.mean():.2f}")

print("\nMaximum number of matches for one Source 1 entity:")
print(match_counts.max())

print("\nMatch distribution analysis complete.")