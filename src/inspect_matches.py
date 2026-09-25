import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TRAIN = ROOT / "dataset" / "train"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("INSPECTING REAL MATCHES")
print("=" * 70)

print("\nLoading Source 1...")
s1 = pd.read_csv(TRAIN / "train_source1.tsv", sep="\t")

print("Loading Source 2...")
s2 = pd.read_csv(TRAIN / "train_source2.tsv", sep="\t")

print("Loading Source 3...")
s3 = pd.read_csv(TRAIN / "train_source3.tsv", sep="\t")

print("Loading Ground Truth...")
gt = pd.read_csv(TRAIN / "train_ground_truth.tsv", sep="\t")


# ============================================================
# CREATE LOOKUP TABLES
# ============================================================

s1_lookup = s1.set_index("entity_id")
s2_lookup = s2.set_index("entity_id")
s3_lookup = s3.set_index("entity_id")


# ============================================================
# SELECT EXAMPLES
# ============================================================

# Only use entities that actually have matches.
matched_gt = gt[
    gt["matched_entity_ids"].notna()
    & (gt["matched_entity_ids"].str.strip() != "")
]

# Take the first 10 examples.
examples = matched_gt.head(10)


# ============================================================
# DISPLAY MATCHES
# ============================================================

for _, gt_row in examples.iterrows():

    source1_id = gt_row["source1_entity_id"]

    matched_ids = [
        x.strip()
        for x in str(gt_row["matched_entity_ids"]).split(",")
        if x.strip()
    ]

    print("\n" + "=" * 70)
    print(f"SOURCE 1 ENTITY: {source1_id}")
    print("=" * 70)

    if source1_id in s1_lookup.index:

        row = s1_lookup.loc[source1_id]

        print("\nSOURCE 1")
        print(f"Name:    {row['business_name']}")
        print(f"Address: {row['business_address']}")
        print(f"Country: {row['country']}")

    print("\nTRUE MATCHES")

    for match_id in matched_ids:

        if match_id.startswith("S2-") and match_id in s2_lookup.index:

            row = s2_lookup.loc[match_id]

            print("\n[SOURCE 2]")
            print(f"ID:      {match_id}")
            print(f"Name:    {row['business_name']}")
            print(f"Address: {row['business_address']}")
            print(f"Country: {row['country']}")

        elif match_id.startswith("S3-") and match_id in s3_lookup.index:

            row = s3_lookup.loc[match_id]

            print("\n[SOURCE 3]")
            print(f"ID:      {match_id}")
            print(f"Name:    {row['business_name']}")
            print(f"Address: {row['business_address']}")
            print(f"Country: {row['country']}")

        else:
            print(f"\nCould not find record: {match_id}")


print("\n" + "=" * 70)
print("MATCH INSPECTION COMPLETE")
print("=" * 70)