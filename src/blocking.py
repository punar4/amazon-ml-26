import pandas as pd
import re
from pathlib import Path
from collections import defaultdict, Counter

from normalization import normalize_name, normalize_address


ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "dataset" / "train"

S1_ROWS = 300
NOISY_ROWS = 100_000


print("=" * 70)
print("FAST BLOCKING DIAGNOSTIC")
print("=" * 70)


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

print("\nLoading Source 1...")
s1 = pd.read_csv(
    TRAIN / "train_source1.tsv",
    sep="\t",
    nrows=S1_ROWS
)

print("Loading Source 2 sample...")
s2 = pd.read_csv(
    TRAIN / "train_source2.tsv",
    sep="\t",
    nrows=NOISY_ROWS
)

print("Loading Source 3 sample...")
s3 = pd.read_csv(
    TRAIN / "train_source3.tsv",
    sep="\t",
    nrows=NOISY_ROWS
)

print("Loading Ground Truth...")
gt = pd.read_csv(
    TRAIN / "train_ground_truth.tsv",
    sep="\t"
)


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

print("\nNormalizing...")

for df in (s1, s2, s3):
    df["name_norm"] = df["business_name"].map(normalize_name)
    df["address_norm"] = df["business_address"].map(
        normalize_address
    )
    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def name_tokens(text):
    return {
        x for x in str(text).split()
        if len(x) >= 4
    }


def address_tokens(text):
    return {
        x for x in str(text).split()
        if len(x) >= 4 and not x.isdigit()
    }


def address_numbers(text):
    return set(
        re.findall(r"\d+", str(text))
    )


# ---------------------------------------------------------
# TOKEN FREQUENCIES
# ---------------------------------------------------------

print("Calculating token frequencies...")

frequency = Counter()

for text in pd.concat(
    [s2["name_norm"], s3["name_norm"]],
    ignore_index=True
):
    frequency.update(name_tokens(text))


def rare_tokens(text):
    return {
        token
        for token in name_tokens(text)
        if frequency[token] <= 50
    }


# ---------------------------------------------------------
# INDEX BUILDER
# ---------------------------------------------------------

def build_index(rows, extractor):

    index = defaultdict(set)

    for row in rows:

        country = row.country_norm

        for value in extractor(row):

            index[(country, value)].add(
                row.entity_id
            )

    return index


# We use itertuples because it is much faster than iterrows.

print("\nBuilding lightweight indexes...")

s2_rows = list(
    s2[
        [
            "entity_id",
            "name_norm",
            "address_norm",
            "country_norm"
        ]
    ].itertuples(index=False)
)

s3_rows = list(
    s3[
        [
            "entity_id",
            "name_norm",
            "address_norm",
            "country_norm"
        ]
    ].itertuples(index=False)
)


def name_extractor(row):
    return name_tokens(row.name_norm)


def rare_name_extractor(row):
    return rare_tokens(row.name_norm)


def number_extractor(row):
    return address_numbers(row.address_norm)


def address_token_extractor(row):
    return address_tokens(row.address_norm)


print("  exact name...")

s2_exact = defaultdict(set)
s3_exact = defaultdict(set)

for row in s2_rows:
    if row.name_norm:
        s2_exact[
            (row.country_norm, row.name_norm)
        ].add(row.entity_id)

for row in s3_rows:
    if row.name_norm:
        s3_exact[
            (row.country_norm, row.name_norm)
        ].add(row.entity_id)


print("  address numbers...")

s2_numbers = build_index(
    s2_rows,
    number_extractor
)

s3_numbers = build_index(
    s3_rows,
    number_extractor
)


print("  name tokens...")

s2_name_tokens = build_index(
    s2_rows,
    name_extractor
)

s3_name_tokens = build_index(
    s3_rows,
    name_extractor
)


print("  rare name tokens...")

s2_rare = build_index(
    s2_rows,
    rare_name_extractor
)

s3_rare = build_index(
    s3_rows,
    rare_name_extractor
)


print("  address tokens...")

s2_address_tokens = build_index(
    s2_rows,
    address_token_extractor
)

s3_address_tokens = build_index(
    s3_rows,
    address_token_extractor
)


# ---------------------------------------------------------
# GROUND TRUTH
# ---------------------------------------------------------

gt_lookup = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].fillna("")
    )
)

s2_ids = set(s2["entity_id"])
s3_ids = set(s3["entity_id"])

print("\nIndexes ready.")


# ---------------------------------------------------------
# RULES
# ---------------------------------------------------------

def candidates_for_rule(rule, row):

    country = row.country_norm
    name = row.name_norm
    address = row.address_norm

    candidates = set()

    # ---------------------------------------------
    # 1. EXACT NAME
    # ---------------------------------------------

    if rule == "exact_name":

        key = (country, name)

        candidates |= s2_exact.get(key, set())
        candidates |= s3_exact.get(key, set())


    # ---------------------------------------------
    # 2. ADDRESS NUMBER
    # ---------------------------------------------

    elif rule == "address_number":

        for number in address_numbers(address):

            key = (country, number)

            candidates |= s2_numbers.get(
                key,
                set()
            )

            candidates |= s3_numbers.get(
                key,
                set()
            )


    # ---------------------------------------------
    # 3. NAME TOKEN
    # ---------------------------------------------

    elif rule == "name_token":

        for token in name_tokens(name):

            key = (country, token)

            candidates |= s2_name_tokens.get(
                key,
                set()
            )

            candidates |= s3_name_tokens.get(
                key,
                set()
            )


    # ---------------------------------------------
    # 4. RARE NAME TOKEN
    # ---------------------------------------------

    elif rule == "rare_name_token":

        for token in rare_tokens(name):

            key = (country, token)

            candidates |= s2_rare.get(
                key,
                set()
            )

            candidates |= s3_rare.get(
                key,
                set()
            )


    # ---------------------------------------------
    # 5. ADDRESS NUMBER + TOKEN
    # ---------------------------------------------

    elif rule == "address_number_and_address_token":

        numbers = address_numbers(address)
        tokens = address_tokens(address)

        for number in numbers:

            number_key = (country, number)

            number_s2 = s2_numbers.get(
                number_key,
                set()
            )

            number_s3 = s3_numbers.get(
                number_key,
                set()
            )

            for token in tokens:

                token_key = (country, token)

                token_s2 = s2_address_tokens.get(
                    token_key,
                    set()
                )

                token_s3 = s3_address_tokens.get(
                    token_key,
                    set()
                )

                candidates |= (
                    number_s2 & token_s2
                )

                candidates |= (
                    number_s3 & token_s3
                )

    return candidates


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

rules = [
    "exact_name",
    "address_number",
    "name_token",
    "rare_name_token",
    "address_number_and_address_token",
]


print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)


for rule in rules:

    print(f"\n{rule}")
    print("-" * 70)

    total_candidates = 0
    recovered = 0
    available = 0

    for row in s1[
        [
            "entity_id",
            "name_norm",
            "address_norm",
            "country_norm"
        ]
    ].itertuples(index=False):

        candidates = candidates_for_rule(
            rule,
            row
        )

        total_candidates += len(candidates)

        gt_value = gt_lookup.get(
            row.entity_id,
            ""
        )

        true_ids = {
            x.strip()
            for x in str(gt_value).split(",")
            if x.strip()
        }

        # Only positives actually contained
        # in our 100k samples.
        loaded_true = {
            x for x in true_ids
            if (
                (
                    x.startswith("S2-")
                    and x in s2_ids
                )
                or
                (
                    x.startswith("S3-")
                    and x in s3_ids
                )
            )
        }

        available += len(loaded_true)

        recovered += len(
            loaded_true & candidates
        )

    avg_candidates = (
        total_candidates / len(s1)
    )

    recall = (
        recovered / available * 100
        if available
        else 0
    )

    print(
        f"Average candidates: {avg_candidates:,.2f}"
    )

    print(
        f"Loaded true matches: {available}"
    )

    print(
        f"Recovered true matches: {recovered}"
    )

    print(
        f"Recall: {recall:.2f}%"
    )


print("\n" + "=" * 70)
print("FAST DIAGNOSTIC COMPLETE")
print("=" * 70)