import re
import unicodedata


def normalize_text(value):
    """
    Basic normalization for business names and addresses.

    Steps:
    1. Handle missing values.
    2. Convert to string.
    3. Convert Unicode characters to a consistent form.
    4. Convert to lowercase.
    5. Remove accents where possible.
    6. Replace punctuation with spaces.
    7. Collapse repeated whitespace.
    """

    if value is None:
        return ""

    text = str(value)

    if text.lower() == "nan":
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Remove accent marks
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    # Lowercase
    text = text.lower()

    # Replace punctuation/symbols with spaces
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(value):
    """
    Normalize a business name.
    """
    return normalize_text(value)


def normalize_address(value):
    """
    Normalize a business address.
    """
    return normalize_text(value)


if __name__ == "__main__":

    # Small examples from the real data we inspected
    name_examples = [
        "Payne Énterprises",
        "PAYNE-ENRTPRMISES",
        "Hendricks and  Flowers Inc",
        "Hendricks and Inc Flowers",
    ]

    address_examples = [
        "3315 FREMONT ST, PEORIA, IL",
        "3315 Fremont Street, Peoria, Illinois",
        "85 Wanye Avenue, Ticonderoga Townshiip, New York",
        "6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI",
    ]

    print("=" * 70)
    print("NORMALIZATION TEST")
    print("=" * 70)

    print("\nNAME EXAMPLES:")

    for value in name_examples:
        print(f"\nOriginal:   {value}")
        print(f"Normalized: {normalize_name(value)}")

    print("\n" + "=" * 70)
    print("ADDRESS EXAMPLES")
    print("=" * 70)

    for value in address_examples:
        print(f"\nOriginal:   {value}")
        print(f"Normalized: {normalize_address(value)}")

    print("\n" + "=" * 70)
    print("NORMALIZATION TEST COMPLETE")
    print("=" * 70)