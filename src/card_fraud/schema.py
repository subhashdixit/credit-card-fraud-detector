"""Shared input contract for training, API, form and CSV scoring."""

import math
import pandas as pd

# These fields are present in both source tables and the supplied unseen workbook.
FEATURES = {
    "V1": ("Anonymized PCA component V1", -1000000, 1000000),
    "V2": ("Anonymized PCA component V2", -1000000, 1000000),
    "V3": ("Anonymized PCA component V3", -1000000, 1000000),
    "V4": ("Anonymized PCA component V4", -1000000, 1000000),
    "V5": ("Anonymized PCA component V5", -1000000, 1000000),
    "V6": ("Anonymized PCA component V6", -1000000, 1000000),
    "V7": ("Anonymized PCA component V7", -1000000, 1000000),
    "V8": ("Anonymized PCA component V8", -1000000, 1000000),
    "V9": ("Anonymized PCA component V9", -1000000, 1000000),
    "V10": ("Anonymized PCA component V10", -1000000, 1000000),
    "V11": ("Anonymized PCA component V11", -1000000, 1000000),
    "V12": ("Anonymized PCA component V12", -1000000, 1000000),
    "V13": ("Anonymized PCA component V13", -1000000, 1000000),
    "V14": ("Anonymized PCA component V14", -1000000, 1000000),
    "V15": ("Anonymized PCA component V15", -1000000, 1000000),
    "V16": ("Anonymized PCA component V16", -1000000, 1000000),
    "V17": ("Anonymized PCA component V17", -1000000, 1000000),
    "V18": ("Anonymized PCA component V18", -1000000, 1000000),
    "V19": ("Anonymized PCA component V19", -1000000, 1000000),
    "V20": ("Anonymized PCA component V20", -1000000, 1000000),
    "V21": ("Anonymized PCA component V21", -1000000, 1000000),
    "V22": ("Anonymized PCA component V22", -1000000, 1000000),
    "V23": ("Anonymized PCA component V23", -1000000, 1000000),
    "V24": ("Anonymized PCA component V24", -1000000, 1000000),
    "V25": ("Anonymized PCA component V25", -1000000, 1000000),
    "V26": ("Anonymized PCA component V26", -1000000, 1000000),
    "V27": ("Anonymized PCA component V27", -1000000, 1000000),
    "V28": ("Anonymized PCA component V28", -1000000, 1000000),
    "Amount": ("Transaction amount (source currency)", 0, 1000000000),
}
INTEGER_FIELDS = []
LABELS = {"0": "Non-fraud source label", "1": "Fraud source label"}


def validate_records(records):
    """Validate ordered numeric features; explicit nulls mean missing values."""
    if not isinstance(records, list) or not 1 <= len(records) <= 500:
        raise ValueError("Provide a list of 1 to 500 records.")
    cleaned = []
    for row_number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Row {row_number}: each record must be an object.")
        missing = set(FEATURES) - set(record)
        if missing:
            raise ValueError(
                f"Row {row_number}: missing columns: {', '.join(sorted(missing))}"
            )
        row = {}
        for field, (_, lower, upper) in FEATURES.items():
            value = record[field]
            if value is None or value == "":
                row[field] = float("nan")
                continue
            try:
                if isinstance(value, bool):
                    raise ValueError
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {row_number}: {field} must be numeric.") from exc
            if not math.isfinite(number) or not lower <= number <= upper:
                raise ValueError(
                    f"Row {row_number}: {field} must be between {lower} and {upper}."
                )
            if field in INTEGER_FIELDS and number != int(number):
                raise ValueError(f"Row {row_number}: {field} must be a whole number.")
            row[field] = number
        if all(math.isnan(value) for value in row.values()):
            raise ValueError(f"Row {row_number}: provide at least one observed value.")
        cleaned.append(row)
    return pd.DataFrame(cleaned, columns=list(FEATURES))
