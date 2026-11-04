"""Shared property vocabulary for ProtParams sourcing and evaluation."""

PROTPARAM_COLUMNS = (
    "length",
    "mass",
    *(f"percent_{amino_acid}" for amino_acid in "acdefghiklmnpqrstvwy"),
    "fraction_helix_aas",
    "fraction_turn_aas",
    "fraction_sheet_aas",
    "instability_index",
    "gravy",
    "isoelectric_point",
    "charge_at_ph4_7",
    "charge_at_ph7_2",
    "charge_at_ph8",
)
