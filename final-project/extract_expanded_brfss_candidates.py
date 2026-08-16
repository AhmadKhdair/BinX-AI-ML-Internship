from pathlib import Path

import pyreadstat

XPT_PATH = Path(
    r"C:\Users\HP\Downloads\LLCP2021XPT\LLCP2021.XPT"
)

OUTPUT_PATH = Path(
    "data/processed/experimental/"
    "brfss2021_expanded_raw_candidates.csv.gz"
)

BASE_COLUMNS = [
    "SEXVAR",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "PRIMINSR",
    "PERSDOC3",
    "MEDCOST1",
    "CHECKUP1",
    "EXERANY2",
    "BPHIGH6",
    "CHOLCHK3",
    "TOLDHI3",
    "CVDSTRK3",
    "ASTHMA3",
    "CHCCOPD3",
    "ADDEPEV3",
    "CHCKDNY2",
    "DIABETE4",
    "HAVARTH5",
    "MARITAL",
    "EMPLOY1",
    "DIFFWALK",
    "_RFCHOL3",
    "_MICHD",
    "_RACEGR3",
    "_AGEG5YR",
    "_AGE80",
    "_BMI5",
    "_BMI5CAT",
    "_EDUCAG",
    "_INCOMG1",
    "_SMOKER3",
    "DRNKANY5",
    "_RFBING5",
    "_RFDRHV7",
    "_FRTLT1A",
    "_VEGLT1A",
]

# These are EXPERIMENTAL candidates only.
# They are extracted raw so their codebook semantics,
# structural missingness, leakage/proxy risk, and CV value
# can be audited before any feature is accepted.
EXTRA_CANDIDATES = [
    # Health / treatment / disease-duration candidates
    "POORHLTH",
    "BPMEDS",
    "CHOLMED3",
    "ASTHNOW",
    "CHCSCNCR",
    "CHCOCNCR",
    "DIABAGE3",

    # Demographic / socioeconomic candidates
    "EDUCA",
    "RENTHOM1",
    "VETERAN3",
    "CHILDREN",
    "INCOME3",

    # Functional-status candidates
    "DEAF",
    "BLIND",
    "DECIDE",
    "DIFFDRES",
    "DIFFALON",

    # More detailed tobacco representation
    "SMOKE100",
    "SMOKDAY2",
    "USENOW3",
    "ECIGNOW1",

    # More detailed alcohol representation
    "ALCDAY5",
    "AVEDRNK3",
    "DRNK3GE5",
    "MAXDRNKS",

    # Preventive-care candidates
    "FLUSHOT7",
    "PNEUVAC4",

    # More detailed diet representation
    "FRUIT2",
    "FRUITJU2",
    "FVGREEN1",
    "FRENCHF1",
    "POTATOE1",
    "VEGETAB2",

    # Diabetes-related module candidates
    "PDIABTST",
    "PREDIAB1",

    # Optional BP / sodium modules: extracted only so
    # structural missingness can be measured before use.
    "HOMBPCHK",
    "HOMRGCHK",
    "WTCHSALT",
    "DRADVISE",

    # Geographic sensitivity candidates
    "_STATE",
    "_METSTAT",
    "_URBSTAT",

    # Fine-grained calculated anthropometric variables
    "HTIN4",
    "HTM4",
    "WTKG3",

    # Alternative/fine-grained calculated representations
    "_RFHLTH",
    "_PHYS14D",
    "_MENT14D",
    "_TOTINDA",
    "_RFHYPE6",
    "_CHOLCH3",
    "_RFSMOK3",
    "_CURECI1",
    "DROCDY3_",
    "_DRNKWK1",
    "_FRUTSU1",
    "_VEGESU1",
]

DIRECT_TARGET_LEAKAGE = {
    "CVDINFR4",
    "CVDCRHD4",
}

usecols = list(dict.fromkeys(
    BASE_COLUMNS + EXTRA_CANDIDATES
))

assert DIRECT_TARGET_LEAKAGE.isdisjoint(usecols)
assert XPT_PATH.exists(), f"XPT not found: {XPT_PATH}"

print("Reading selected raw columns from:")
print(XPT_PATH)
print("Number of selected columns:", len(usecols))

raw_df, _ = pyreadstat.read_xport(
    XPT_PATH,
    usecols=usecols,
    disable_datetime_conversion=True,
)

assert len(raw_df) == 438_693
assert "_MICHD" in raw_df.columns
assert DIRECT_TARGET_LEAKAGE.isdisjoint(raw_df.columns)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

raw_df.to_csv(
    OUTPUT_PATH,
    index=False,
    compression="gzip",
)

print("\nExtraction complete.")
print("Shape:", raw_df.shape)
print("Saved:", OUTPUT_PATH.resolve())
print(
    "File size:",
    f"{OUTPUT_PATH.stat().st_size / 1024**2:.1f} MB",
)
print(
    "\nIMPORTANT: this is a RAW experimental extraction. "
    "Do not model it before codebook-aware cleaning."
)
