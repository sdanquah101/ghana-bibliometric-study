"""
Shared configuration for Ghana Research Leadership Analysis.
All four analysis scripts import from this file.
"""
from pathlib import Path

# ── Paths ──
DATA_DIR = Path("filtered_biomed_engineering")
OUTPUT_DIR = Path("analysis_results")
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"

# ── Study Parameters ──
STUDY_START = 2000
STUDY_END = 2025
COVID_YEAR = 2020  # publications from 2020 onward = post-COVID

# ── Color Palette (use EXACTLY these in every chart) ──
COLORS = {
    "Ghanaian": "#2E7D32",
    "Dual-affiliated": "#F9A825",
    "Non-Ghanaian": "#1565C0",
    "total": "#616161",
}

# ── Marker Shapes (for line charts — colorblind accessibility) ──
MARKERS = {
    "Ghanaian": "o",        # circle
    "Dual-affiliated": "^",  # triangle
    "Non-Ghanaian": "s",     # square
}

# ── Chart Defaults ──
CHART_DPI = 300
CHART_FONT = "Arial"
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
SINGLE_FIG_SIZE = (8, 5)
MULTI_FIG_SIZE = (10, 8)

# ── Time Periods ──
TIME_PERIODS = {
    "2000–2005": (2000, 2005),
    "2006–2010": (2006, 2010),
    "2011–2015": (2011, 2015),
    "2016–2019": (2016, 2019),
    "2020–2025": (2020, 2025),
}

# ── Partner Bloc Country Codes ──
PARTNER_BLOCS = {
    "Western": {
        "US", "GB", "CA", "AU", "NZ",
        "DE", "FR", "NL", "SE", "DK", "NO", "CH", "BE", "IT", "ES",
        "AT", "FI", "IE", "PT", "LU", "GR", "CZ", "PL", "HU", "RO",
        "BG", "HR", "SK", "SI", "LT", "LV", "EE", "MT", "CY", "IS", "LI",
    },
    "East Asian": {
        "CN", "JP", "KR", "TW", "SG", "HK", "MY", "TH", "VN",
        "ID", "PH", "MM", "KH", "LA", "BN",
    },
    "South Asian": {
        "IN", "BD", "PK", "LK", "NP", "BT", "MV",
    },
    "Latin American": {
        "BR", "MX", "AR", "CO", "CL", "PE", "CU", "EC", "VE", "BO",
        "UY", "PY", "CR", "PA", "DO", "GT", "HN", "NI", "SV", "JM", "TT", "HT",
    },
    "African": {
        "ZA", "NG", "KE", "TZ", "ET", "UG", "CM", "SN", "BF", "MW",
        "RW", "BJ", "ML", "NE", "GN", "CI", "CD", "MZ", "ZW", "ZM",
        "MG", "AO", "GA", "SD", "SS", "SO", "TG", "SL", "LR", "ER",
        "DJ", "MR", "GM", "GW", "CV", "ST", "KM", "MU", "SC", "SZ",
        "LS", "BW", "NA", "TD", "CF", "CG", "GQ", "BI",
    },
    "MENA": {
        "EG", "SA", "IR", "IL", "TR", "MA", "TN", "AE", "QA",
        "JO", "LB", "IQ", "SY", "YE", "OM", "KW", "BH", "LY", "DZ", "PS",
    },
}

# Reverse lookup: country code -> bloc name
COUNTRY_TO_BLOC = {}
for bloc_name, codes in PARTNER_BLOCS.items():
    for code in codes:
        COUNTRY_TO_BLOC[code] = bloc_name

# ── Funder Classification Keywords ──
NORTHERN_FUNDERS_KEYWORDS = [
    "NIH", "National Institutes of Health", "Wellcome", "Gates",
    "USAID", "World Health Organization", "WHO", "DFID",
    "Medical Research Council", "European Commission", "EU",
    "DANIDA", "SIDA", "NORAD", "Global Fund", "EDCTP",
    "World Bank", "CDC", "Centers for Disease Control",
    "National Science Foundation", "UKRI", "BMGF",
    "Canadian Institutes", "CIHR", "NHMRC",
    "Fogarty", "PEPFAR", "UNICEF",
]

GHANAIAN_FUNDERS_KEYWORDS = [
    "Ghana", "GETFund", "KNUST", "University of Ghana",
    "Noguchi", "Ghana Health Service", "CSIR-Ghana",
]


def classify_funder(name):
    """Classify a funder name into Northern/Ghanaian/Other."""
    if pd.isna(name):
        return "Other/Unclassified"
    name_upper = str(name).upper()
    for kw in NORTHERN_FUNDERS_KEYWORDS:
        if kw.upper() in name_upper:
            return "International (Northern)"
    for kw in GHANAIAN_FUNDERS_KEYWORDS:
        if kw.upper() in name_upper:
            return "Ghanaian"
    return "Other/Unclassified"


def assign_partner_bloc(countries_str):
    """
    Given a pipe-separated string of country codes (from works.countries),
    determine the partner bloc.
    Returns (partner_bloc, partner_countries_list, is_western_collab, western_vs_nonwestern)
    """
    if pd.isna(countries_str):
        return "Unknown", [], False, "Unknown"

    codes = [c.strip() for c in str(countries_str).split("|") if c.strip()]
    non_gh = [c for c in codes if c != "GH"]

    if not non_gh:
        return "Unknown", [], False, "Unknown"

    blocs_present = set()
    has_western = False
    has_non_western = False

    for c in non_gh:
        bloc = COUNTRY_TO_BLOC.get(c, "Other")
        blocs_present.add(bloc)
        if bloc == "Western":
            has_western = True
        else:
            has_non_western = True

    # Assign bloc
    if len(blocs_present) == 1:
        partner_bloc = blocs_present.pop()
    else:
        partner_bloc = "Multi-bloc"

    # Western vs non-western
    if has_western and not has_non_western:
        western_vs_nonwestern = "Western"
    elif not has_western and has_non_western:
        western_vs_nonwestern = "Non-Western"
    elif has_western and has_non_western:
        western_vs_nonwestern = "Mixed"
    else:
        western_vs_nonwestern = "Unknown"

    is_western = has_western

    return partner_bloc, non_gh, is_western, western_vs_nonwestern


def assign_time_period(year):
    """Map a publication year to its time period."""
    for period_name, (start, end) in TIME_PERIODS.items():
        if start <= year <= end:
            return period_name
    return "Unknown"


# Need pandas for classify_funder
import pandas as pd
