from typing import Dict, Any, List, Tuple
import pandas as pd

INPUT_COLUMNS: List[str] = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

OUTPUT_COLUMN: str = "quality"

FEATURE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "fixed_acidity": {
        "label": "Fixed Acidity (g/dm³)",
        "min": 3.0,
        "max": 16.0,
        "default": 7.4,
        "step": 0.1,
        "help": "Most acids involved with wine, tartaric acid concentration.",
    },
    "volatile_acidity": {
        "label": "Volatile Acidity (g/dm³)",
        "min": 0.05,
        "max": 2.0,
        "default": 0.70,
        "step": 0.01,
        "help": "Amount of acetic acid in wine, can lead to unpleasant vinegar taste at high levels.",
    },
    "citric_acid": {
        "label": "Citric Acid (g/dm³)",
        "min": 0.0,
        "max": 1.5,
        "default": 0.00,
        "step": 0.01,
        "help": "Adds 'freshness' and flavor to wines in small quantities.",
    },
    "residual_sugar": {
        "label": "Residual Sugar (g/dm³)",
        "min": 0.1,
        "max": 25.0,
        "default": 1.9,
        "step": 0.1,
        "help": "Amount of sugar remaining after fermentation stops.",
    },
    "chlorides": {
        "label": "Chlorides (g/dm³)",
        "min": 0.005,
        "max": 0.700,
        "default": 0.076,
        "step": 0.001,
        "help": "Amount of salt in the wine.",
    },
    "free_sulfur_dioxide": {
        "label": "Free Sulfur Dioxide (mg/dm³)",
        "min": 1.0,
        "max": 100.0,
        "default": 11.0,
        "step": 1.0,
        "help": "Free form of SO2 prevents microbial growth and oxidation.",
    },
    "total_sulfur_dioxide": {
        "label": "Total Sulfur Dioxide (mg/dm³)",
        "min": 5.0,
        "max": 350.0,
        "default": 34.0,
        "step": 1.0,
        "help": "Total amount of SO2 present in the wine.",
    },
    "density": {
        "label": "Density (g/cm³)",
        "min": 0.9800,
        "max": 1.0100,
        "default": 0.9978,
        "step": 0.0001,
        "format": "%.4f",
        "help": "Density close to water depending on alcohol and sugar content.",
    },
    "pH": {
        "label": "pH Level",
        "min": 2.5,
        "max": 4.5,
        "default": 3.51,
        "step": 0.01,
        "help": "Describes how acidic or basic the wine is (0 = very acidic, 14 = very basic).",
    },
    "sulphates": {
        "label": "Sulphates (g/dm³)",
        "min": 0.1,
        "max": 2.5,
        "default": 0.56,
        "step": 0.01,
        "help": "Wine additive that contributes to SO2 levels, acts as antimicrobial and antioxidant.",
    },
    "alcohol": {
        "label": "Alcohol (% vol)",
        "min": 6.0,
        "max": 16.0,
        "default": 9.4,
        "step": 0.1,
        "help": "Percent alcohol content of the wine.",
    },
}

PRESET_SAMPLES: Dict[str, Dict[str, float]] = {
    "Sample 1 (Typical Medium Red, ~5.2)": {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    },
    "Sample 2 (High Alcohol / Balanced, ~6.1)": {
        "fixed_acidity": 6.8,
        "volatile_acidity": 0.5,
        "citric_acid": 0.12,
        "residual_sugar": 2.5,
        "chlorides": 0.065,
        "free_sulfur_dioxide": 18.0,
        "total_sulfur_dioxide": 45.0,
        "density": 0.9955,
        "pH": 3.42,
        "sulphates": 0.62,
        "alcohol": 10.5,
    },
    "Sample 3 (High Sulphates & Alcohol, ~5.8)": {
        "fixed_acidity": 8.2,
        "volatile_acidity": 0.6,
        "citric_acid": 0.08,
        "residual_sugar": 2.2,
        "chlorides": 0.09,
        "free_sulfur_dioxide": 15.0,
        "total_sulfur_dioxide": 50.0,
        "density": 0.9965,
        "pH": 3.3,
        "sulphates": 0.7,
        "alcohol": 11.0,
    },
}


def get_quality_badge(score: float) -> Tuple[str, str]:
    """Return descriptive rating label and color given a numeric quality score."""
    if score >= 7.0:
        return "Exceptional / Premium", "green"
    elif score >= 6.0:
        return "Good Quality", "blue"
    elif score >= 5.0:
        return "Average / Standard", "orange"
    else:
        return "Poor Quality", "red"


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that an uploaded DataFrame has all required columns."""
    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing
