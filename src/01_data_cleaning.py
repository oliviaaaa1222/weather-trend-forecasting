"""
01_data_cleaning.py
-------------------
Cleans and preprocesses the Global Weather Repository dataset.

Steps:
  1. Parse the `last_updated` timestamp and derive calendar features.
  2. Normalize messy country names (the raw file mixes languages, e.g.
     'Inde' -> 'India', '火鸡' -> 'Turkey') and attach a continent.
  3. Drop exact duplicate rows.
  4. Detect & treat outliers on key numeric columns using the IQR rule
     (capping / winsorizing rather than deleting, to preserve volume).
  5. Handle missing values (median impute numeric, mode impute categorical).
  6. Produce a Min-Max normalized copy of the modelling features.
  7. Write the cleaned dataset to data/weather_clean.csv.

Run:  python src/01_data_cleaning.py
"""
import numpy as np
import pandas as pd

from utils import load_raw, CLEAN_CSV, ensure_dirs, ROOT

# --- Map non-English / inconsistent country labels to canonical names ---
COUNTRY_FIX = {
    "Bélgica": "Belgium",
    "Estonie": "Estonia",
    "Inde": "India",
    "Jemen": "Yemen",
    "Komoren": "Comoros",
    "Kyrghyzstan": "Kyrgyzstan",
    "Letonia": "Latvia",
    "Malásia": "Malaysia",
    "Marrocos": "Morocco",
    "Mexique": "Mexico",
    "Polônia": "Poland",
    "Saudi Arabien": "Saudi Arabia",
    "Saint-Vincent-et-les-Grenadines": "Saint Vincent and the Grenadines",
    "Südkorea": "South Korea",
    "Turkménistan": "Turkmenistan",
    "USA United States of America": "United States of America",
    "Гватемала": "Guatemala",
    "Польша": "Poland",
    "Турция": "Turkey",
    "كولومبيا": "Colombia",
    "火鸡": "Turkey",
}

# Continent lookup for the geographic analyses. Defaults to "Other" if unseen.
CONTINENT = {
    # Africa
    **{c: "Africa" for c in [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
        "Congo", "Cote d'Ivoire", "Democratic Republic of Congo", "Djibouti",
        "Egypt", "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia",
        "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia",
        "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius",
        "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
        "Senegal", "Seychelles Islands", "Sierra Leone", "Somalia",
        "South Africa", "Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia",
        "Uganda", "Zambia", "Zimbabwe"]},
    # Europe
    **{c: "Europe" for c in [
        "Albania", "Andorra", "Austria", "Belarus", "Belgium",
        "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
        "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany",
        "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia",
        "Liechtenstein", "Lithuania", "Luxembourg", "Macedonia", "Malta",
        "Monaco", "Montenegro", "Netherlands", "Norway", "Poland", "Portugal",
        "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia",
        "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom",
        "Vatican City"]},
    # Asia
    **{c: "Asia" for c in [
        "Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh",
        "Bhutan", "Brunei Darussalam", "Cambodia", "China", "Georgia", "India",
        "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan",
        "Kuwait", "Kyrgyzstan", "Lao People's Democratic Republic", "Lebanon",
        "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal", "North Korea",
        "Oman", "Pakistan", "Philippines", "Qatar", "Saudi Arabia", "Singapore",
        "South Korea", "Sri Lanka", "Syria", "Tajikistan", "Thailand",
        "Timor-Leste", "Turkey", "Turkmenistan", "United Arab Emirates",
        "Uzbekistan", "Vietnam", "Yemen"]},
    # North America
    **{c: "North America" for c in [
        "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada",
        "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "El Salvador",
        "Grenada", "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico",
        "Nicaragua", "Panama", "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Trinidad and Tobago",
        "United States of America"]},
    # South America
    **{c: "South America" for c in [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"]},
    # Oceania
    **{c: "Oceania" for c in [
        "Australia", "Fiji Islands", "Kiribati", "Marshall Islands",
        "Micronesia", "New Zealand", "Palau", "Papua New Guinea", "Samoa",
        "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu"]},
}

# Numeric columns we treat for outliers / normalization.
NUMERIC_FEATURES = [
    "temperature_celsius", "wind_kph", "pressure_mb", "precip_mm", "humidity",
    "cloud", "feels_like_celsius", "visibility_km", "uv_index", "gust_kph",
    "air_quality_Carbon_Monoxide", "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5", "air_quality_PM10",
]


def cap_outliers_iqr(s: pd.Series, k: float = 1.5):
    """Winsorize a numeric series to [Q1-k*IQR, Q3+k*IQR]. Returns (series, n_capped)."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    # If the IQR is ~0 (a heavily concentrated column such as visibility,
    # which is 10 km for most rows), winsorizing would collapse the column to
    # a constant. Skip those to preserve genuine spread.
    if iqr <= 1e-9:
        return s, 0
    lo, hi = q1 - k * iqr, q3 + k * iqr
    capped = ((s < lo) | (s > hi)).sum()
    return s.clip(lo, hi), int(capped)


def main():
    ensure_dirs()
    df = load_raw()
    print(f"Loaded raw data: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # 1. Timestamp + calendar features
    df = df.dropna(subset=["last_updated"]).copy()
    df["date"] = df["last_updated"].dt.date
    df["year"] = df["last_updated"].dt.year
    df["month"] = df["last_updated"].dt.month
    df["dayofyear"] = df["last_updated"].dt.dayofyear

    # 2. Normalize country names + continent
    df["country"] = df["country"].replace(COUNTRY_FIX)
    df["continent"] = df["country"].map(CONTINENT).fillna("Other")
    n_other = int((df["continent"] == "Other").sum())
    print(f"Country names normalized; rows with unmapped continent: {n_other}")

    # 3. Duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} exact duplicate rows")

    # 4. Outlier treatment (IQR winsorization)
    total_capped = 0
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col], n = cap_outliers_iqr(df[col])
            total_capped += n
    print(f"Winsorized {total_capped:,} outlier values across {len(NUMERIC_FEATURES)} numeric columns")

    # 5. Missing values
    n_missing = int(df.isnull().sum().sum())
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype.kind in "biufc":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    print(f"Imputed {n_missing:,} missing values")

    # 6. Min-Max normalized copies of modelling features (suffix _norm)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            rng = (mx - mn) or 1.0
            df[f"{col}_norm"] = (df[col] - mn) / rng

    # 7. Save
    df.to_csv(CLEAN_CSV, index=False)
    print(f"\nClean dataset written -> {CLEAN_CSV.relative_to(ROOT)}")
    print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Countries: {df.country.nunique()} | Continents: {sorted(df.continent.unique())}")


if __name__ == "__main__":
    main()
