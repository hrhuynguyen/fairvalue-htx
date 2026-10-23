# XMarket: Fair Prices and Affordability for Houston Homes

**UH CodeCoogs AI Hackathon 2026 · Track 2: Houston Housing Market Intelligence**

**Live demo:** [DEMO LINK] · **Code:** [REPO LINK]

## Overview

A first-time buyer in Houston can't easily tell whether an asking price is fair. On active listings, Zillow's estimate
stays within 3.5% of the asking price, so it largely echoes the seller. The sticker price also hides Texas-sized
property taxes and insurance.

**XMarket** estimates a fair price for every Greater Houston home **without ever seeing its asking price**, then builds
practical tools on top: fair-price checks, affordability by income, neighborhood recommendations, and an
under/overpriced-listing finder.

| Result (out-of-fold, 19,430 homes) | Value |
|---|---|
| Median absolute % error (MdAPE) | **7.12%** (linear baseline 18.7%) |
| R² on log price | **0.917** |
| Homes priced within ±10% | 62.6% |
| ZIP codes never seen in training | 12.2% MdAPE |
| 80% price intervals: held-out coverage | 80.8% |
| Reference: Zillow Zestimate, off-market (national) | 7.01% MdAPE |

---

## Part I: Data Science

### 1. Data

| Source | Use | Coverage |
|---|---|---|
| Houston Housing Market 2024, raw Zillow scrape (Kaggle) | Price, home facts, coordinates, listing status, Zestimate (benchmark only) | 25,948 listings, June 2024 |
| Houston Housing Market 2024, clean CSV (Kaggle) | Year built | Same listings |
| IRS Statistics of Income by ZIP | Average household income per ZIP | Tax year 2021 |
| Zillow Research ZHVI / ZORI | ZIP typical home value, growth, rent | Monthly, 2000 – Sept 2026 |
| US Census ZCTA boundaries | ZIP maps | 159 ZIP codes |

After cleaning: **19,430 homes, 159 ZIP codes and 62 cities** mainly across Harris,
Fort Bend, Montgomery, Brazoria and Galveston counties. The listings carry location but no income data, so income and market context were joined by ZIP code.

### 2. Exploratory analysis

- **Skewed target.** Median $382k vs average $532k; skewness 7.8, or 0.9 after a log transform. We model **log price**.
- **Size dominates.** Living area has the strongest rank correlation with price (ρ = 0.79). Bathrooms (0.66) beat
  bedrooms (0.48). By bedroom count, price rises from $155k to $870k, but $/sqft stays at $169–$206: buyers pay for space.
- **Location is non-linear.** $/sqft is $252 within 5 miles of downtown, falls to $153 at 10–15 miles, then rises in newer,
  larger suburbs. This U-shape motivated tree models.
- **New construction** is 28% of listings, at a 15% $/sqft premium.
- **Market over time.** Typical values rose 128% from 2012 to the August 2022 peak and are down 4.8% since our snapshot.
  Cheaper ZIP codes grew fastest (77011 +263%); the condo-heavy core grew slowest (77056 +47%).
- **Redundant signals.** ZIP income and ZIP home value correlate at 0.92.
- **Affordability.** At 28% of a $75k income, only **6% of listings** are affordable. Tax and insurance are 37% of the
  monthly payment.

### 3. Data cleaning

The provided "clean" CSV hid four problems: 3,400 builder floor-plan ads, 3,067 vacant land lots, 5,670 lot sizes recorded
in acres, and 3,057 placeholder build years (2001.5). We rebuilt the dataset from the raw JSON and converted acres to sqft.
We removed 6,518 rows that were not homes for sale or were physically implausible (price, size, $/sqft bounds):
**25,948 → 19,430**. Single bad fields were blanked rather than dropping the home. Missing values stay missing, and the
tree models handle them natively.

### 4. Feature engineering

**30 features:** 14 raw (structure, location, listing flags) and **16 engineered**:

- **Ratios:** bedrooms per bathroom, sqft per bedroom.
- **Commute:** Haversine distance to 5 job hubs (Downtown, Texas Medical Center, Galleria, Energy Corridor, UH).
- **Nearby comps:** median price and $/sqft of the 10 nearest *other* listings (BallTree), rebuilt inside every CV fold
  from training rows only. They correlate 0.71 with a home's own $/sqft.
- **Neighborhood context:** ZIP income (IRS), typical value, 1- and 5-year growth, and rent (Zillow).

Top drivers by mean |SHAP|: living area 27%, lot size 10%, year built 9%, and the comps features 16% combined.

### 5. Metric

The primary metric is the **median absolute percentage error (MdAPE)**: "half of our estimates are within X% of the price".
It treats a $150k and a $1.5M home on the same relative scale and is robust to a few extreme homes. In dollars, luxury
homes look 14× harder to price; in percent, about 1.6×. It is also the metric Zillow publishes. Supporting metrics:
within ±10%, median dollar error, and R² on log price.

### 6. Modeling

| Model (out-of-fold, identical folds) | MdAPE | R² |
|---|---|---|
| Ridge regression (baseline) | 18.74% | 0.664 |
| Random forest | 7.78% | 0.887 |
| LightGBM | 7.39% | 0.910 |
| XGBoost | 7.25% | 0.908 |
| CatBoost | 7.23% | 0.916 |
| **Ensemble: CatBoost 0.64 · XGBoost 0.23 · LightGBM 0.14** | **7.12%** | **0.917** |

- **Ensembling.** Weights are non-negative, sum to 1, and minimize absolute log error over out-of-fold predictions. Five
  weighting schemes were compared under **nested cross-validation**. The random forest received weight 0. The gain over
  LightGBM is 0.26 points (paired bootstrap 95% CI 0.17–0.37, better on 5/5 folds), and over CatBoost 0.11 points (CI 0.04–0.17).
- **Ablation (single LightGBM).** Structure only 12.96%; + location 7.91%; + commute 7.70%; + comps 7.44%. ZIP context
  is neutral in random CV but improves the spatial hold-out (14.5% → 13.4%).
- **Explainability.** Each booster's exact TreeSHAP values are combined with the ensemble weights, so every prediction
  decomposes additively into dollar drivers.
- **Compute.** CPU only: Apple M2 Pro (10 cores), 16 GB RAM, about 5 minutes end to end. No GPU and no hyperparameter
  search: conservative settings, with the time spent on validation instead.
- **What didn't help.** Ridge stacking (no gain, negative weights), squared-error weight optimization, and the random
  forest. Using the Zestimate as a feature gives 1.2% MdAPE and R² 0.995, which is target leakage, so it is used only as a benchmark.

### 7. Validation

- **Protocol.** 5-fold `GroupKFold` grouped by exact coordinates, so co-located listings never straddle folds. Every
  reported number is out-of-fold. Fold-to-fold MdAPE is 6.76–7.48%.
- **Leakage controls.** Fold-local comps (self excluded); target proxies (Zestimate, rent Zestimate, tax value) and
  seller-behavior fields (days on market, price cuts) excluded; ensemble weights selected with nested CV.
- **Overfitting.** Validation loss keeps falling through 800 trees with no divergence, and the train/validation gap is
  stable across folds.
- **Generalization.** Holding out entire ZIP codes gives 12.2% MdAPE (R² 0.844).
- **Uncertainty.** Conformalized quantile regression, calibrated per price decile and per tail: 80.8% held-out coverage,
  with each tail at 6.6–11.8% per decile (target 10%).
- **Benchmark.** Zillow reports 1.83% MdAPE on-market, when it sees the asking price, and 7.01% off-market. Ours is 7.12%
  without the asking price. Caveat: Zillow's figures are national and measured against sale prices; ours are Houston list prices.
- **External check.** 49% of homes we flag as underpriced already had a price cut, vs 31% of those flagged overpriced,
  although the model never sees price-cut history.

---

## Part II: Applications

**Application**

1. **Price any home.** Enter a ZIP code and home details to get a fair value, an 80% range, a dollar breakdown of what
   drives the price, and similar homes listed nearby.
2. **Affordability by income.** Enter a household income to see the maximum affordable price (all-in monthly cost ≤ 28% of
   income), what share of the market fits, where the money goes each month, and a map of homes that fit.
3. **Where should I live?** Enter income and workplace; every ZIP code is scored on affordability, commute, neighborhood
   strength and bargains (weights adjustable), giving the top 3 areas and the best-value homes in #1.
4. **Under- & overpriced homes.** Filter by monthly budget, bedrooms and commute to find listings priced below (or above)
   their fair range, on a map and a ranked table.
5. **Home report.** One-listing due diligence: verdict and price drivers, true monthly cost (mortgage, tax, insurance,
   PMI, HOA, flood), and the ZIP's price trend since 2012.
6. **What adds value.** A renovation planner that re-prices *your* home for each upgrade. For example, +500 sqft adds
   about $57k to a typical 77004 home, while carving out an extra bedroom lowers its value.
7. **Neighborhood comparison.** Compare up to 4 ZIP codes side by side on price, $/sqft, income needed, rent and momentum,
   with value-trend lines and a ranking of all ZIP codes.

**Creative extras**

- **Ask XMarket (AI).** Plain-English search ("3-bedroom under $1,800/month near the Medical Center") translated into filters
  by Claude, plus plain-English explanations of any home. All prices come from our model.
- **Beat the model.** Guess five asking prices from photos and specs, and compare your error with the model's.
- **Rent or buy?** Shows the year in which buying a given home overtakes renting it, under adjustable assumptions.

---

## Limitations and next steps

List prices are not sale prices; the data is a June 2024 snapshot; and there is no interior-condition, flood-history or
school data. Distances are straight-line. Next steps: train on HCAD sale records, add FEMA flood claims and school ratings,
and retrain monthly on fresh listings.

**Stack:** Python, pandas, scikit-learn, LightGBM, CatBoost, XGBoost, SHAP, Streamlit, Plotly, pydeck, Anthropic Claude API.
