# FairValue HTX

**Is this Houston home fairly priced, and can I actually afford it?**
UH CodeCoogs AI Hackathon 2026 · Track 2: Houston Housing Market Intelligence

A machine-learning model estimates every Greater Houston listing's fair price from its size, age, location and neighborhood,
**without ever seeing the asking price**. An interactive dashboard explains the whole project in 5 minutes and puts the model
to work for buyers: fair price, affordability by income, under- and overpriced homes, neighborhood comparison, and an AI assistant.

| Result (5-fold cross-validation, homes never seen in training) | |
|---|---|
| Median % error | **7.1%** (LightGBM alone 7.4%; linear baseline 18.7%) |
| R² on log price | 0.917 |
| Unseen ZIP codes | 12.2% |
| 80% price ranges that contain the true price | 80.8% |
| Zillow Zestimate, off-market, nationwide (reference) | 7.01% |

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/app.py          # dashboard at http://localhost:8501
```

The repository ships the processed data and trained model. To rebuild from scratch (about 5 minutes on a laptop CPU):

```bash
kaggle datasets download -d datadetective08/houston-housing-market-2024 -f "houston_housing market 2024_light.json" -p data/raw
kaggle datasets download -d mohamedalmansop/houston-housing-market-2024-clean-dataset -p data/raw --unzip
curl -L -o data/external/irs_soi_21zpallnoagi.csv https://www.irs.gov/pub/irs-soi/21zpallnoagi.csv
curl -L -o data/external/zhvi_zip.csv https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
curl -L -o data/external/zori_zip.csv https://files.zillowstatic.com/research/public_csvs/zori/Zip_zori_uc_sfrcondomfr_sm_month.csv
unzip -o "data/raw/houston_housing%20market%202024_light.json.zip" -d data/raw
.venv/bin/python src/build_dataset.py       # clean + join  -> data/processed/listings.parquet, reports/cleaning.json
cd src && ../.venv/bin/python train.py      # CV, intervals, SHAP, final model -> models/, reports/metrics.json
../.venv/bin/python diagnostics.py          # model comparison, overfitting checks, value drivers -> reports/diagnostics.json
../.venv/bin/python ensemble.py             # ensemble weighting experiment -> reports/ensemble.json
```

The **Ask FairValue** page calls Claude (`claude-opus-5-5`) and needs `ANTHROPIC_API_KEY` in the environment or in `.env`.
Everything else runs offline.

## Project layout

```
src/build_dataset.py   raw Zillow JSON + IRS + Zillow ZIP data -> clean modeling table
src/features.py        feature lists, nearest-neighbor "comps" (BallTree, haversine)
src/train.py           staged feature comparison, 5-fold grouped CV, conformal intervals, SHAP, final model
src/ensemble.py       4 base models + 5 weighting methods, nested CV, bootstrap CI (writes reports/ensemble.json)
src/ensemble_model.py  EnsembleRegressor: weighted LightGBM + CatBoost + XGBoost with exact combined SHAP
src/diagnostics.py     linear / random forest / LightGBM, train-vs-validation gap, learning curve, value drivers
app/app.py             Streamlit navigation
app/common.py          shared loaders, palette, chart and cost helpers
app/ai.py              Claude: plain-English search -> filters; plain-English home explanations
app/views/             one file per dashboard page
PITCH.md               5-minute talk track, demo script, judge Q&A
```

## Data

| Source | Use | Period |
|---|---|---|
| [Houston Housing Market 2024 (raw)](https://www.kaggle.com/datasets/datadetective08/houston-housing-market-2024): 25,948 Zillow listings | prices, home facts, location, Zestimate (benchmark only) | June 2024 snapshot |
| [Houston Housing Market 2024 (clean CSV)](https://www.kaggle.com/datasets/mohamedalmansop/houston-housing-market-2024-clean-dataset) | year built | same |
| [IRS SOI ZIP code data](https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi) | average household income per ZIP | tax year 2021 |
| [Zillow Research ZHVI / ZORI](https://www.zillow.com/research/data/) | ZIP typical home value, trend, rent | 2000 – Sept 2026 |

## Method in brief

1. **Cleaning.** Rebuilt from raw JSON: removed 3,400 builder floor-plan ads and 3,067 vacant lots, converted acres to sqft,
   and restored missing build years as missing. Applied sanity bounds on price, size and price per sqft. 19,430 homes remain.
2. **Features.** Structure; location; distance to 5 job hubs; median price per sqft of the 10 nearest other listings; ZIP income, value
   and rent trends; new-build and foreclosure flags. No leakage: the home's own price per sqft, Zestimate and tax value are never inputs.
3. **Model.** A weighted ensemble of CatBoost (64%), XGBoost (23%) and LightGBM (14%) on log price. Weights are non-negative, sum to 1,
   and minimize absolute log error. They are scored with nested cross-validation and beat LightGBM alone by 0.26 points (bootstrap 95% CI 0.17–0.37, 5/5 folds).
   `src/ensemble.py` compares equal, inverse-error, optimized and stacked weightings.
4. **Validation.** 5-fold CV grouped by exact coordinates. Comps are rebuilt inside each fold. A stress test holds out whole ZIP codes.
   A leakage demo shows that adding the Zestimate fakes a 1.2% error.
5. **Uncertainty.** Conformalized quantile regression, calibrated per price band and per tail, so each band has ~10% below and ~10% above its 80% range.
6. **Explanations.** Exact SHAP values (weighted across the three models), grouped into buyer-friendly drivers and converted to a dollar waterfall that sums exactly to the estimate.

## Limitations

List prices, not sale prices. A June 2024 snapshot. No interior condition, photos or flood-claim history. Ownership-cost
assumptions (rate, tax, insurance, HOA, flood) are editable defaults, not quotes. Not financial advice.
