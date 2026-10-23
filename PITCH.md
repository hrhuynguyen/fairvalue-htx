# XMarket: 5-minute pitch and demo script

**Setup before judging:** run `.venv/bin/streamlit run app/app.py`, open http://localhost:8501 in full-screen Chrome at 100% zoom,
and close other tabs. The three example searches on **Ask XMarket** are cached, so they work without Wi-Fi.
Record one full rehearsal with Cmd+Shift+5 and keep the video on the desktop as a backup if the laptop or projector misbehaves.
Without Wi-Fi the app still runs; only the map background tiles go blank, and the dots still show.

Times are cumulative. Say the **bold lines**; the rest are cues for what to click.

---

### 0:00 – 0:30 · Introduction (hook)
**"Buying a home is the biggest purchase most Houstonians make, but a first-time buyer has no way to tell whether an asking price is fair.
Zillow's estimate for a listed home mostly echoes the asking price, and the sticker price hides Texas-sized taxes and insurance.
Our case study: Maya, a UH grad earning $75,000 who wants to spend under $1,750 a month. XMarket answers her three questions:
what's a fair price, what can I afford, and which listings are bargains?"**

Point at the four headline numbers.

### 0:30 – 0:55 · 1 · The data
**"We used 25,948 Zillow listings from June 2024 across Greater Houston: 159 ZIP codes and 62 cities. The listings had location but no income data,
so we joined IRS income by ZIP code and Zillow's ZIP-level home-value and rent trends."**

Point at the coverage map.

### 0:55 – 1:40 · 2 · Exploring the data
- Histogram: **"Prices are very skewed. The average is $532k and the median is $382k, so we predict the log of price and every home counts equally."**
- Q3 bar charts: **"Location isn't a straight line. Price per sqft is highest downtown, lowest 10 to 15 miles out, then climbs again in the suburbs
  because homes there are bigger and newer. That curve is why we chose a tree model."**
- Q5: **"The real story: on Maya's salary only 6% of listings fit the 28% rule. Tax and insurance are 37% of the payment."**

### 1:40 – 2:10 · 3 · Cleaning + 4 · Features
- **"The 'clean' dataset hid four problems: 3,400 builder floor-plan ads, land mixed in with homes, lot sizes in mixed units, and fake build years.
  We rebuilt it from the raw data and wrote down a reason for every removed row."**
- **"Our best engineered feature is 'comps': the price per sqft of the 10 nearest *other* homes, which is exactly what appraisers use.
  We never use a home's own price per sqft, because that's the answer."**

### 2:10 – 3:00 · 5 · Metric + 6 · Model + 7 · Validation
- **"A linear-regression baseline misses by 18.7%. Gradient-boosted trees get that to 7.4%, and a weighted ensemble of three boosting models (CatBoost, XGBoost, LightGBM) gets 7.1%. We learned the weights on one part of the data and tested them on another, and the ensemble won on all 5 folds."**
- **"Every number is from 5-fold cross-validation: each home is scored by a model that never saw its price. If we'd cheated and fed in Zillow's estimate,
  we'd get 1.2% error and R² 0.995, which looks amazing and means nothing."**
- Zillow chart: **"How close is this to a professional tool? Zillow's own error, when it can't see the asking price, is 7.01%. Ours is 7.1%, built in a day
  from public data."**
- **"And our price ranges are honest: in every price band, 10% of homes fall below and 10% above, as designed."**

### 3:00 – 4:30 · Live demo
0. **Where should I live?** (sidebar, under *Put the model to work*). Inputs are already Maya's: $75k, works at the Medical Center, buying.
   **"Tell XMarket your income and where you work, and it ranks every ZIP code on affordability, commute and neighborhood strength.
   For Maya: #1 is 77033, 4.6 miles from work, where 30 of 94 homes fit her budget."** Move the *Short commute* slider to 5 to show the ranking adapt.
1. **Ask XMarket (AI)** → click the first example. **"Maya types in plain English. Claude turns it into filters; our model supplies every price.
   The AI never makes up a number."** Point at "Claude understood…" and the map.
2. Pick 5742 Cartagena St (first result) → **Explain this home**. **"Listed at $150k; our fair value is $249k. Claude explains why in plain English,
   and flags that a gap this big means: get an inspection."**
3. **Home report** (sidebar) → the waterfall: **"Every estimate breaks down into dollars: size, age, neighborhood comps."** Then the monthly cost:
   **"All-in, $1,425 a month: inside Maya's budget, and cheaper than renting it ($1,980)."**
4. **Rent or buy?** → **"If she stays at least 2 years, buying this home builds more wealth than renting it."** (Check the number on screen; it updates with the assumptions.)
5. If time allows: **Beat the model**: invite a judge to guess one price.

### 4:30 – 5:00 · Close
**"XMarket turns 19,000 listings into three answers for every Houston buyer: is it fair, can I afford it, and is it a deal.
It's honest about uncertainty and it explains itself. Next steps: train on actual sale prices from the county appraisal district,
and add flood-claim history. In Houston, that's the biggest hidden cost."**

---

## Likely judge questions

**Why list prices instead of sale prices?** It's the data we were given, and it matches the buyer's question: "is this asking price fair?"
Sale prices from HCAD would be the next step; we'd expect accuracy to improve, since sale prices are less noisy than asking prices.

**Why not use the Zestimate as a feature?** It's someone else's estimate of the answer. We tested it: error falls to 1.2% and R² rises to 0.995,
but the model just copies Zillow. We keep it as a benchmark only.

**How do you know you're not overfitting?** (1) 5-fold cross-validation, out-of-fold only. (2) Homes at identical coordinates are kept in the same fold.
(3) Comps are rebuilt inside each fold. (4) The validation curve flattens and never rises. (5) All 5 folds agree within 1 point. (6) Whole ZIP codes held out: 12.2%.

**Training error is 3.5% but validation is 7.4% (LightGBM). Isn't that overfitting?** Some gap is normal for boosted trees. The warning sign is validation
error rising as the model grows, and it doesn't. We only ever report the validation number.

**Why median % error?** It's what a buyer understands ("half our estimates are within 7.1%"), it isn't distorted by mansions, and it's the metric
Zillow publishes, so we can compare against a professional tool.

**Why boosted trees, not a neural network?** For ~19k rows of tabular data, gradient-boosted trees are the standard best choice: they handle curves,
interactions and missing values, train in minutes on a laptop CPU, and they're explainable with SHAP.

**How did you choose the ensemble weights?** Out-of-fold predictions from each model, then weights ≥ 0 that sum to 1, chosen to minimize absolute
log error (the cousin of our median % metric). The weights are learned on 4/5 of the homes and scored on the other 1/5 (nested CV), so the gain
isn't from tuning on the test set. A plain average scored almost as well (7.14% vs 7.12%). The bootstrap 95% CI for the gain over LightGBM
alone is 0.17–0.37 points, and it beats the best single model (CatBoost, 7.23%) too. Final weights: CatBoost 64%, XGBoost 23%, LightGBM 14%.
Random forest got 0.

**Does the ensemble lose explainability?** No. Each model's SHAP values are exact, and SHAP is linear, so the ensemble's explanation is the
weighted sum. The dollar waterfall still adds up exactly.

**Are the "underpriced" homes just bad houses?** Sometimes, and we say so in the app. The model can't see condition or flood history.
Gaps beyond −40% are labeled "verify first" and hidden by default. Treat a blue dot as a reason to look closer.

**Why does a budget search show mostly bargains?** A budget filter keeps homes that are cheap *for what they are*, which is what underpriced means.
Across all homes, each price band is ~10% under and ~10% over by design (see the band chart on page 6).

**Why didn't ZIP income help?** In random cross-validation the comps feature already captured it. In unseen ZIP codes it cut the error
from 14.5% to 13.4%, so we kept it.

**What does Claude actually do? Can it hallucinate prices?** Two things only: it turns plain English into structured filters (schema-constrained
JSON), and it rewrites our model's numbers as a paragraph. All prices come from our model. Answers are cached, and the app works without the AI.

**How does "Where should I live?" rank neighborhoods?** Every ZIP code that fits the budget and commute gets a 0–100 percentile on
affordability (share and number of homes that fit, or rent as a share of income), commute (straight-line miles), neighborhood strength
(5-year value growth and change since 2024) and bargains (share of homes our model flags as underpriced). Then it takes a weighted average using
the user's sliders. It's transparent and adjustable, not a black box. We don't have school, crime or flood data, and the page says so.

**Is the 2024 data stale?** It's a snapshot. The home report shows how each ZIP's values have moved since (Zillow index to Sept 2026). The pipeline
reruns in about 3 minutes, so it can be retrained on fresh listings.

**Fairness?** The model values homes; it doesn't approve loans. ZIP income is a neighborhood feature. We show affordability to empower buyers,
and we'd audit residuals by neighborhood before any real-world use.

## Key numbers (all measured on held-out homes)
| | |
|---|---|
| Homes modeled | 19,430 (from 25,948 raw listings) |
| Median % error | **7.1%** ensemble (LightGBM 7.4%, CatBoost 7.2%, linear baseline 18.7%) |
| R² (log price) | 0.917 |
| Within 10% of list price | 63% |
| Unseen ZIP codes | 12.2% |
| 80% range coverage | 80.8% (each price band ~10% below / ~10% above) |
| Zillow Zestimate, off-market (nationwide) | 7.01% |
| Fake score with Zestimate leak | 1.2% / R² 0.995 |
| Ensemble gain vs LightGBM | 0.26 pts, 95% CI 0.17–0.37, won 5/5 folds |
| +250 sqft / +1 bath / foreclosure | +$25k / +$13k / −16% |
| Listings affordable on $75k (28% rule) | 6% |
