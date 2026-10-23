5-fold CV (grouped by location), n = 19,430 listings

| stage                                                  |    MAE_$ |   MdAE_$ |   MdAPE_% |   within_10% |   R2_log |
|:-------------------------------------------------------|---------:|---------:|----------:|-------------:|---------:|
| 0. Linear regression baseline (structure only)         | 157886   |  69786.7 |    18.738 |       29.156 |    0.664 |
| 1. Structure only (beds, baths, sqft, lot, year, type) | 121732   |  47682.8 |    12.956 |       40.777 |    0.797 |
| 2. + Raw location (lat/lon, ZIP)                       |  81094.6 |  29718.2 |     7.908 |       58.765 |    0.905 |
| 3. + Distance to Houston job hubs                      |  79666.1 |  28799.2 |     7.695 |       59.573 |    0.909 |
| 4. + Neighborhood comps (10 nearest listings)          |  78715.9 |  27877.5 |     7.443 |       60.932 |    0.91  |
| 5. + ZIP income, value & rent trends (IRS, Zillow)     |  79201.8 |  27721.5 |     7.467 |       60.911 |    0.91  |
| 6. + Listing type (new build, builder, foreclosure)    |  78656.5 |  27730.9 |     7.386 |       60.968 |    0.91  |
| 7. Ensemble: LightGBM + CatBoost + XGBoost             |  75472.9 |  26339.9 |     7.118 |       62.563 |    0.917 |