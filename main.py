#!/usr/bin/env python
"""
Kenya October–November–December (OND) Short-Rains – 
ML & Climate Diagnostics Pipeline
© 26 Nov 2025
"""

# ───────────────────────── standard / 3rd-party ────────────────────────────
import argparse, time, calendar
from pathlib import Path
from typing   import Tuple, Optional, List

import numpy as np, pandas as pd, requests, rasterio
from rasterio.windows import from_bounds
from scipy   import stats
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster             import KMeans
from sklearn.preprocessing       import StandardScaler
from sklearn.ensemble            import RandomForestClassifier
import joblib, matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ════════════════════════════════════════════════════════════════════════════
# 0. CONFIG: SEASON + CLIMATOLOGY
# ════════════════════════════════════════════════════════════════════════════

SEASON_CODE   = "OND"
SEASON_NAME   = "Kenya Short-Rains"
SEASON_MONTHS = (10, 11, 12)             # Oct–Nov–Dec
SEASON_FREQ   = "AS-OCT"                 # “water year” starting in October

CLIM_REF_START = 1991                    # climatological baseline period
CLIM_REF_END   = 2020

# ════════════════════════════════════════════════════════════════════════════
# 1. Constants / paths
# ════════════════════════════════════════════════════════════════════════════

DEF_BBOX   = (-4.62, 4.62, 33.5, 41.9)          # S, N, W, E (Kenya)
DATA_DIR   = Path("data/raw");      DATA_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR   = Path("data/processed");PROC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR    = Path("outputs");       OUT_DIR .mkdir(parents=True, exist_ok=True)

MONTH_FILE = DATA_DIR / f"chirps_{SEASON_CODE.lower()}.csv"   # OND monthly 0.05°

URL_MONTH = ("https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/"
             "tifs/chirps-v3.0.{year}.{month:02d}.tif")

# ════════════════════════════════════════════════════════════════════════════
# 2. ENSO / IOD drivers 
# ════════════════════════════════════════════════════════════════════════════
def fetch_climate_drivers(
        nino_path: Path = DATA_DIR / "nino34.long.anom.csv",
        dmi_path : Path = DATA_DIR / "dmi.had.long.csv",
) -> pd.DataFrame:
    """
    Return DataFrame [year, nino34, dmi, nino34_AS, dmi_AS].

    - nino34, dmi: OND seasonal means (Oct–Dec of that year)
    - nino34_AS, dmi_AS: Aug–Sep means of the same year (lead for OND)
    """
    def _load(csv: Path, name: str, miss_flag: float) -> pd.Series:
        if not csv.exists():
            raise FileNotFoundError(f"{csv} missing – download from PSL first")
        df = (pd.read_csv(csv, comment="#")
                .rename(columns=lambda c: c.strip())
                .replace(miss_flag, np.nan))
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")[df.columns[1]]
        df.name = name
        return df

    # Monthly indices
    n34 = _load(nino_path, "nino34", -99.99)
    dmi = _load(dmi_path , "dmi"   , -9999)
    mon = pd.concat([n34, dmi], axis=1)

    # 1) OND seasonal means (Oct–Dec of same calendar year)
    ond = (mon.resample(SEASON_FREQ).mean())      # SEASON_FREQ = "AS-OCT"
    ond.index = ond.index.year                    # -> 1981, 1982, ...
    ond.index.name = "year"

    # 2) Aug–Sep means (same calendar year as OND)
    mdf = mon.copy()
    mdf["year"]  = mdf.index.year
    mdf["month"] = mdf.index.month

    aug_sep = (mdf[mdf["month"].isin([8, 9])]
               .groupby("year")[["nino34", "dmi"]]
               .mean()
               .add_suffix("_AS"))

    # Join OND + Aug–Sep on year
    drv = ond.join(aug_sep, how="left")

    return drv.reset_index()  # columns: year, nino34, dmi, nino34_AS, dmi_AS


# ════════════════════════════════════════════════════════════════════════════
# 3. CHIRPS helpers
# ════════════════════════════════════════════════════════════════════════════

def _mean_rain_mm(tif_bytes: bytes, bbox: Tuple[float,float,float,float]) -> float:
    with rasterio.MemoryFile(tif_bytes) as mem, mem.open() as src:
        win  = from_bounds(bbox[2], bbox[0], bbox[3], bbox[1], src.transform)
        arr  = src.read(1, window=win)
        arr  = np.where(arr <= -9990, np.nan, arr)
        return float(np.nanmean(arr))

def download_chirps_month(y:int, m:int, bbox, retry=3, t=60)->float:
    url = URL_MONTH.format(year=y, month=m)
    for k in range(1, retry+1):
        try:
            r = requests.get(url, timeout=t); r.raise_for_status()
            return _mean_rain_mm(r.content, bbox)
        except Exception as e:
            if k == retry: raise
            time.sleep(5*k)

# ════════════════════════════════════════════════════════════════════════════
# 4. Download monthly OND 1981-…
# ════════════════════════════════════════════════════════════════════════════

def download_year_range(start:int,end:int,bbox,*,force=False):
    """
    Download CHIRPS monthly data for OND (Oct–Dec) and store as CSV.
    """
    if MONTH_FILE.exists() and not force:
        print("✔️ using cached", MONTH_FILE); return
    rec=[]
    for y in range(start, end+1):
        for m in SEASON_MONTHS:
            try:
                mm = download_chirps_month(y, m, bbox)
                rec.append(dict(year=y, month=m, rain_mm=mm))
                print(f"Downloaded {y}-{m:02d}")
            except Exception as e:
                print(f"⚠️  {y}-{m:02d} failed: {e}")
    pd.DataFrame(rec).to_csv(MONTH_FILE, index=False)
    print("📁  Saved", MONTH_FILE)

# ───────────────────────────────────────────────────────────────
# 5. AUDIT seasons where only 1 or 2 of the three OND months are present 
# ───────────────────────────────────────────────────────────────

def audit_missing_season(month_file: Path = MONTH_FILE) -> None:
    if not month_file.exists():
        print("✖ No monthly file – run download first"); return

    df = pd.read_csv(month_file)
    df["month"] = df["month"].astype(int)

    pivot = (df.pivot_table(index="year",
                            columns="month",
                            values="rain_mm",
                            aggfunc="first")
               .reindex(columns=list(SEASON_MONTHS)))

    miss_cnt = pivot.isna().sum(axis=1)

    flagged = miss_cnt[(miss_cnt > 0) & (miss_cnt < len(SEASON_MONTHS))]
    if flagged.empty:
        print(f"✔️  All {SEASON_CODE} seasons have full "
              f"{len(SEASON_MONTHS)}-month coverage.")
    else:
        print(f"⚠️  Seasons with 1 or 2 missing {SEASON_CODE} tiles:")
        print(flagged.rename("missing_months").to_string())

# ════════════════════════════════════════════════════════════════════════════
# 6. Preprocess  (drop seasons with <3 months)
# ════════════════════════════════════════════════════════════════════════════

def preprocess(bbox) -> pd.DataFrame:
    """
    Build OND seasonal totals, anomalies, and SPI-3.
    """
    df = pd.read_csv(MONTH_FILE)
    df["month"] = df["month"].astype(int)
    df.loc[df.rain_mm < 0, "rain_mm"] = np.nan

    full = pd.MultiIndex.from_product(
                [range(df.year.min(), df.year.max() + 1), SEASON_MONTHS],
                names=["year", "month"])
    df = (df.set_index(["year", "month"])
            .reindex(full)
            .reset_index())

    # Seasonal totals – require all 3 months
    season = (df.groupby("year")
                .rain_mm
                .sum(min_count=len(SEASON_MONTHS))
                .rename("total_mm")
                .reset_index())

    dropped = season.total_mm.isna().sum()
    if dropped:
        print(f"ℹ️ dropped {dropped} incomplete {SEASON_CODE} season(s)")

    # Climatological baseline
    ref_mask = season.year.between(CLIM_REF_START, CLIM_REF_END)
    ref = season.loc[ref_mask, "total_mm"].mean(skipna=True)
    if np.isnan(ref):
        ref = season.total_mm.mean(skipna=True)

    season["anom_pct"] = 100 * (season.total_mm - ref) / ref
    season["anom_z"]   = (season.total_mm - season.total_mm.mean(skipna=True)) \
                         / season.total_mm.std(ddof=0, skipna=True)

    # SPI-3 (using gamma fit on valid totals)
    valid = season.total_mm.dropna()
    if valid.empty:
        print("⚠️ No valid season totals – skipping SPI computation")
        season["spi3"] = np.nan
    else:
        shp, loc, scl = stats.gamma.fit(valid, floc=0)
        season["spi3"] = stats.norm.ppf(
            stats.gamma.cdf(season.total_mm, shp, loc=loc, scale=scl))

    # NOTE: MAM-specific QC overrides removed; add OND-specific ones here if available.

    season.to_csv(PROC_DIR / f"{SEASON_CODE.lower()}_season_totals.csv",
                  index=False)
    return season

# ════════════════════════════════════════════════════════════════════════════
# 7. TREND SIGNIFICANCE 
# ════════════════════════════════════════════════════════════════════════════

def trend_test(season_df: pd.DataFrame):
    """
    Kendall rank test + Theil–Sen slope on OND totals.
    Drops NaN seasons before testing.
    """
    # Work only with seasons that have a valid total
    df = season_df.dropna(subset=["total_mm"]).copy()

    if len(df) < 20:
        print("Insufficient valid seasons for robust trend test "
              f"({len(df)} < 20) – skipping.")
        return

    tau, p = stats.kendalltau(df.year, df.total_mm)
    slope  = stats.theilslopes(df.total_mm, df.year)[0]

    if np.isnan(p):
        print("⚠️ Trend test returned p=NaN – likely due to tied values.")
        return

    if p < 0.05:
        print(f"⭑ Trend: {slope:+.1f} mm / yr (τ={tau:.2f}, p={p:.3f})")
    else:
        print(f"No significant monotonic trend in {SEASON_CODE} totals "
              f"(p={p:.2f}).")


# ════════════════════════════════════════════════════════════════════════════
# 8. FORECAST (ARIMA)
# ════════════════════════════════════════════════════════════════════════════

def forecast(season_df: pd.DataFrame, steps: int = 3):
    """
    ARIMA forecast of OND season totals. Soft-skips when < 4 seasons.
    """
    series = season_df.set_index("year")["total_mm"].asfreq(SEASON_FREQ)
    if series.isna().any():
        series = series.interpolate()
    if len(series) < 4:
        print(f"⚠️  Only {len(series)} season(s) available — "
              f"skipping forecast step.")
        return  

    model = ARIMA(series, order=(1, 0, 0)).fit()
    fc = model.get_forecast(steps=steps)
    fc_df = fc.summary_frame()
    fc_df.index = [series.index[-1].year + i for i in range(1, steps + 1)]
    fc_df.to_csv(OUT_DIR / f"{SEASON_CODE.lower()}_forecast.csv")

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(series.index.year, series, label="observed")
    plt.plot(fc_df.index, fc_df["mean"], "--", label="forecast")
    plt.fill_between(fc_df.index,
                     fc_df["mean_ci_lower"],
                     fc_df["mean_ci_upper"],
                     alpha=0.2)
    plt.title(f"{SEASON_NAME} — observed & ARIMA forecast")
    plt.xlabel("Year"); plt.ylabel("mm"); plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{SEASON_CODE.lower()}_forecast.png", dpi=150)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 9. CLUSTERING (monthly OND matrix)
# ════════════════════════════════════════════════════════════════════════════

def cluster(season_df: pd.DataFrame, k: int = 4, *, impute: bool = False):
    """
    K-means on the OND monthly matrix.
    Set impute=True to mean-fill NaNs instead of dropping seasons.
    """
    df = pd.read_csv(MONTH_FILE)
    pivot = (
        df.pivot(index="year", columns="month", values="rain_mm")
          .reindex(columns=list(SEASON_MONTHS))
    )

    if impute:
        pivot_filled = pivot.fillna(pivot.mean())
        n_dropped = 0
    else:
        n_dropped = pivot.isna().any(axis=1).sum()
        pivot_filled = pivot.dropna()

    if n_dropped:
        print(f"ℹ️  Dropped {n_dropped} {SEASON_CODE} season(s) with "
              f"missing months before clustering")

    scaler = StandardScaler()
    X      = scaler.fit_transform(pivot_filled.values)

    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
    pivot_filled["cluster"] = km.labels_
    pivot_filled.to_csv(OUT_DIR / f"{SEASON_CODE.lower()}_clustered_years.csv")

    joblib.dump(km, OUT_DIR / f"{SEASON_CODE.lower()}_kmeans.pkl")
    print("Cluster counts:", np.bincount(km.labels_))

# ════════════════════════════════════════════════════════════════════════════
# 10. CLASSIFICATION (dry/normal/wet terciles)
# ════════════════════════════════════════════════════════════════════════════

def classify(season_df: pd.DataFrame):
    """
    Random Forest classifier using lagged OND totals to predict tercile class.
    Labels: 0=dry, 1=normal, 2=wet.
    """
    q_lo, q_hi = season_df.total_mm.quantile([0.33, 0.66])
    def lab(x): return 0 if x < q_lo else 2 if x > q_hi else 1
    season_df["label"] = season_df.total_mm.apply(lab)

    for lag in (1, 2, 3):
        season_df[f"lag{lag}"] = season_df.total_mm.shift(lag)

    season_df = season_df.dropna()
    if season_df.empty:
        print("⚠️  Not enough labelled seasons to train classifier — "
              "skipping class step.")
        return

    X = season_df[[f"lag{l}" for l in (1, 2, 3)]].values
    y = season_df["label"].values
    clf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X, y)
    joblib.dump(clf, OUT_DIR / f"{SEASON_CODE.lower()}_rf_classifier.pkl")
    print(f"Training accuracy: {clf.score(X, y):.2f}")

# ════════════════════════════════════════════════════════════════════════════
# 11. PREDICTION
# ════════════════════════════════════════════════════════════════════════════
def run_predictive_models(season_df: pd.DataFrame,
                          target: str = "spi3") -> None:
    """
    Simple seasonal prediction using Aug–Sep drivers and lagged rainfall.

    Parameters
    ----------
    season_df : DataFrame
        Must contain columns:
        - 'year'
        - target ('total_mm' or 'spi3')
        - 'dmi_AS', 'nino34_AS'
    target : str
        Predictand: 'total_mm' or 'spi3'.

    Method
    ------
    - Predictors: dmi_AS, nino34_AS, target_lag1, target_lag2
    - Models: LinearRegression and RandomForestRegressor
    - Validation: Leave-one-year-out cross-validation (LOOCV)
    - Metrics: correlation, RMSE, tercile hit rates.
    """

    df = season_df.copy()

    if target not in df.columns:
        print(f"✖ Target '{target}' not found in DataFrame – skipping.")
        return

    required = ["year", "dmi_AS", "nino34_AS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"✖ Missing required columns for predictive model: {missing}")
        return

    # Create lagged targets
    for lag in (1, 2):
        df[f"{target}_lag{lag}"] = df[target].shift(lag)

    # Select columns and drop rows with missing predictors/target
    feature_cols = ["dmi_AS", "nino34_AS",
                    f"{target}_lag1", f"{target}_lag2"]

    use_cols = ["year", target] + feature_cols
    df_model = df[use_cols].dropna()

    if len(df_model) < 10:
        print(f"⚠️ Only {len(df_model)} usable seasons for modelling – "
              "results will be noisy.")
    if len(df_model) < 3:
        print("✖ Not enough seasons for LOOCV – skipping predictive model.")
        return

    years = df_model["year"].values
    y     = df_model[target].values
    X     = df_model[feature_cols].values

    n = len(df_model)
    preds_lin = np.full(n, np.nan)
    preds_rf  = np.full(n, np.nan)

    # LOOCV loop
    for i in range(n):
        train_idx = np.arange(n) != i
        test_idx  = ~train_idx

        X_train, y_train = X[train_idx], y[train_idx]
        X_test           = X[test_idx]

        # Linear regression
        lin = LinearRegression()
        lin.fit(X_train, y_train)
        preds_lin[i] = lin.predict(X_test)[0]

        # Random Forest regression
        rf = RandomForestRegressor(
            n_estimators=300,
            random_state=0,
            min_samples_leaf=1
        )
        rf.fit(X_train, y_train)
        preds_rf[i] = rf.predict(X_test)[0]
    

    # Helper: evaluate predictions
    def eval_predictions(y_true, y_pred, label: str):
        mask = ~np.isnan(y_pred)
        y_t = y_true[mask]
        y_p = y_pred[mask]

        if len(y_t) < 3:
            print(f"✖ Not enough valid cases for {label} metrics.")
            return

        corr = np.corrcoef(y_t, y_p)[0, 1]
        rmse = np.sqrt(mean_squared_error(y_t, y_p))

        # Terciles based on observed distribution
        q_lo, q_hi = np.quantile(y_t, [1/3, 2/3])
        obs_cat = np.where(y_t < q_lo, 0,
                            np.where(y_t > q_hi, 2, 1))
        pred_cat = np.where(y_p < q_lo, 0,
                            np.where(y_p > q_hi, 2, 1))

        overall_hit = (obs_cat == pred_cat).mean()

        # Hit rates for dry / wet extremes
        dry_mask = obs_cat == 0
        wet_mask = obs_cat == 2

        dry_hit = ((pred_cat == 0) & dry_mask).sum() / max(dry_mask.sum(), 1)
        wet_hit = ((pred_cat == 2) & wet_mask).sum() / max(wet_mask.sum(), 1)

        print(f"\n[{label}] LOOCV skill for target={target}")
        print(f"  Years used:           {len(y_t)}")
        print(f"  Correlation (r):      {corr:5.2f}")
        print(f"  RMSE:                 {rmse:5.2f}")
        print(f"  Tercile overall hit:  {100*overall_hit:5.1f}%")
        print(f"  Dry tercile hit:      {100*dry_hit:5.1f}%")
        print(f"  Wet tercile hit:      {100*wet_hit:5.1f}%")

        # Print metrics
        eval_predictions(y, preds_lin, "Linear Regression")
        eval_predictions(y, preds_rf , "Random Forest")

        # Save predictions to CSV for inspection
        out = df_model.copy()
        out[f"pred_{target}_lin_loocv"] = preds_lin
        out[f"pred_{target}_rf_loocv"]  = preds_rf

        out_file = OUT_DIR / f"{SEASON_CODE.lower()}_{target}_loocv_predictions.csv"
        out.to_csv(out_file, index=False)
        print(f"\n📁 Saved LOOCV predictions for {target} to: {out_file}")

def rf_predict_next_season(season_df: pd.DataFrame,
                        target: str = "total_mm",
                        year_next: Optional[int] = None) -> Tuple[Optional[int], Optional[float]]:
    """
    Train a RandomForestRegressor on all available seasons with Aug–Sep drivers
    and lagged target, then predict the target for the next season.

    Parameters
    ----------
    season_df : DataFrame
        Must contain:
        - 'year', target ('total_mm' or 'spi3')
        - 'dmi_AS', 'nino34_AS'
    target : str
        Predictand: 'total_mm' or 'spi3'.
    year_next : int or None
        Year to predict. If None, uses max(year) in season_df.

    Returns
    -------
    (year_next, prediction) or (None, None) if not possible.
    """
    df = season_df.copy()

    if target not in df.columns:
        print(f"✖ Target '{target}' not found – cannot predict.")
        return None, None

    for col in ["year", "dmi_AS", "nino34_AS"]:
        if col not in df.columns:
            print(f"✖ Missing '{col}' – cannot predict next season for {target}.")
            return None, None

    if year_next is None:
        year_next = int(df["year"].max())

    # Create lagged target
    for lag in (1, 2):
        df[f"{target}_lag{lag}"] = df[target].shift(lag)

    feature_cols = ["dmi_AS", "nino34_AS",
                    f"{target}_lag1", f"{target}_lag2"]

    # Training data: all years strictly before year_next with full data
    train_df = df[(df["year"] < year_next)].dropna(subset=[target] + feature_cols)
    if len(train_df) < 10:
        print(f"⚠️ Only {len(train_df)} seasons for training {target} – forecast will be noisy.")
    if len(train_df) < 3:
        print(f"✖ Not enough seasons to train RF for {target}.")
        return None, None

    X_train = train_df[feature_cols].values
    y_train = train_df[target].values

    # Features for the prediction year
    try:
        row_next = df.loc[df["year"] == year_next].iloc[0]
    except IndexError:
        print(f"✖ No row for year {year_next} – cannot build features for forecast.")
        return None, None

    # Need Aug–Sep drivers for year_next
    if pd.isna(row_next["dmi_AS"]) or pd.isna(row_next["nino34_AS"]):
        print(f"✖ Missing Aug–Sep drivers (dmi_AS / nino34_AS) for {year_next} – "
                f"cannot forecast {target}.")
        return None, None

    # Lagged targets for year_next come from previous years
    for lag in (1, 2):
        y_lag = df.loc[df["year"] == year_next - lag, target]
        if y_lag.isna().all():
            print(f"✖ Missing {target} for year {year_next - lag} – cannot forecast {target}.")
            return None, None
        row_next[f"{target}_lag{lag}"] = float(y_lag.values[0])

    x_next = row_next[feature_cols].values.reshape(1, -1)

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=0,
        min_samples_leaf=1
    )
    rf.fit(X_train, y_train)
    y_pred = float(rf.predict(x_next)[0])

    print(f"\n🌧 RF forecast for {target} in {year_next}: {y_pred:.2f}")
    return year_next, y_pred

# ════════════════════════════════════════════════════════════════════════════
# 11. PLOT
# ════════════════════════════════════════════════════════════════════════════

def plot_season(season_df: pd.DataFrame,
                window: Optional[int] = 10,
                fc_year: Optional[int] = None,
                fc_total: Optional[float] = None,
                fc_spi: Optional[float] = None):
    fig, ax  = plt.subplots(figsize=(10,4))
    ax.plot(season_df.year, season_df.total_mm, lw=1.6, marker="o", label="Season total")

    q_lo,q_hi = season_df.total_mm.quantile([.33,.66])
    ax.axhspan(0,q_lo, 0,1, color="red",  alpha=.07, label="Dry tercile")
    ax.axhspan(q_hi,ax.get_ylim()[1],0,1,color="blue", alpha=.07,label="Wet tercile")

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 0))
    ax3.bar(season_df.year, season_df.spi3, width=.6, color="purple", alpha=.3, label="SPI-3")
    ax3.set_ylabel("SPI-3"); ax3.axhline(0,color="purple",lw=.8,alpha=.4)

    # Rolling mean
    if window and len(season_df)>=window:
        ax.plot(season_df.year,
                season_df.total_mm.rolling(window,center=True).mean(),
                lw=2.2, ls="--", label=f"{window}-season mean")

    # === NEW: embed forecast into plot ===
    if fc_year is not None and fc_total is not None:
        ax.scatter(fc_year, fc_total, marker="*", s=150,
                   edgecolor="k", facecolor="gold",
                   zorder=5, label=f"Forecast total {fc_year}")
        ax.annotate(f"{fc_total:.0f}",
                    xy=(fc_year, fc_total),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8)

    if fc_year is not None and fc_spi is not None:
        ax3.bar(fc_year, fc_spi, width=0.4,
                color="black", alpha=.6, label=f"Forecast SPI-3 {fc_year}")

    ax.set_title(f"Kenya {SEASON_CODE} Rains 1981–{season_df.year.max()}")
    ax.set_xlabel("Year"); ax.set_ylabel("mm")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(alpha=.3)

    lines,labels = [],[]
    for a in (ax,ax3):
        l,lbl = a.get_legend_handles_labels(); lines+=l; labels+=lbl
    ax.legend(lines,labels,loc="upper center",
              bbox_to_anchor=(0.5,-0.18), ncol=3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{SEASON_CODE.lower()}_season_totals.png", dpi=150)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 12. CLI ENTRY-POINT
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=f"{SEASON_NAME} ({SEASON_CODE}) ML & diagnostics pipeline"
    )
    ap.add_argument("--predict-only", action="store_true",
                    help="Run simple predictive models (OND totals/SPI-3) with LOOCV")
    
    ap.add_argument("--predict-next", action="store_true",
                    help="Train RF on all past seasons and hindcast last OND season with drivers")

    ap.add_argument("--audit-missing", action="store_true",
                    help=f"List years with 1 or 2 missing {SEASON_CODE} months")
    ap.add_argument("--start", type=int, default=1981)
    ap.add_argument("--end",   type=int, default=2024)
    ap.add_argument("--bbox",  nargs=4, type=float, help="S N W E")

    ap.add_argument("--run-all",        action="store_true")
    ap.add_argument("--redownload",     action="store_true")
    ap.add_argument("--download-only",  action="store_true")
    ap.add_argument("--preprocess-only",action="store_true")
    ap.add_argument("--forecast-only",  action="store_true")
    ap.add_argument("--cluster-only",   action="store_true")
    ap.add_argument("--classify-only",  action="store_true")
    ap.add_argument("--plot-only",      action="store_true",
                    help="Just refresh the OND season_totals plot")
    
    args = ap.parse_args()
    bbox = tuple(args.bbox) if args.bbox else DEF_BBOX

    # ── 1) Download (only when requested) ───────────────────────
    if args.download_only or args.run_all:
        download_year_range(args.start, args.end, bbox,
                            force=args.redownload)
        # If this is *only* a download run, stop here.
        if args.download_only and not args.run_all:
            return

    # ── 2) Build seasonal series ────────────────────────────────
    season = preprocess(bbox)

    # ── 3) Add climate drivers if available (ENSO / IOD, including Aug–Sep) ─
    nino_file = DATA_DIR / "nino34.long.anom.csv"
    dmi_file  = DATA_DIR / "dmi.had.long.csv"

    if nino_file.exists() and dmi_file.exists():
        drv = fetch_climate_drivers(nino_file, dmi_file)
        season = season.merge(drv, on="year", how="left")

        # Aug–Sep 1-year lag (optional extra predictors)
        season['nino34_AS_L1'] = season['nino34_AS'].shift(1)
        season['dmi_AS_L1']    = season['dmi_AS'].shift(1)

        cols_for_corr = [
            "total_mm",      # OND seasonal rainfall
            "nino34", "dmi",               # OND-season means (lag 0)
            "nino34_AS", "dmi_AS",         # Aug–Sep lead
            "nino34_AS_L1", "dmi_AS_L1"    # Aug–Sep of previous year
        ]

        corr_df = season[cols_for_corr].dropna(subset=["total_mm"])
        corr = corr_df.corr().loc[
            ["nino34", "dmi", "nino34_AS", "dmi_AS",
             "nino34_AS_L1", "dmi_AS_L1"],
            "total_mm"
        ]

        print(f"Lag/lead correlations with {SEASON_CODE} total "
              f"({season.year.min()}–{season.year.max()}*)")
        print(corr.round(2))
    else:
        print("⚠️ ENSO/IOD driver CSVs not found – skipping teleconnection "
              "diagnostics (nino34 / DMI, Aug–Sep signals).")

    # ── 4) Save merged dataset (with whatever columns are available) ────────
    season.to_csv(
        PROC_DIR / f"{SEASON_CODE.lower()}_season_totals_with_drivers.csv",
        index=False
    )

    # ── 5) Misc utilities / early exits ─────────────────────────
    if args.audit_missing:
        audit_missing_season()
        return

    if args.preprocess_only and not args.run_all:
        # We've already done preprocessing + teleconnections + CSV output
        return 

    # ── 6) Simple predictive models (LOOCV) ─────────────────────
    if args.run_all or args.predict_only:
        if "total_mm" in season.columns:
            run_predictive_models(season, target="total_mm")
        if "spi3" in season.columns:
            run_predictive_models(season, target="spi3")

    # ── 7) RF hindcast for *last year with valid drivers* ───────
    fc_year = fc_total = fc_spi = None
    if args.run_all or args.predict_next:
        if ("dmi_AS" in season.columns) and ("nino34_AS" in season.columns):
            driver_mask = season["dmi_AS"].notna() & season["nino34_AS"].notna()
            valid_driver_years = season.loc[driver_mask, "year"]

            if valid_driver_years.empty:
                print("✖ No years with valid Aug–Sep drivers – cannot run predict-next.")
            else:
                target_year = int(valid_driver_years.max())
                print(f"\nHindcasting OND {target_year} using RF + Aug–Sep drivers...")

                # Forecast total_mm
                fc_year, fc_total = rf_predict_next_season(
                    season, target="total_mm", year_next=target_year
                )

                # Forecast SPI-3 if available
                if "spi3" in season.columns:
                    _, fc_spi = rf_predict_next_season(
                        season, target="spi3", year_next=target_year
                    )
        else:
            print("✖ dmi_AS / nino34_AS not in season DataFrame – cannot run predict-next.")

    # ── 8) Plot (with forecast star/bar if available) ───────────
    if args.run_all or args.plot_only or args.predict_next:
        plot_season(season,
                    fc_year=fc_year,
                    fc_total=fc_total,
                    fc_spi=fc_spi)

    # ── 9) Other modules (time-series ARIMA, clustering, classification) ────
    if args.run_all or args.forecast_only:
        forecast(season)

    if args.run_all or args.cluster_only:
        cluster(season)

    if args.run_all or args.classify_only:
        classify(season)

    # ── 10) Trend diagnostics ───────────────────────────────────
    trend_test(season)


if __name__ == "__main__":
    main()
