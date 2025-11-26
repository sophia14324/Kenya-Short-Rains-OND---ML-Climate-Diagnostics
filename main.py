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
# 1. ENSO / IOD drivers 
# ════════════════════════════════════════════════════════════════════════════

def fetch_climate_drivers(
        nino_path: Path = Path("data/raw/nino34.long.anom.csv"),
        dmi_path : Path = Path("data/raw/dmi.had.long.csv"),
) -> pd.DataFrame:
    """
    Return DataFrame [year, nino34, dmi] – OND seasonal means.

    Assumes monthly values in the PSL-style files. We resample to
    “AS-OCT” so each ‘year’ corresponds to OND of that year.
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

    n34 = _load(nino_path, "nino34", -99.99)
    dmi = _load(dmi_path , "dmi"   , -9999)

    # OND seasonal means aligned to the OND season of each year.
    drv = (pd.concat([n34, dmi], axis=1)
             .resample(SEASON_FREQ).mean())
    drv.index = drv.index.year
    return drv.reset_index(names="year")

# ════════════════════════════════════════════════════════════════════════════
# 2. Constants / paths
# ════════════════════════════════════════════════════════════════════════════

DEF_BBOX   = (-4.62, 4.62, 33.5, 41.9)          # S, N, W, E (Kenya)
DATA_DIR   = Path("data/raw");      DATA_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR   = Path("data/processed");PROC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR    = Path("outputs");       OUT_DIR .mkdir(parents=True, exist_ok=True)

MONTH_FILE = DATA_DIR / f"chirps_{SEASON_CODE.lower()}.csv"   # OND monthly 0.05°

URL_MONTH = ("https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/"
             "tifs/chirps-v3.0.{year}.{month:02d}.tif")

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
    if len(season_df) < 20:
        return
    tau, p = stats.kendalltau(season_df.year, season_df.total_mm)
    slope  = stats.theilslopes(season_df.total_mm, season_df.year)[0]
    if p < 0.05:
        print(f"⭑ {SEASON_CODE} trend: {slope:+.1f} mm / yr "
              f"(τ={tau:.2f}, p={p:.3f})")
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
# 11. PLOT
# ════════════════════════════════════════════════════════════════════════════

def plot_season(season_df: pd.DataFrame, window:Optional[int]=10):
    fig, ax  = plt.subplots(figsize=(10,4))
    ax.plot(season_df.year, season_df.total_mm, lw=1.6,
            marker="o", label="Season total")

    q_lo,q_hi = season_df.total_mm.quantile([.33,.66])
    ax.axhspan(0,q_lo, 0,1, color="red",  alpha=.07, label="Dry tercile")
    ax.axhspan(q_hi,ax.get_ylim()[1],0,1,color="blue", alpha=.07,
               label="Wet tercile")

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 0))
    ax3.bar(season_df.year, season_df.spi3, width=.6,
            color="purple", alpha=.3, label="SPI-3")
    ax3.set_ylabel("SPI-3")
    ax3.axhline(0,color="purple",lw=.8,alpha=.4)

    if window and len(season_df)>=window:
        ax.plot(season_df.year,
                season_df.total_mm.rolling(window,center=True).mean(),
                lw=2.2, ls="--", label=f"{window}-season mean")
        
    ax.set_title(f"{SEASON_NAME} 1981–{season_df.year.max()}")
    ax.set_xlabel("Year"); ax.set_ylabel("mm")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(alpha=.3)

    lines,labels = [],[]
    for a in (ax,ax3):
        l,lbl = a.get_legend_handles_labels(); lines+=l; labels+=lbl
    ax.legend(lines,labels,loc="upper center",
              bbox_to_anchor=(0.5,-0.15), ncol=3)

    fig.tight_layout()
    fig.savefig(OUT_DIR/f"{SEASON_CODE.lower()}_season_totals.png", dpi=150)
    plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# 12. CLI ENTRY-POINT
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=f"{SEASON_NAME} ({SEASON_CODE}) ML & diagnostics pipeline"
    )
    ap.add_argument("--audit-missing", action="store_true",
                help=f"List years with 1 or 2 missing {SEASON_CODE} months")
    ap.add_argument("--start",type=int,default=1981)
    ap.add_argument("--end",  type=int,default=2024)  
    ap.add_argument("--bbox", nargs=4,type=float,help="S N W E")
    ap.add_argument("--run-all",action="store_true")
    ap.add_argument("--redownload",action="store_true")
    ap.add_argument("--download-only",  action="store_true")
    ap.add_argument("--preprocess-only",action="store_true")
    ap.add_argument("--forecast-only",  action="store_true")
    ap.add_argument("--cluster-only",   action="store_true")
    ap.add_argument("--classify-only",  action="store_true")
    ap.add_argument("--plot-only",      action="store_true",
                    help="Just refresh the OND season_totals plot")
    
    args = ap.parse_args()
    bbox = tuple(args.bbox) if args.bbox else DEF_BBOX

    # 1) Download (only when requested)
    if args.download_only or args.run_all:
        download_year_range(args.start, args.end, bbox,
                            force=args.redownload)

    # 2) Build seasonal series
    season = preprocess(bbox)

    # 3) Try to add climate drivers, but soft-skip if files missing
    nino_file = Path("data/raw/nino34.long.anom.csv")
    dmi_file  = Path("data/raw/dmi.had.long.csv")

    if nino_file.exists() and dmi_file.exists():
        drv = fetch_climate_drivers(nino_file, dmi_file)
        season = season.merge(drv, on="year", how="left")

        season['nino34_L1'] = season['nino34'].shift(1)
        season['dmi_L1']    = season['dmi'].shift(1)

        corr = (
            season[["total_mm", "nino34", "dmi"]]
            .corr()
            .loc[["nino34", "dmi"], "total_mm"]
        )

        print(f"Lag-0 correlations with {SEASON_CODE} total "
              f"({season.year.min()}–{season.year.max()}*)")
        print(corr.round(2))
    else:
        print("⚠️ ENSO/IOD driver CSVs not found – skipping teleconnection "
              "diagnostics (nino34 / DMI).")

    season.to_csv(
        PROC_DIR / f"{SEASON_CODE.lower()}_season_totals_with_drivers.csv",
        index=False
    )

    if args.audit_missing:
        audit_missing_season()
        return 

    if args.run_all or args.plot_only:
        plot_season(season)

    if args.run_all or args.forecast_only:
        forecast(season)

    if args.run_all or args.cluster_only:
        cluster(season)

    if args.run_all or args.classify_only:
        classify(season)

    trend_test(season)

if __name__ == "__main__":
    main()
