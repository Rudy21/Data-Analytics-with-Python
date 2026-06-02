"""
Practical 7: Time Series Data Analysis Using Pandas
POs: PO1, PO2, PO3, PO4, PO5 | KLs: K1, K2, K3, K4, K5, K6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

np.random.seed(42)
plt.rcParams["figure.dpi"] = 100

# ── 1. Creating a Time Series ───────────────────────────────────────────────
print("=" * 60)
print("1. Creating a Time Series")
print("=" * 60)

dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
n     = len(dates)

# Simulate stock-price-like data with trend + seasonality + noise
trend      = np.linspace(100, 250, n)
seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
noise       = np.random.normal(0, 8, n)
price       = trend + seasonality + noise

ts = pd.Series(price, index=dates, name="Stock_Price")
print(ts.head(10))
print("\nDate range:", ts.index[0].date(), "→", ts.index[-1].date())
print("Length:", len(ts))

# ── 2. Basic Indexing & Slicing ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Time-Based Indexing & Slicing")
print("=" * 60)
print("Single date  :", ts["2021-06-15"])
print("Month slice  :", ts["2022-03"].mean().round(2), "(mean for Mar-2022)")
print("Year slice:\n", ts["2021"].describe().round(2))
print("Range slice:\n", ts["2022-01-01":"2022-01-10"])

# ── 3. Resampling ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Resampling")
print("=" * 60)
monthly_mean = ts.resample("ME").mean().round(2)
quarterly_max = ts.resample("QE").max().round(2)
yearly_stats  = ts.resample("YE").agg(["mean","min","max","std"]).round(2)

print("Monthly mean (last 6):\n", monthly_mean.tail(6))
print("\nQuarterly max:\n", quarterly_max)
print("\nYearly stats:\n", yearly_stats)

# ── 4. Rolling Window ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Rolling Window Statistics")
print("=" * 60)
ts_df = ts.to_frame()
ts_df["MA_7"]   = ts.rolling(window=7).mean()
ts_df["MA_30"]  = ts.rolling(window=30).mean()
ts_df["MA_90"]  = ts.rolling(window=90).mean()
ts_df["Std_30"] = ts.rolling(window=30).std()
ts_df["Min_7"]  = ts.rolling(window=7).min()
ts_df["Max_7"]  = ts.rolling(window=7).max()
print(ts_df.tail(10).round(2))

# ── 5. Exponential Weighted Moving Average ─────────────────────────────────
print("\n" + "=" * 60)
print("5. Exponential Weighted Moving Average (EWMA)")
print("=" * 60)
ts_df["EWM_30"] = ts.ewm(span=30).mean()
print(ts_df[["Stock_Price","MA_30","EWM_30"]].tail(8).round(2))

# ── 6. Shifting & Lag Features ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Shifting & Lag Features")
print("=" * 60)
ts_df["Lag_1"]  = ts.shift(1)
ts_df["Lag_7"]  = ts.shift(7)
ts_df["Lead_1"] = ts.shift(-1)
ts_df["Daily_Return"]  = ts.pct_change().round(6)
ts_df["7d_Return"]     = ts.pct_change(periods=7).round(6)
ts_df["30d_Return"]    = ts.pct_change(periods=30).round(6)
print(ts_df[["Stock_Price","Lag_1","Daily_Return","7d_Return"]].tail(8).round(4))

# ── 7. Date-Part Feature Engineering ───────────────────────────────────────
print("\n" + "=" * 60)
print("7. Date-Part Feature Engineering")
print("=" * 60)
ts_df["Year"]      = ts_df.index.year
ts_df["Month"]     = ts_df.index.month
ts_df["Day"]       = ts_df.index.day
ts_df["DayOfWeek"] = ts_df.index.day_name()
ts_df["Quarter"]   = ts_df.index.quarter
ts_df["WeekNo"]    = ts_df.index.isocalendar().week.astype(int)
ts_df["IsWeekend"] = ts_df.index.dayofweek >= 5
print(ts_df[["Stock_Price","Year","Month","DayOfWeek","Quarter","IsWeekend"]].head(8))

# ── 8. Handling Missing Values in Time Series ───────────────────────────────
print("\n" + "=" * 60)
print("8. Handling Missing Values (Time Series)")
print("=" * 60)
ts_missing = ts.copy()
missing_idx = np.random.choice(len(ts_missing), 30, replace=False)
ts_missing.iloc[missing_idx] = np.nan
print("Missing count:", ts_missing.isnull().sum())
ts_ffill   = ts_missing.fillna(method="ffill")
ts_bfill   = ts_missing.fillna(method="bfill")
ts_interp  = ts_missing.interpolate(method="time")
ts_rolling_fill = ts_missing.fillna(ts_missing.rolling(7, min_periods=1).mean())
print("After interpolation nulls:", ts_interp.isnull().sum())

# ── 9. Stationarity & Differencing ─────────────────────────────────────────
print("\n" + "=" * 60)
print("9. Stationarity & Differencing")
print("=" * 60)
ts_diff1 = ts.diff().dropna()
ts_diff2 = ts.diff().diff().dropna()
print("Original  – mean: %.2f  std: %.2f" % (ts.mean(), ts.std()))
print("1st Diff  – mean: %.4f  std: %.4f" % (ts_diff1.mean(), ts_diff1.std()))
print("2nd Diff  – mean: %.4f  std: %.4f" % (ts_diff2.mean(), ts_diff2.std()))
try:
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(ts.dropna())
    print("\nADF Statistic :", round(result[0], 4))
    print("p-value       :", round(result[1], 4))
    print("Stationary?   :", "Yes ✅" if result[1] < 0.05 else "No ❌")
except ImportError:
    print("statsmodels not installed; skipping ADF test.")

# ── 10. Seasonality Decomposition ──────────────────────────────────────────
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    print("\n" + "=" * 60)
    print("10. Seasonal Decomposition")
    print("=" * 60)
    decomp = seasonal_decompose(ts.resample("ME").mean(), model="additive")
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, comp, title in zip(
        axes,
        [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
        ["Observed", "Trend", "Seasonality", "Residual"]
    ):
        ax.plot(comp); ax.set_title(title); ax.set_ylabel("Value")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Seasonal Decomposition (Monthly Resampled)", fontsize=14)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/p7_decomposition.png", dpi=100); plt.show()
    print("✅ Decomposition plot saved.")
except ImportError:
    print("statsmodels not installed; skipping decomposition.")

# ── 11. Visualisation ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. Time Series Visualisations")
print("=" * 60)

# a) Raw + MAs
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ts.index, ts.values, alpha=0.3, color="steelblue", label="Daily Price")
ax.plot(ts_df["MA_30"].index, ts_df["MA_30"], color="orange", lw=2, label="30-d MA")
ax.plot(ts_df["MA_90"].index, ts_df["MA_90"], color="red",    lw=2, label="90-d MA")
ax.fill_between(ts_df.index,
                ts_df["MA_30"] - ts_df["Std_30"],
                ts_df["MA_30"] + ts_df["Std_30"],
                alpha=0.15, color="orange", label="±1 Std (30-d)")
ax.set_title("Stock Price with Moving Averages & Bollinger Band")
ax.set_xlabel("Date"); ax.set_ylabel("Price (₹)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=30); ax.legend(); plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p7_price_ma.png", dpi=100); plt.show()
print("✅ Moving average plot saved.")

# b) Monthly box-plot (seasonality pattern)
monthly_df = ts_df[["Stock_Price","Month"]].copy()
monthly_df["Month_Name"] = monthly_df.index.month_name().str[:3]
month_order = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(12, 5))
monthly_df["Month_Name"] = pd.Categorical(monthly_df["Month_Name"],
                                           categories=month_order, ordered=True)
for mo in month_order:
    vals = monthly_df[monthly_df["Month_Name"] == mo]["Stock_Price"]
    ax.boxplot(vals, positions=[month_order.index(mo)],
               widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
ax.set_xticks(range(12)); ax.set_xticklabels(month_order)
ax.set_title("Monthly Price Distribution (Seasonal Pattern)")
ax.set_xlabel("Month"); ax.set_ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p7_seasonal_box.png", dpi=100); plt.show()
print("✅ Seasonal box-plot saved.")

# c) Daily returns distribution
fig, ax = plt.subplots(figsize=(8, 4))
returns = ts_df["Daily_Return"].dropna()
ax.hist(returns, bins=60, color="steelblue", edgecolor="white", density=True)
mean_r, std_r = returns.mean(), returns.std()
x = np.linspace(returns.min(), returns.max(), 200)
from scipy.stats import norm
ax.plot(x, norm.pdf(x, mean_r, std_r), "r-", lw=2, label="Normal fit")
ax.axvline(mean_r, color="green", linestyle="--", label=f"Mean={mean_r:.4f}")
ax.set_title("Daily Returns Distribution")
ax.set_xlabel("Return"); ax.set_ylabel("Density")
ax.legend(); plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p7_returns_dist.png", dpi=100); plt.show()
print("✅ Returns distribution saved.")

# ── 12. Summary Statistics ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("12. Summary Statistics")
print("=" * 60)
yearly_summary = ts_df.groupby("Year")["Stock_Price"].agg(
    Mean="mean", Std="std", Min="min", Max="max",
    Pct_25=lambda x: x.quantile(0.25),
    Pct_75=lambda x: x.quantile(0.75)
).round(2)
print(yearly_summary)

print("\n✅ Practical 7 Complete!")
