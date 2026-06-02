import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")
np.random.seed(42)

# ── Dataset ──────────────────────────────────────────────────────────────
df = sns.load_dataset("titanic")
df_num = df[["age","fare","sibsp","parch"]].copy()

print("=" * 60)
print("Dataset: Titanic")
print("=" * 60)
print(df.shape)
print(df.head())

# ════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. Descriptive Statistics")
print("=" * 60)

def describe_custom(series, name):
    s = series.dropna()
    print(f"\n── {name} ──")
    print(f"  Count   : {len(s)}")
    print(f"  Mean    : {s.mean():.4f}")
    print(f"  Median  : {s.median():.4f}")
    mode_val = s.mode().values
    print(f"  Mode    : {mode_val}")
    print(f"  Std Dev : {s.std():.4f}")
    print(f"  Variance: {s.var():.4f}")
    print(f"  Min     : {s.min():.4f}")
    print(f"  Max     : {s.max():.4f}")
    print(f"  Range   : {s.max()-s.min():.4f}")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    print(f"  IQR     : {q3-q1:.4f}")
    print(f"  Skewness: {s.skew():.4f}")
    print(f"  Kurtosis: {s.kurtosis():.4f}")

describe_custom(df["age"], "Age")
describe_custom(df["fare"], "Fare")

# ── Frequency Table ─────────────────────────────────────────────────────────
print("\n── Frequency Table: Pclass ──")
freq = df["pclass"].value_counts().reset_index()
freq.columns = ["pclass","count"]
freq["percent"] = (freq["count"] / len(df) * 100).round(2)
freq["cumulative"] = freq["count"].cumsum()
print(freq)

# ════════════════════════════════════════════════════════════
# 2. PROBABILITY DISTRIBUTIONS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Probability Distributions")
print("=" * 60)
age_clean = df["age"].dropna()
mu, sigma = age_clean.mean(), age_clean.std()
print(f"Age: μ={mu:.2f}, σ={sigma:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
x = np.linspace(age_clean.min(), age_clean.max(), 200)
axes[0].hist(age_clean, bins=30, density=True, alpha=0.6, color="steelblue", label="Data")
axes[0].plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label="Normal PDF")
axes[0].set_title("Age – Normal Fit"); axes[0].legend()

# Empirical CDF
sorted_age = np.sort(age_clean)
cdf = np.arange(1, len(sorted_age)+1) / len(sorted_age)
axes[1].step(sorted_age, cdf, color="steelblue", label="Empirical CDF")
axes[1].plot(x, stats.norm.cdf(x, mu, sigma), "r--", lw=2, label="Normal CDF")
axes[1].set_title("CDF – Age"); axes[1].legend()

# QQ Plot
(osm, osr), (slope, intercept, r) = stats.probplot(age_clean)
axes[2].scatter(osm, osr, s=10, alpha=0.5, color="steelblue")
axes[2].plot(osm, slope*np.array(osm)+intercept, "r-", lw=2)
axes[2].set_title(f"Q-Q Plot – Age  (R²={r**2:.3f})")
axes[2].set_xlabel("Theoretical"); axes[2].set_ylabel("Sample")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p9_1_distributions.png", dpi=100); plt.show()
print("Distribution plots saved.")

# ════════════════════════════════════════════════════════════
# 3. HYPOTHESIS TESTING
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Hypothesis Testing")
print("=" * 60)
alpha = 0.05

def report_test(test_name, stat, p_val, h0, ha):
    print(f"\n{test_name}")
    print(f"  H0: {h0}")
    print(f"  Ha: {ha}")
    print(f"  Statistic: {stat:.4f}  |  p-value: {p_val:.4f}")
    if p_val < alpha:
        print(f"  → Reject H0 (p={p_val:.4f} < α={alpha})")
    else:
        print(f"  → Fail to Reject H0 (p={p_val:.4f} ≥ α={alpha})")

# 3a. One-Sample t-test
t_stat, p_val = stats.ttest_1samp(age_clean, 30)
report_test("3a. One-Sample t-test",
            t_stat, p_val,
            "Mean age = 30",
            "Mean age ≠ 30")

# 3b. Independent Two-Sample t-test (age: survived vs not)
age_surv   = df[df["survived"]==1]["age"].dropna()
age_nosurv = df[df["survived"]==0]["age"].dropna()
t_stat, p_val = stats.ttest_ind(age_surv, age_nosurv)
report_test("3b. Two-Sample t-test",
            t_stat, p_val,
            "Age of survivors == age of non-survivors",
            "Ages differ between groups")

# 3c. Paired t-test (synthetic – before/after training)
before = np.random.normal(65, 10, 30)
after  = before + np.random.normal(5, 3, 30)
t_stat, p_val = stats.ttest_rel(before, after)
report_test("3c. Paired t-test (Before vs After Training)",
            t_stat, p_val,
            "No difference before/after",
            "Training improved scores")

# 3d. Chi-Square Test of Independence
contingency = pd.crosstab(df["survived"], df["sex"])
chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
report_test("3d. Chi-Square Test (Survival × Sex)",
            chi2, p_val,
            "Survival is independent of sex",
            "Survival depends on sex")
print(f"  DoF: {dof}")
print("  Contingency Table:\n", contingency)

# 3e. One-Way ANOVA (fare across classes)
fare_c1 = df[df["pclass"]==1]["fare"].dropna()
fare_c2 = df[df["pclass"]==2]["fare"].dropna()
fare_c3 = df[df["pclass"]==3]["fare"].dropna()
f_stat, p_val = stats.f_oneway(fare_c1, fare_c2, fare_c3)
report_test("3e. One-Way ANOVA (Fare across Classes)",
            f_stat, p_val,
            "Mean fare is equal across classes",
            "At least one class has different mean fare")

# 3f. Shapiro-Wilk Normality Test
stat, p_val = stats.shapiro(age_clean.sample(50))
report_test("3f. Shapiro-Wilk Normality Test (Age, n=50)",
            stat, p_val,
            "Age is normally distributed",
            "Age is NOT normally distributed")

# 3g. Levene's Test for Equal Variances
stat, p_val = stats.levene(fare_c1, fare_c2, fare_c3)
report_test("3g. Levene's Test (Equal Variance – Fare by Class)",
            stat, p_val,
            "Variances are equal across classes",
            "Variances differ across classes")

# ════════════════════════════════════════════════════════════
# 4. CORRELATION
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Correlation Analysis")
print("=" * 60)
num_df = df[["survived","pclass","age","sibsp","parch","fare"]].dropna()

# Pearson
pearson_corr = num_df.corr(method="pearson")
# Spearman
spearman_corr = num_df.corr(method="spearman")

print("Pearson Correlation:\n", pearson_corr.round(3))
print("\nSpearman Correlation:\n", spearman_corr.round(3))

# Age vs Fare point-biserial equivalent (Pearson r)
r, p = stats.pearsonr(num_df["age"], num_df["fare"])
print(f"\nPearson r (Age vs Fare): {r:.4f}  p={p:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
axes[0].set_title("Pearson Correlation")
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
axes[1].set_title("Spearman Correlation")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p9_2_correlation.png", dpi=100); plt.show()
print("Correlation heatmaps saved.")

# ════════════════════════════════════════════════════════════
# 5. REGRESSION ANALYSIS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. Simple Linear Regression (Age → Fare)")
print("=" * 60)
x = num_df["age"].values
y = num_df["fare"].values
slope, intercept, r_val, p_val, se = stats.linregress(x, y)
print(f"  Intercept : {intercept:.4f}")
print(f"  Slope     : {slope:.4f}")
print(f"  R         : {r_val:.4f}")
print(f"  R²        : {r_val**2:.4f}")
print(f"  p-value   : {p_val:.4f}")
print(f"  Std Error : {se:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, alpha=0.4, color="steelblue", s=20, label="Data")
x_line = np.linspace(x.min(), x.max(), 200)
ax.plot(x_line, intercept + slope * x_line, "r-", lw=2,
        label=f"y = {slope:.3f}x + {intercept:.2f}  (R²={r_val**2:.3f})")
ax.set_xlabel("Age"); ax.set_ylabel("Fare")
ax.set_title("Simple Linear Regression: Age → Fare")
ax.legend(); plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p9_3_regression.png", dpi=100); plt.show()
print("Regression plot saved.")

# ════════════════════════════════════════════════════════════
# 6. CONFIDENCE INTERVALS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. Confidence Intervals (95%)")
print("=" * 60)
for col in ["age", "fare"]:
    s = df[col].dropna()
    ci = stats.t.interval(confidence=0.95, df=len(s)-1,
                           loc=s.mean(), scale=stats.sem(s))
    print(f"  {col}: mean={s.mean():.2f}  95% CI = [{ci[0]:.2f}, {ci[1]:.2f}]")

# ════════════════════════════════════════════════════════════
# 7. NON-PARAMETRIC TESTS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. Non-Parametric Tests")
print("=" * 60)
stat, p = stats.mannwhitneyu(age_surv, age_nosurv, alternative="two-sided")
report_test("7a. Mann-Whitney U (Age: Survivors vs Non-survivors)",
            stat, p,
            "Same age distribution in both groups",
            "Age distributions differ")

stat, p = stats.kruskal(fare_c1, fare_c2, fare_c3)
report_test("7b. Kruskal-Wallis (Fare across Classes)",
            stat, p,
            "Same fare distribution across classes",
            "Fare distributions differ across classes")

print("\nPractical 9 Complete!")
