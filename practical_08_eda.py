import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")

# ── 1. Load & First Look ────────────────────────────────────────────────────
print("=" * 60)
print("1. Loading Dataset")
print("=" * 60)
df = sns.load_dataset("titanic")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nColumn names:\n", df.columns.tolist())

# ── 2. Data Types & Memory ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Data Types & Memory Usage")
print("=" * 60)
print(df.dtypes)
print("\nMemory usage:")
print(df.memory_usage(deep=True))

# ── 3. Summary Statistics ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Summary Statistics")
print("=" * 60)
print("Numerical:\n",    df.describe().round(2))
print("\nCategorical:\n", df.describe(include=["object","category"]))

# ── 4. Missing Value Analysis ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Missing Value Analysis")
print("=" * 60)
missing = pd.DataFrame({
    "Count":   df.isnull().sum(),
    "Percent": (df.isnull().sum() / len(df) * 100).round(2)
}).sort_values("Percent", ascending=False)
print(missing[missing["Count"] > 0])

# Visualise missing values
fig, ax = plt.subplots(figsize=(10, 4))
missing_filt = missing[missing["Count"] > 0]
ax.bar(missing_filt.index, missing_filt["Percent"],
       color=["tomato","steelblue","orange"])
ax.set_title("Missing Values (%) per Column")
ax.set_ylabel("% Missing"); ax.set_xlabel("Column")
for i, (col, row) in enumerate(missing_filt.iterrows()):
    ax.text(i, row["Percent"] + 0.3, f"{row['Percent']}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p8_1_missing.png", dpi=100); plt.show()
print("Missing value chart saved.")

# ── 5. Target Variable Analysis ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Target Variable – Survival")
print("=" * 60)
print(df["survived"].value_counts())
print("\nSurvival Rate:", df["survived"].mean().round(4))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df["survived"].value_counts().plot.bar(ax=axes[0], color=["tomato","steelblue"],
                                       edgecolor="white")
axes[0].set_title("Survival Count"); axes[0].set_xticklabels(["Died","Survived"], rotation=0)
df["survived"].value_counts().plot.pie(ax=axes[1], autopct="%1.1f%%",
                                       labels=["Died","Survived"],
                                       colors=["tomato","steelblue"], startangle=140)
axes[1].set_title("Survival Rate"); axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p8_2_target.png", dpi=100); plt.show()

# ── 6. Univariate Analysis – Numerical ─────────────────────────────────────
print("\n" + "=" * 60)
print("6. Univariate Analysis – Numerical Features")
print("=" * 60)
num_cols = ["age", "fare", "parch", "sibsp"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, col in enumerate(num_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[0, i], color="steelblue")
    axes[0, i].set_title(f"{col} – Distribution")
    sns.boxplot(y=df[col].dropna(), ax=axes[1, i], color="lightblue")
    axes[1, i].set_title(f"{col} – Box Plot")
fig.suptitle("Univariate – Numerical Columns", fontsize=13)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p8_3_univariate_num.png", dpi=100); plt.show()
print("Univariate numerical plots saved.")

# ── 7. Univariate Analysis – Categorical ───────────────────────────────────
print("\n" + "=" * 60)
print("7. Univariate Analysis – Categorical Features")
print("=" * 60)
cat_cols = ["pclass", "sex", "embarked", "who"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, cat_cols):
    df[col].value_counts().plot.bar(ax=ax, edgecolor="white", colormap="Set2")
    ax.set_title(col); ax.tick_params(axis="x", rotation=15)
fig.suptitle("Univariate – Categorical Columns", fontsize=13)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p8_4_univariate_cat.png", dpi=100); plt.show()
print("Univariate categorical plots saved.")

# ── 8. Bivariate Analysis – Survival vs Features ───────────────────────────
print("\n" + "=" * 60)
print("8. Bivariate Analysis – Survival vs Features")
print("=" * 60)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
# Survival by Sex
sns.countplot(data=df, x="sex", hue="survived", ax=axes[0,0], palette=["tomato","steelblue"])
axes[0,0].set_title("Survival by Sex")
# Survival by Pclass
sns.countplot(data=df, x="pclass", hue="survived", ax=axes[0,1], palette=["tomato","steelblue"])
axes[0,1].set_title("Survival by Class")
# Survival by Embarked
sns.countplot(data=df, x="embarked", hue="survived", ax=axes[0,2], palette=["tomato","steelblue"])
axes[0,2].set_title("Survival by Embarkation")
# Age distribution by survival
sns.histplot(data=df, x="age", hue="survived", kde=True, ax=axes[1,0], palette=["tomato","steelblue"])
axes[1,0].set_title("Age Distribution by Survival")
# Fare distribution by survival
sns.boxplot(data=df, x="survived", y="fare", ax=axes[1,1], palette=["tomato","steelblue"])
axes[1,1].set_title("Fare by Survival")
# Survival rate by class
surv_rate = df.groupby("pclass")["survived"].mean() * 100
axes[1,2].bar(surv_rate.index.astype(str), surv_rate.values,
              color=["#4C72B0","#DD8452","#55A868"])
axes[1,2].set_title("Survival Rate (%) by Class")
axes[1,2].set_ylabel("Survival Rate %")
for i, v in enumerate(surv_rate.values):
    axes[1,2].text(i, v + 1, f"{v:.1f}%", ha="center")
fig.suptitle("Bivariate Analysis – Survival vs Features", fontsize=14)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p8_5_bivariate.png", dpi=100); plt.show()
print("Bivariate plots saved.")

# ── 9. Correlation Analysis ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. Correlation Analysis")
print("=" * 60)
corr = df[["survived","pclass","age","sibsp","parch","fare"]].corr()
print(corr.round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=axes[0])
axes[0].set_title("Correlation Matrix")
surv_corr = corr["survived"].drop("survived").sort_values()
axes[1].barh(surv_corr.index, surv_corr.values,
             color=["tomato" if v < 0 else "steelblue" for v in surv_corr.values])
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_title("Feature Correlation with Survival")
axes[1].set_xlabel("Pearson r")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p8_6_correlation.png", dpi=100); plt.show()
print("Correlation plots saved.")

# ── 10. Outlier Detection ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. Outlier Detection")
print("=" * 60)
for col in ["age", "fare"]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}, "
          f"Bounds=[{lower:.1f}, {upper:.1f}], Outliers={len(outliers)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ["age", "fare"]):
    z = np.abs(stats.zscore(df[col].dropna()))
    ax.scatter(range(len(z)), df[col].dropna(), c=["tomato" if zi > 3 else "steelblue" for zi in z],
               alpha=0.5, s=20)
    ax.set_title(f"Z-score Outliers – {col} (red = |z|>3)")
    ax.set_xlabel("Index"); ax.set_ylabel(col)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p8_7_outliers.png", dpi=100); plt.show()
print("Outlier plots saved.")

# ── 11. Multivariate Analysis ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. Multivariate Analysis")
print("=" * 60)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Heatmap: survival rate by sex × class
pivot = df.pivot_table(values="survived", index="sex", columns="pclass")
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGn",
            linewidths=0.5, ax=axes[0])
axes[0].set_title("Survival Rate by Sex & Class")
# FacetGrid equivalent using groupby
fare_by_class_sex = df.groupby(["pclass","sex"])["fare"].mean().unstack()
fare_by_class_sex.plot(kind="bar", ax=axes[1], colormap="Set1", edgecolor="white")
axes[1].set_title("Avg Fare by Class and Sex")
axes[1].set_xlabel("Passenger Class"); axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p8_8_multivariate.png", dpi=100); plt.show()
print("Multivariate plots saved.")

# ── 12. Pair Plot ───────────────────────────────────────────────────────────
g = sns.pairplot(df[["survived","age","fare","pclass","sibsp"]].dropna(),
                 hue="survived", diag_kind="kde",
                 palette=["tomato","steelblue"], plot_kws={"alpha":0.5}, height=2)
g.figure.suptitle("Pair Plot", y=1.02)
g.figure.savefig("/mnt/user-data/outputs/p8_9_pairplot.png", dpi=80, bbox_inches="tight")
plt.show()
print("Pair plot saved.")

# ── 13. EDA Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("13. EDA Summary – Key Findings")
print("=" * 60)
print(f"  Dataset        : Titanic ({len(df)} passengers)")
print(f"  Survival Rate  : {df['survived'].mean()*100:.1f}%")
print(f"  Female Survival: {df[df['sex']=='female']['survived'].mean()*100:.1f}%")
print(f"  Male Survival  : {df[df['sex']=='male']['survived'].mean()*100:.1f}%")
for cls in [1, 2, 3]:
    rate = df[df['pclass']==cls]['survived'].mean()*100
    print(f"  Class {cls} Survival: {rate:.1f}%")
print(f"  Avg Age        : {df['age'].mean():.1f} years")
print(f"  Avg Fare       : ₹{df['fare'].mean():.2f}")
print(f"  Missing: Age={df['age'].isnull().sum()}, Cabin={df['cabin'].isnull().sum()}")

print("\nAll EDA plots saved.")
