import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set_theme(style="whitegrid")

# ── Sample Dataset ──────────────────────────────────────────────────────────
n = 100
df = pd.DataFrame({
    "Age":        np.random.randint(18, 60, n),
    "Salary":     np.random.randint(20000, 120000, n),
    "Department": np.random.choice(["HR", "IT", "Finance", "Marketing"], n),
    "Gender":     np.random.choice(["Male", "Female"], n),
    "Experience": np.random.randint(0, 20, n),
    "Score":      np.random.normal(70, 15, n).clip(0, 100).round(2),
})

# ════════════════════════════════════════════════════════════
# MATPLOTLIB PLOTS
# ════════════════════════════════════════════════════════════

# ── Plot 1: Line Chart ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
months  = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
sales_A = np.random.randint(5000, 20000, 12)
sales_B = np.random.randint(4000, 18000, 12)
ax.plot(months, sales_A, marker="o", label="Product A", color="steelblue")
ax.plot(months, sales_B, marker="s", label="Product B", color="tomato",
        linestyle="--")
ax.set_title("Monthly Sales Comparison", fontsize=14)
ax.set_xlabel("Month")
ax.set_ylabel("Sales (₹)")
ax.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p5_1_line_chart.png", dpi=100)
plt.show()
print("✅ Plot 1: Line Chart saved.")

# ── Plot 2: Bar Chart ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
dept_salary = df.groupby("Department")["Salary"].mean().sort_values()
axes[0].bar(dept_salary.index, dept_salary.values,
            color=["#4C72B0","#DD8452","#55A868","#C44E52"])
axes[0].set_title("Average Salary by Department")
axes[0].set_xlabel("Department")
axes[0].set_ylabel("Avg Salary (₹)")
for i, v in enumerate(dept_salary.values):
    axes[0].text(i, v + 500, f"₹{v:,.0f}", ha="center", fontsize=8)

# Horizontal bar
axes[1].barh(dept_salary.index, dept_salary.values,
             color=["#4C72B0","#DD8452","#55A868","#C44E52"])
axes[1].set_title("Horizontal Bar – Avg Salary")
axes[1].set_xlabel("Avg Salary (₹)")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p5_2_bar_chart.png", dpi=100)
plt.show()
print("✅ Plot 2: Bar Chart saved.")

# ── Plot 3: Histogram ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["Salary"], bins=20, color="steelblue", edgecolor="white")
axes[0].set_title("Salary Distribution")
axes[0].set_xlabel("Salary (₹)")
axes[0].set_ylabel("Frequency")
axes[0].axvline(df["Salary"].mean(), color="red", linestyle="--",
                label=f"Mean = ₹{df['Salary'].mean():,.0f}")
axes[0].legend()

axes[1].hist(df["Score"], bins=15, color="tomato", edgecolor="white",
             density=True)
axes[1].set_title("Score Distribution (Density)")
axes[1].set_xlabel("Score")
axes[1].set_ylabel("Density")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p5_3_histogram.png", dpi=100)
plt.show()
print("✅ Plot 3: Histogram saved.")

# ── Plot 4: Scatter Plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"HR": "red", "IT": "blue", "Finance": "green", "Marketing": "orange"}
for dept, grp in df.groupby("Department"):
    ax.scatter(grp["Experience"], grp["Salary"],
               label=dept, color=colors[dept], alpha=0.7, s=60)
ax.set_title("Experience vs Salary by Department")
ax.set_xlabel("Experience (years)")
ax.set_ylabel("Salary (₹)")
ax.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p5_4_scatter.png", dpi=100)
plt.show()
print("✅ Plot 4: Scatter Plot saved.")

# ── Plot 5: Pie Chart ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
dept_counts  = df["Department"].value_counts()
gender_counts = df["Gender"].value_counts()
axes[0].pie(dept_counts, labels=dept_counts.index, autopct="%1.1f%%",
            startangle=140, colors=["#4C72B0","#DD8452","#55A868","#C44E52"])
axes[0].set_title("Department Distribution")
axes[1].pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%",
            startangle=90, colors=["skyblue", "lightcoral"], explode=[0.05, 0])
axes[1].set_title("Gender Distribution")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p5_5_pie_chart.png", dpi=100)
plt.show()
print("✅ Plot 5: Pie Chart saved.")

# ── Plot 6: Box Plot (Matplotlib) ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
dept_groups = [df[df["Department"] == d]["Salary"].values
               for d in df["Department"].unique()]
ax.boxplot(dept_groups, labels=df["Department"].unique(), patch_artist=True,
           boxprops=dict(facecolor="lightblue"))
ax.set_title("Salary Distribution by Department (Box Plot)")
ax.set_xlabel("Department")
ax.set_ylabel("Salary (₹)")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p5_6_boxplot.png", dpi=100)
plt.show()
print("✅ Plot 6: Box Plot saved.")

# ── Plot 7: Subplots Grid ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Employee Data – Overview Dashboard", fontsize=14, fontweight="bold")

axes[0,0].hist(df["Age"], bins=15, color="#4C72B0", edgecolor="white")
axes[0,0].set_title("Age Distribution")

dept_salary = df.groupby("Department")["Salary"].mean()
axes[0,1].bar(dept_salary.index, dept_salary.values, color=["#DD8452","#55A868","#C44E52","#8172B2"])
axes[0,1].set_title("Avg Salary by Department")

axes[1,0].scatter(df["Age"], df["Salary"], alpha=0.5, c="steelblue", s=40)
axes[1,0].set_title("Age vs Salary")

axes[1,1].pie(df["Gender"].value_counts(), labels=df["Gender"].value_counts().index,
              autopct="%1.1f%%", colors=["skyblue","lightcoral"])
axes[1,1].set_title("Gender Split")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p5_7_dashboard.png", dpi=100)
plt.show()
print("✅ Plot 7: Dashboard saved.")

# ════════════════════════════════════════════════════════════
# SEABORN PLOTS
# ════════════════════════════════════════════════════════════

# ── Plot 8: Seaborn Histogram + KDE ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["Salary"], bins=20, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Salary Distribution (Seaborn)")
sns.histplot(df, x="Score", hue="Gender", kde=True, ax=axes[1])
axes[1].set_title("Score by Gender")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p5_8_sns_histplot.png", dpi=100)
plt.show()
print("✅ Plot 8: Seaborn Hist+KDE saved.")

# ── Plot 9: Seaborn Box & Violin ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x="Department", y="Salary", hue="Gender", ax=axes[0])
axes[0].set_title("Salary by Dept & Gender (Box)")
sns.violinplot(data=df, x="Department", y="Score", ax=axes[1], palette="muted")
axes[1].set_title("Score Distribution (Violin)")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p5_9_sns_box_violin.png", dpi=100)
plt.show()
print("✅ Plot 9: Box + Violin saved.")

# ── Plot 10: Seaborn Scatter (lmplot) ──────────────────────────────────────
g = sns.lmplot(data=df, x="Experience", y="Salary", hue="Department",
               height=5, aspect=1.4, scatter_kws={"alpha": 0.6})
g.set_axis_labels("Experience (yrs)", "Salary (₹)")
g.figure.suptitle("Experience vs Salary with Trend Lines", y=1.02)
g.figure.savefig("/mnt/user-data/outputs/p5_10_sns_lmplot.png", dpi=100, bbox_inches="tight")
plt.show()
print("✅ Plot 10: LM Plot saved.")

# ── Plot 11: Seaborn Count Plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(data=df, x="Department", hue="Gender", ax=ax, palette="Set2")
ax.set_title("Employee Count by Department and Gender")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p5_11_sns_countplot.png", dpi=100)
plt.show()
print("✅ Plot 11: Count Plot saved.")

# ── Plot 12: Seaborn Heatmap (Correlation) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
corr = df[["Age", "Salary", "Experience", "Score"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p5_12_sns_heatmap.png", dpi=100)
plt.show()
print("✅ Plot 12: Heatmap saved.")
