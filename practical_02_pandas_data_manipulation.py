"""
Practical 2: Data Manipulation Tasks with Pandas
POs: PO1, PO2, PO3, PO4, PO5 | KLs: K2, K3, K4, K6
"""

import pandas as pd
import numpy as np

# ── 1. Creating DataFrames ──────────────────────────────────────────────────
print("=" * 60)
print("1. Creating DataFrames")
print("=" * 60)

data = {
    "Name":       ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "Age":        [25, 30, 35, 28, 22],
    "Department": ["HR", "IT", "Finance", "IT", "HR"],
    "Salary":     [50000, 70000, 80000, 72000, 48000],
    "Experience": [2, 5, 10, 6, 1],
}
df = pd.DataFrame(data)
print(df)

# ── 2. Basic Exploration ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Basic Exploration")
print("=" * 60)
print("\nShape:", df.shape)
print("\ndtypes:\n", df.dtypes)
print("\nDescribe:\n", df.describe())
print("\nInfo:")
df.info()

# ── 3. Selecting Columns & Rows ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Selecting Columns & Rows")
print("=" * 60)
print("\nSingle column (Age):\n", df["Age"].values)
print("\nMultiple columns:\n", df[["Name", "Salary"]])
print("\nloc (rows 1-3):\n", df.loc[1:3])
print("\niloc (first 2 rows, first 3 cols):\n", df.iloc[:2, :3])

# ── 4. Filtering ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Filtering Data")
print("=" * 60)
print("\nSalary > 65000:\n", df[df["Salary"] > 65000])
print("\nDepartment == IT:\n", df[df["Department"] == "IT"])
print("\nAge > 25 AND Salary > 60000:\n",
      df[(df["Age"] > 25) & (df["Salary"] > 60000)])

# ── 5. Adding / Modifying Columns ───────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Adding & Modifying Columns")
print("=" * 60)
df["Bonus"] = df["Salary"] * 0.10
df["Salary_Hike"] = df["Salary"] + 5000
df["Senior"] = df["Experience"] >= 5
print(df)

# ── 6. Dropping Columns & Rows ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Dropping Columns & Rows")
print("=" * 60)
df_dropped = df.drop(columns=["Bonus", "Salary_Hike"])
df_dropped = df_dropped.drop(index=0)          # drop first row
print(df_dropped)

# ── 7. Sorting ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Sorting")
print("=" * 60)
print("\nSort by Salary desc:\n", df.sort_values("Salary", ascending=False))
print("\nSort by Department, then Age:\n",
      df.sort_values(["Department", "Age"]))

# ── 8. GroupBy & Aggregation ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. GroupBy & Aggregation")
print("=" * 60)
print("\nMean salary per department:\n",
      df.groupby("Department")["Salary"].mean())
print("\nMultiple aggregations:\n",
      df.groupby("Department").agg(
          Avg_Salary=("Salary", "mean"),
          Max_Age=("Age", "max"),
          Count=("Name", "count")
      ))

# ── 9. Handling Missing Values ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. Handling Missing Values")
print("=" * 60)
df_missing = df.copy()
df_missing.loc[1, "Salary"] = np.nan
df_missing.loc[3, "Age"]    = np.nan
print("\nDataFrame with NaN:\n", df_missing)
print("\nNull counts:\n", df_missing.isnull().sum())
df_filled = df_missing.fillna({"Salary": df_missing["Salary"].mean(),
                                "Age":    df_missing["Age"].median()})
print("\nAfter fillna:\n", df_filled)
df_dropped_na = df_missing.dropna()
print("\nAfter dropna:\n", df_dropped_na)

# ── 10. String Operations ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. String Operations")
print("=" * 60)
print("\nUppercase names:\n", df["Name"].str.upper().values)
print("\nNames containing 'a' (case-insensitive):\n",
      df[df["Name"].str.contains("a", case=False)])

# ── 11. Applying Functions ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. Applying Functions")
print("=" * 60)
df["Tax"] = df["Salary"].apply(lambda s: s * 0.30 if s > 60000 else s * 0.20)
print(df[["Name", "Salary", "Tax"]])

# ── 12. Merging DataFrames ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("12. Merging DataFrames")
print("=" * 60)
dept_info = pd.DataFrame({
    "Department": ["HR", "IT", "Finance"],
    "Location":   ["Mumbai", "Bangalore", "Delhi"],
})
merged = pd.merge(df, dept_info, on="Department", how="left")
print(merged[["Name", "Department", "Salary", "Location"]])

# ── 13. Pivot Table ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("13. Pivot Table")
print("=" * 60)
pivot = df.pivot_table(values="Salary", index="Department",
                       aggfunc=["mean", "max", "count"])
print(pivot)

print("\n✅ Practical 2 Complete!")
