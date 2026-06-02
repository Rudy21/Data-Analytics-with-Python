"""
Practical 3: Advanced Data Manipulation Exercises Using Pandas
POs: PO1, PO2, PO3, PO4, PO5 | KLs: K1, K2, K3, K4, K5, K6
"""

import pandas as pd
import numpy as np

# ── Sample Dataset ──────────────────────────────────────────────────────────
np.random.seed(42)
n = 100
df = pd.DataFrame({
    "ID":         range(1, n + 1),
    "Name":       [f"Student_{i}" for i in range(1, n + 1)],
    "Age":        np.random.randint(18, 26, n),
    "Gender":     np.random.choice(["Male", "Female"], n),
    "Department": np.random.choice(["CS", "IT", "ECE", "ME", "CE"], n),
    "Marks_Math": np.random.randint(40, 100, n),
    "Marks_Sci":  np.random.randint(40, 100, n),
    "Marks_Eng":  np.random.randint(40, 100, n),
    "Attendance": np.random.uniform(50, 100, n).round(2),
})
# Inject some missing values
df.loc[np.random.choice(df.index, 10, replace=False), "Marks_Math"] = np.nan
df.loc[np.random.choice(df.index, 5,  replace=False), "Attendance"] = np.nan

print("=" * 60)
print("Dataset Overview")
print("=" * 60)
print(df.head(10))
print("\nShape:", df.shape)

# ── 1. Multi-Column Sorting ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. Multi-Column Sorting")
print("=" * 60)
sorted_df = df.sort_values(["Department", "Marks_Math"], ascending=[True, False])
print(sorted_df[["Name", "Department", "Marks_Math"]].head(10))

# ── 2. Advanced Filtering with query() ─────────────────────────────────────
print("\n" + "=" * 60)
print("2. Advanced Filtering with query()")
print("=" * 60)
result = df.query("Age > 20 and Department == 'CS' and Marks_Math > 70")
print(result[["Name", "Age", "Department", "Marks_Math"]])

# ── 3. Handling Missing Data – Advanced ────────────────────────────────────
print("\n" + "=" * 60)
print("3. Advanced Missing Data Handling")
print("=" * 60)
print("Missing values:\n", df.isnull().sum())
# Forward fill, then backward fill
df["Marks_Math"] = df["Marks_Math"].fillna(df["Marks_Math"].median())
df["Attendance"] = df["Attendance"].fillna(method="ffill").fillna(method="bfill")
print("\nAfter imputation:", df.isnull().sum().sum(), "nulls remain")

# ── 4. Derived Columns ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Derived / Calculated Columns")
print("=" * 60)
df["Total_Marks"]  = df["Marks_Math"] + df["Marks_Sci"] + df["Marks_Eng"]
df["Average"]      = (df["Total_Marks"] / 3).round(2)
df["Grade"] = pd.cut(df["Average"],
                     bins=[0, 50, 60, 70, 80, 100],
                     labels=["F", "D", "C", "B", "A"])
df["Pass"] = df["Average"] >= 50
print(df[["Name", "Average", "Grade", "Pass"]].head(10))

# ── 5. GroupBy – Complex Aggregations ──────────────────────────────────────
print("\n" + "=" * 60)
print("5. GroupBy – Complex Aggregations")
print("=" * 60)
dept_stats = df.groupby("Department").agg(
    Students    = ("ID", "count"),
    Avg_Marks   = ("Average", "mean"),
    Max_Marks   = ("Average", "max"),
    Min_Marks   = ("Average", "min"),
    Pass_Rate   = ("Pass",    "mean"),
    Avg_Attend  = ("Attendance", "mean"),
).round(2)
print(dept_stats)

# ── 6. Transform & Assign (within-group normalisation) ─────────────────────
print("\n" + "=" * 60)
print("6. Transform – Within-Group Normalization")
print("=" * 60)
df["Dept_Rank"] = df.groupby("Department")["Average"].rank(
    ascending=False, method="min")
df["Norm_Score"] = df.groupby("Department")["Average"].transform(
    lambda x: (x - x.mean()) / x.std())
print(df[["Name", "Department", "Average", "Dept_Rank", "Norm_Score"]].head(10))

# ── 7. Reshaping – Melt ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Reshaping – melt()")
print("=" * 60)
marks_df = df[["ID", "Marks_Math", "Marks_Sci", "Marks_Eng"]].head(5)
melted = marks_df.melt(id_vars="ID",
                        var_name="Subject",
                        value_name="Marks")
print(melted)

# ── 8. Reshaping – Pivot ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. Pivot Table")
print("=" * 60)
pivot = df.pivot_table(values="Average",
                       index="Department",
                       columns="Gender",
                       aggfunc="mean").round(2)
print(pivot)

# ── 9. String Operations ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. String Operations")
print("=" * 60)
df["Short_Name"] = df["Name"].str.replace("Student_", "S", regex=False)
df["Dept_Upper"] = df["Department"].str.upper()
print(df[["Name", "Short_Name", "Dept_Upper"]].head(5))

# ── 10. Datetime Operations ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. Datetime Operations")
print("=" * 60)
date_range = pd.date_range(start="2023-01-01", periods=n, freq="D")
df["Enroll_Date"] = date_range
df["Year"]   = df["Enroll_Date"].dt.year
df["Month"]  = df["Enroll_Date"].dt.month_name()
df["DayOfWeek"] = df["Enroll_Date"].dt.day_name()
print(df[["Name", "Enroll_Date", "Year", "Month", "DayOfWeek"]].head(5))

# ── 11. Merge / Join ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. Merge / Join")
print("=" * 60)
dept_map = pd.DataFrame({
    "Department": ["CS", "IT", "ECE", "ME", "CE"],
    "HOD":        ["Dr. Sharma", "Dr. Verma", "Dr. Patel",
                   "Dr. Gupta", "Dr. Singh"],
    "Building":   ["A", "B", "C", "D", "E"],
})
merged = pd.merge(df, dept_map, on="Department", how="left")
print(merged[["Name", "Department", "HOD", "Building"]].head(8))

# ── 12. Concatenation ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("12. Concatenation")
print("=" * 60)
df_batch1 = df.iloc[:50].copy()
df_batch2 = df.iloc[50:].copy()
full_df   = pd.concat([df_batch1, df_batch2], ignore_index=True)
print("Batch1:", len(df_batch1), "| Batch2:", len(df_batch2),
      "| Combined:", len(full_df))

# ── 13. Duplicates ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("13. Handling Duplicates")
print("=" * 60)
df_dup = pd.concat([df.head(5), df.head(5)], ignore_index=True)
print("With duplicates:", len(df_dup))
df_dup = df_dup.drop_duplicates(subset=["ID"])
print("After dedup:", len(df_dup))

# ── 14. Crosstab ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("14. Crosstab – Dept × Grade")
print("=" * 60)
print(pd.crosstab(df["Department"], df["Grade"]))

# ── 15. Export ──────────────────────────────────────────────────────────────
df.to_csv("/mnt/user-data/outputs/practical3_output.csv", index=False)
print("\n✅ Practical 3 Complete! Output saved to practical3_output.csv")
