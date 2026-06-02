import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Plotly – graceful import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not installed. Run: pip install plotly")
    print("Matplotlib/Seaborn advanced plots will still run.\n")

np.random.seed(42)
sns.set_theme(style="darkgrid")

# ── Dataset ─────────────────────────────────────────────────────────────────
n = 200
df = pd.DataFrame({
    "Age":        np.random.randint(20, 65, n),
    "Salary":     np.random.randint(25000, 150000, n),
    "Department": np.random.choice(["HR", "IT", "Finance", "Marketing", "Ops"], n),
    "Gender":     np.random.choice(["Male", "Female"], n),
    "Experience": np.random.randint(0, 30, n),
    "Score":      np.random.normal(70, 15, n).clip(0, 100).round(2),
    "City":       np.random.choice(["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune"], n),
})
df["Year"] = np.random.choice(range(2018, 2024), n)

# ════════════════════════════════════════════════════════════
# ADVANCED MATPLOTLIB
# ════════════════════════════════════════════════════════════

# ── 1. Annotated Line Chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
revenue = np.random.randint(100000, 500000, 12)
ax.plot(months, revenue, "o-", color="royalblue", lw=2, label="Revenue")
ax.fill_between(months, revenue, alpha=0.15, color="royalblue")
max_idx = np.argmax(revenue)
min_idx = np.argmin(revenue)
ax.annotate(f"Peak\n₹{revenue[max_idx]:,}",
            xy=(months[max_idx], revenue[max_idx]),
            xytext=(max_idx + 1 if max_idx < 11 else max_idx - 2,
                    revenue[max_idx] + 20000),
            arrowprops=dict(arrowstyle="->", color="green"), color="green", fontsize=9)
ax.annotate(f"Low\n₹{revenue[min_idx]:,}",
            xy=(months[min_idx], revenue[min_idx]),
            xytext=(min_idx + 1 if min_idx < 11 else min_idx - 2,
                    revenue[min_idx] - 50000),
            arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)
ax.set_title("Monthly Revenue with Annotations", fontsize=14)
ax.set_xlabel("Month"); ax.set_ylabel("Revenue (₹)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v/1e5:.1f}L"))
ax.legend(); plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p6_1_annotated_line.png", dpi=100); plt.show()
print("Plot 1: Annotated line chart saved.")

# ── 2. Stacked Bar Chart ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
dept_gender = df.groupby(["Department", "Gender"]).size().unstack(fill_value=0)
dept_gender.plot(kind="bar", stacked=True, ax=axes[0], colormap="Set2", edgecolor="white")
axes[0].set_title("Stacked Bar – Department by Gender")
axes[0].set_xlabel("Department"); axes[0].tick_params(axis="x", rotation=15)

dept_gender.plot(kind="bar", ax=axes[1], colormap="Set2", edgecolor="white")
axes[1].set_title("Grouped Bar – Department by Gender")
axes[1].set_xlabel("Department"); axes[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p6_2_stacked_bar.png", dpi=100); plt.show()
print("Plot 2: Stacked bar chart saved.")

# ── 3. Bubble Chart ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
dept_stats = df.groupby("Department").agg(
    Avg_Salary=("Salary","mean"),
    Avg_Exp=("Experience","mean"),
    Count=("Age","count")).reset_index()
scatter = ax.scatter(
    dept_stats["Avg_Exp"], dept_stats["Avg_Salary"],
    s=dept_stats["Count"] * 15,
    c=range(len(dept_stats)), cmap="viridis", alpha=0.8, edgecolors="black")
for _, row in dept_stats.iterrows():
    ax.annotate(row["Department"],
                (row["Avg_Exp"], row["Avg_Salary"]),
                textcoords="offset points", xytext=(8, 4), fontsize=9)
ax.set_title("Bubble Chart – Avg Experience vs Salary (size = count)")
ax.set_xlabel("Avg Experience (yrs)"); ax.set_ylabel("Avg Salary (₹)")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p6_3_bubble.png", dpi=100); plt.show()
print("Plot 3: Bubble chart saved.")

# ── 4. Twin-Axis Plot ───────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5))
yearly = df.groupby("Year").agg(Avg_Salary=("Salary","mean"),
                                 Avg_Score=("Score","mean")).reset_index()
color1, color2 = "royalblue", "tomato"
ax1.bar(yearly["Year"], yearly["Avg_Salary"], color=color1, alpha=0.7, label="Avg Salary")
ax1.set_xlabel("Year"); ax1.set_ylabel("Avg Salary (₹)", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax2 = ax1.twinx()
ax2.plot(yearly["Year"], yearly["Avg_Score"], "o-", color=color2, lw=2, label="Avg Score")
ax2.set_ylabel("Avg Score", color=color2); ax2.tick_params(axis="y", labelcolor=color2)
fig.suptitle("Yearly Avg Salary & Score (Twin Axis)", fontsize=13)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p6_4_twin_axis.png", dpi=100); plt.show()
print("Plot 4: Twin-axis plot saved.")

# ── 5. Heatmap (Custom) ─────────────────────────────────────────────────────
pivot = df.pivot_table(values="Salary", index="Department",
                        columns="Gender", aggfunc="mean").round(0)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white", ax=ax,
            cbar_kws={"label": "Avg Salary (₹)"})
ax.set_title("Avg Salary Heatmap – Department × Gender", fontsize=13)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p6_5_heatmap.png", dpi=100); plt.show()
print("Plot 5: Heatmap saved.")

# ── 6. Seaborn PairPlot ─────────────────────────────────────────────────────
pair_df = df[["Age", "Salary", "Experience", "Score", "Department"]].sample(80)
g = sns.pairplot(pair_df, hue="Department", diag_kind="kde",
                 plot_kws={"alpha": 0.5}, height=2.2)
g.figure.suptitle("Pair Plot – Numerical Features", y=1.02, fontsize=13)
g.figure.savefig("/mnt/user-data/outputs/p6_6_pairplot.png", dpi=80, bbox_inches="tight")
plt.show()
print("Plot 6: Pair plot saved.")

# ── 7. Seaborn FacetGrid ────────────────────────────────────────────────────
g = sns.FacetGrid(df, col="Department", col_wrap=3, height=3.2, sharey=False)
g.map(sns.histplot, "Salary", bins=12, kde=True)
g.set_axis_labels("Salary (₹)", "Count")
g.figure.suptitle("Salary Distribution by Department (FacetGrid)", y=1.03)
g.figure.savefig("/mnt/user-data/outputs/p6_7_facetgrid.png", dpi=100, bbox_inches="tight")
plt.show()
print("Plot 7: FacetGrid saved.")

# ── 8. Seaborn ClusterMap ────────────────────────────────────────────────────
corr_data = df[["Age", "Salary", "Experience", "Score"]].corr()
g = sns.clustermap(corr_data, annot=True, fmt=".2f", cmap="vlag",
                   linewidths=0.5, figsize=(6, 5))
g.figure.suptitle("Cluster Map – Feature Correlations", y=1.03)
g.figure.savefig("/mnt/user-data/outputs/p6_8_clustermap.png", dpi=100, bbox_inches="tight")
plt.show()
print("Plot 8: ClusterMap saved.")

# ── 9. Matplotlib Animation ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x_data, y_data = [], []
line, = ax.plot([], [], "b-", lw=2)
ax.set_xlim(0, 4 * np.pi); ax.set_ylim(-1.2, 1.2)
ax.set_title("Animated Sine Wave"); ax.set_xlabel("x"); ax.set_ylabel("sin(x)")
x_vals = np.linspace(0, 4 * np.pi, 200)

def init():
    line.set_data([], [])
    return (line,)

def update(frame):
    x_data.append(x_vals[frame])
    y_data.append(np.sin(x_vals[frame]))
    line.set_data(x_data, y_data)
    return (line,)

anim = FuncAnimation(fig, update, frames=len(x_vals),
                     init_func=init, blit=True, interval=20)
try:
    anim.save("/mnt/user-data/outputs/p6_9_animation.gif", writer="pillow", fps=30)
    print("Plot 9: Animation saved as GIF.")
except Exception as e:
    print(f"Could not save animation ({e}). Install pillow: pip install pillow")
plt.show()

# ════════════════════════════════════════════════════════════
# PLOTLY INTERACTIVE PLOTS
# ════════════════════════════════════════════════════════════
if PLOTLY_AVAILABLE:
    # ── 10. Plotly Scatter (interactive) ───────────────────────────────────
    fig = px.scatter(df, x="Experience", y="Salary",
                     color="Department", size="Score",
                     hover_data=["Age", "City", "Gender"],
                     title="Interactive Scatter – Experience vs Salary",
                     template="plotly_white")
    fig.update_traces(marker=dict(opacity=0.75))
    fig.write_html("/mnt/user-data/outputs/p6_10_plotly_scatter.html")
    print("Plot 10: Plotly scatter saved.")

    # ── 11. Plotly Bar (animated over years) ───────────────────────────────
    dept_year = df.groupby(["Year", "Department"])["Salary"].mean().reset_index()
    fig = px.bar(dept_year, x="Department", y="Salary",
                 color="Department", animation_frame="Year",
                 title="Avg Salary by Department (Animated by Year)",
                 range_y=[0, dept_year["Salary"].max() * 1.1],
                 template="plotly_white")
    fig.write_html("/mnt/user-data/outputs/p6_11_plotly_animated_bar.html")
    print("Plot 11: Plotly animated bar saved.")

    # ── 12. Plotly Box ──────────────────────────────────────────────────────
    fig = px.box(df, x="Department", y="Salary", color="Gender",
                 points="outliers", notched=True,
                 title="Salary Distribution – Box Plot (Interactive)",
                 template="plotly_white")
    fig.write_html("/mnt/user-data/outputs/p6_12_plotly_box.html")
    print("Plot 12: Plotly box plot saved.")

    # ── 13. Plotly Sunburst ─────────────────────────────────────────────────
    fig = px.sunburst(df, path=["Department", "City", "Gender"],
                      values="Salary",
                      title="Salary Hierarchy – Sunburst Chart",
                      color_discrete_sequence=px.colors.qualitative.Set3)
    fig.write_html("/mnt/user-data/outputs/p6_13_plotly_sunburst.html")
    print("Plot 13: Plotly sunburst saved.")

    # ── 14. Plotly 3D Scatter ───────────────────────────────────────────────
    fig = px.scatter_3d(df, x="Age", y="Experience", z="Salary",
                        color="Department", size="Score",
                        title="3D Scatter – Age / Experience / Salary",
                        opacity=0.7, template="plotly_white")
    fig.write_html("/mnt/user-data/outputs/p6_14_plotly_3d.html")
    print("Plot 14: Plotly 3D scatter saved.")

    # ── 15. Plotly Heatmap ──────────────────────────────────────────────────
    corr = df[["Age","Salary","Experience","Score"]].corr().round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}", showscale=True,
    ))
    fig.update_layout(title="Interactive Correlation Heatmap",
                      template="plotly_white")
    fig.write_html("/mnt/user-data/outputs/p6_15_plotly_heatmap.html")
    print("Plot 15: Plotly heatmap saved.")
else:
    print("Skipped Plotly plots (not installed).")

print("\nPractical 6 Complete! All plots saved.")
