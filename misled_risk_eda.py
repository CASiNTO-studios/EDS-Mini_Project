# %% [markdown]
# # E-Commerce User Feedback Survey - EDA with target `Misled_Risk`
# Run top to bottom (Jupyter / Colab / VS Code cells, or `python misled_risk_eda.py`).
# Requires: pandas, numpy, matplotlib, seaborn, openpyxl (scipy optional for the chi-square test).

# %% 1. Setup
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE = "E-Commerce Platform - User Feedback Survey.xlsx"   # <-- change to your file path
OUTDIR = "figures"                                          # charts are saved here as PNGs
os.makedirs(OUTDIR, exist_ok=True)

# Colours: platforms = categorical, risk classes = one blue hue from light (Low) to dark (High)
PLAT = {"Flipkart": "#2F6DB5", "Amazon": "#E08A1E", "Multiple": "#1B9E8A", "Meesho": "#8A5FB5"}
RISK = {"Low": "#C6DBEF", "Medium": "#6BAED6", "High": "#08519C"}
ORDER = ["Flipkart", "Amazon", "Multiple", "Meesho"]
INK, GRID = "#222222", "#E3E3E3"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": "#999999", "axes.labelcolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
})

def finish(name, ax=None):
    """Tidy, save and show the current figure."""
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{name}.png")
    plt.show()

def label_bars(ax, fmt="{:.0f}", pad=2):
    for c in ax.containers:
        ax.bar_label(c, labels=[fmt.format(v) for v in c.datavalues], padding=pad, fontsize=8.5, color=INK)

# %% 2. Load and understand the data
raw = pd.read_excel(FILE)
print("Shape:", raw.shape)
print(raw.dtypes.value_counts())
print("Duplicate rows:", raw.duplicated().sum())
print("Missing values per column (only columns with gaps):")
print(raw.isnull().sum()[raw.isnull().sum() > 0])

# %% 3. Cleaning: short column names, fix labels, drop incomplete rows
names = ["platform",
         "amz_freq", "amz_fake", "amz_ui", "amz_rent", "amz_coupon", "amz_acct", "amz_sent", "amz_words", "amz_ref",
         "flip_freq", "flip_fake", "flip_voice", "flip_rent", "flip_coupon", "flip_guest", "flip_sent", "flip_words", "flip_ref",
         "mee_freq", "mee_reviews", "mee_trust", "mee_sent", "mee_words", "mee_rent", "mee_login", "mee_ui", "mee_ref",
         "open_feature"]
assert raw.shape[1] == len(names), "Unexpected number of columns"
df = raw.copy()
df.columns = names

df = df[df.drop(columns="open_feature").notna().all(axis=1)].reset_index(drop=True)   # drop incomplete respondent(s)
for c in names[:-1]:
    df[c] = df[c].astype(str).str.strip().str.rstrip("*")                           # stray spaces / asterisk
df["platform"] = df["platform"].replace({"I use multiple platforms": "Multiple"})

junk = {"no", "none", "yes", "val1", "i am good with these", "nil", "na", "n/a"}
df["open_feature"] = df["open_feature"].apply(
    lambda v: "No suggestion" if pd.isna(v) or str(v).strip().lower() in junk else str(v).strip())
print(f"Rows after cleaning: {len(df)}")

# %% 4. Encoding (text answers -> numbers)
def yes_maybe_no(v):              # Yes = 2, Maybe = 1, anything else = 0
    return 2 if v.startswith("Yes") else (1 if v.startswith("Maybe") else 0)

freq   = {"Rarely": 1, "Monthly": 2, "Weekly": 3, "Daily": 4}
coupon = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Yes mostly": 3}
fake   = {"Yes, many times": 2, "Sometimes": 1}

enc = pd.DataFrame(index=df.index)
enc["platform"] = df["platform"]
for p in ["amz", "flip", "mee"]:
    enc[f"{p}_freq"] = df[f"{p}_freq"].map(freq)
for c in ["amz_rent", "amz_sent", "amz_words", "amz_ref", "flip_rent", "flip_guest", "flip_sent", "flip_words",
          "flip_ref", "mee_sent", "mee_words", "mee_rent", "mee_ref"]:
    enc[c] = df[c].apply(yes_maybe_no)
enc["amz_fake"]    = df["amz_fake"].map(fake).fillna(0)
enc["flip_fake"]   = df["flip_fake"].map(fake).fillna(0)
enc["amz_ui"]      = df["amz_ui"].map({"Yes, very overwhelming": 2, "Sometimes": 1}).fillna(0)
enc["amz_coupon"]  = df["amz_coupon"].map(coupon)
enc["flip_coupon"] = df["flip_coupon"].map(coupon)
enc["amz_acct"]    = df["amz_acct"].map({"Yes, I left without buying": 2, "It was annoying but I still bought": 1}).fillna(0)
enc["mee_reviews"] = df["mee_reviews"].map({"No, very few reviews": 3, "Reviews are there but not helpful": 2,
                                            "Sometimes": 1, "Yes, enough reviews available": 0})
enc["mee_trust"]   = df["mee_trust"].map({"Yes, I don't trust Meesho reviews at all": 2,
                                          "Somewhat affects my trust": 1}).fillna(0)
enc["mee_login"]   = df["mee_login"].map({"Yes, I just wanted to buy fast": 2, "Sometimes": 1}).fillna(0)
enc["mee_ui"]      = df["mee_ui"].map({"Poor, needs improvement": 1, "Basic but usable": 2, "Okay": 3, "Very good": 4})
# flip_voice is dropped: every answer means "did not know / never used", so it has no variation.
assert enc.drop(columns="platform").notna().all().all(), "Some answers were not encoded - check the label spelling"

# %% 5. Target variable: Misled_Risk
# Score = Amazon fake-review exposure + Flipkart fake-review exposure + Meesho review distrust (0-6)
enc["Misled_Score"] = enc["amz_fake"] + enc["flip_fake"] + enc["mee_trust"]
enc["Misled_Risk"] = pd.cut(enc["Misled_Score"], bins=[-1, 1, 3, 6], labels=["Low", "Medium", "High"])
print(enc["Misled_Risk"].value_counts().reindex(["Low", "Medium", "High"]))

# Helper columns for analysis (these are NOT inputs to the target)
enc["feature_interest"] = enc[["amz_sent", "amz_words", "flip_sent", "flip_words", "mee_sent", "mee_words"]].sum(axis=1)  # 0-12
enc["shop_activity"] = enc[["amz_freq", "flip_freq", "mee_freq"]].sum(axis=1)                                            # 3-12
SOURCE = ["amz_fake", "flip_fake", "mee_trust"]   # columns that define the target (leakage if used as predictors)

# %% 6. Statistical analysis (mean, median, mode, std, min, max)
num = enc.drop(columns=["platform", "Misled_Risk"]).astype(float)
stats = pd.DataFrame({
    "count": num.count(), "mean": num.mean(), "median": num.median(),
    "mode": num.mode().iloc[0], "std": num.std(), "min": num.min(), "max": num.max(),
}).round(2)
print(stats)
stats.to_csv(f"{OUTDIR}/descriptive_stats.csv")

# %% 7. Respondents by primary platform
counts = enc["platform"].value_counts().reindex(ORDER)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(counts.index, counts.values, color=[PLAT[p] for p in counts.index], width=0.6)
label_bars(ax); ax.set_title("Respondents by primary platform"); ax.set_ylabel("Respondents"); ax.grid(axis="x", visible=False)
finish("01_platform_counts")

# %% 8. Shopping frequency by platform
lab = {1: "Rarely", 2: "Monthly", 3: "Weekly", 4: "Daily"}
ft = pd.DataFrame({p: enc[f"{k}_freq"].value_counts().sort_index() for p, k in
                   [("Amazon", "amz"), ("Flipkart", "flip"), ("Meesho", "mee")]}).rename(index=lab)
ax = ft.plot.bar(figsize=(7, 4), color=[PLAT["Amazon"], PLAT["Flipkart"], PLAT["Meesho"]], width=0.75, rot=0)
label_bars(ax); ax.set_title("Shopping frequency by platform"); ax.set_ylabel("Respondents"); ax.grid(axis="x", visible=False)
ax.legend(title=None, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
finish("02_shopping_frequency")

# %% 9. Fake-review exposure and review distrust
fk = pd.DataFrame({
    "Amazon fake reviews": enc["amz_fake"].value_counts().sort_index(),
    "Flipkart fake reviews": enc["flip_fake"].value_counts().sort_index(),
    "Meesho distrust": enc["mee_trust"].value_counts().sort_index(),
})
fk.index = ["None / no effect", "Sometimes / somewhat", "Many times / no trust"]
ax = fk.plot.bar(figsize=(7.5, 4), color=[PLAT["Amazon"], PLAT["Flipkart"], PLAT["Meesho"]], width=0.75, rot=0)
label_bars(ax); ax.set_title("Fake-review exposure and review distrust"); ax.set_ylabel("Respondents"); ax.grid(axis="x", visible=False)
finish("03_fake_review_exposure")

# %% 10. Interest in proposed features (% answering "Yes")
feat = {"Sentiment labels": ["amz_sent", "flip_sent", "mee_sent"],
        "Review word summary": ["amz_words", "flip_words", "mee_words"],
        "Rental option": ["amz_rent", "flip_rent", "mee_rent"],
        "Referral rewards": ["amz_ref", "flip_ref", "mee_ref"]}
fy = pd.DataFrame({k: [(enc[c] == 2).mean() * 100 for c in v] for k, v in feat.items()},
                  index=["Amazon", "Flipkart", "Meesho"]).T
ax = fy.plot.bar(figsize=(8, 4.2), color=[PLAT["Amazon"], PLAT["Flipkart"], PLAT["Meesho"]], width=0.75, rot=0)
label_bars(ax, "{:.0f}%"); ax.set_title('Interest in proposed features (% answering "Yes")'); ax.set_ylabel("% of respondents")
ax.grid(axis="x", visible=False); ax.set_ylim(0, 62)
finish("04_feature_interest")

# %% 11. Purchase barriers (descriptive)
bar = pd.Series({"Amazon: left without buying\n(account needed)": (enc["amz_acct"] == 2).mean() * 100,
                 "Flipkart: wants\nguest checkout": (enc["flip_guest"] == 2).mean() * 100,
                 "Meesho: login is\na barrier": (enc["mee_login"] == 2).mean() * 100})
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(bar.index, bar.values, color=[PLAT["Amazon"], PLAT["Flipkart"], PLAT["Meesho"]], width=0.55)
label_bars(ax, "{:.0f}%"); ax.set_title("Share reporting the strongest purchase barrier"); ax.set_ylabel("% of respondents")
ax.grid(axis="x", visible=False)
finish("05_purchase_barriers")

# %% 12. Meesho UI rating
ui = enc["mee_ui"].value_counts().sort_index().rename(index={1: "Poor", 2: "Basic but usable", 3: "Okay", 4: "Very good"})
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(ui.index, ui.values, color=PLAT["Meesho"], width=0.6)
label_bars(ax); ax.set_title("Meesho UI and shopping experience rating"); ax.set_ylabel("Respondents"); ax.grid(axis="x", visible=False)
finish("06_meesho_ui")

# %% 13. Histogram of Misled_Score
fig, ax = plt.subplots(figsize=(6.5, 4))
h = enc["Misled_Score"].value_counts().sort_index()
ax.bar(h.index, h.values, color=RISK["Medium"], width=0.85)
label_bars(ax); ax.set_xticks(range(7)); ax.set_xlabel("Misled_Score (0 = low exposure, 6 = high)")
ax.set_ylabel("Respondents"); ax.set_title("Histogram of Misled_Score"); ax.grid(axis="x", visible=False)
ax.axvline(enc["Misled_Score"].mean(), color=INK, lw=1.2, ls="--")
ax.text(enc["Misled_Score"].mean() + 0.08, ax.get_ylim()[1] * 0.95, f"mean {enc['Misled_Score'].mean():.2f}", fontsize=9)
finish("07_misled_score_histogram")

# %% 14. Target class distribution (Misled_Risk)
tc = enc["Misled_Risk"].value_counts().reindex(["Low", "Medium", "High"])
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.bar(tc.index, tc.values, color=[RISK[k] for k in tc.index], width=0.6)
label_bars(ax); ax.set_title("Misled_Risk class distribution"); ax.set_ylabel("Respondents"); ax.grid(axis="x", visible=False)
finish("08_misled_risk_counts")

# %% 15. Misled_Risk by primary platform (100% stacked)
ct = pd.crosstab(enc["platform"], enc["Misled_Risk"], normalize="index").reindex(ORDER)[["Low", "Medium", "High"]] * 100
ax = ct.plot.barh(stacked=True, figsize=(8, 3.8), color=[RISK[k] for k in ct.columns], width=0.65, edgecolor="white", linewidth=1.5)
for c, col in zip(ax.containers, ct.columns):
    ax.bar_label(c, labels=[f"{v:.0f}%" for v in c.datavalues], label_type="center", fontsize=9,
                 color="white" if col != "Low" else INK)
ax.invert_yaxis(); ax.set_xlabel("% of respondents in group"); ax.set_ylabel("")
ax.set_title("Misled_Risk by primary platform"); ax.grid(axis="y", visible=False); ax.set_xlim(0, 100)
ax.legend(title="Misled_Risk", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
finish("09_risk_by_platform")

# %% 16. Box plot: Misled_Score by primary platform
fig, ax = plt.subplots(figsize=(7, 4.2))
sns.boxplot(data=enc, x="platform", y="Misled_Score", order=ORDER, palette=PLAT, width=0.5, linewidth=1.2,
            fliersize=3, ax=ax, hue="platform", legend=False)
sns.pointplot(data=enc, x="platform", y="Misled_Score", order=ORDER, estimator="mean", errorbar=None,
              linestyle="none", marker="D", color=INK, markersize=5, ax=ax)
ax.set_title("Misled_Score by primary platform (diamond = mean)"); ax.set_xlabel(""); ax.grid(axis="x", visible=False)
finish("10_boxplot_platform")

# %% 17. Misled_Risk vs shopping frequency (% High risk)
rows = []
for k, name in [("amz_freq", "Amazon"), ("flip_freq", "Flipkart"), ("mee_freq", "Meesho")]:
    g = enc.groupby(k)["Misled_Risk"].apply(lambda s: (s == "High").mean() * 100).rename(index=lab)
    rows.append(g.rename(name))
hr = pd.concat(rows, axis=1)
ax = hr.plot.bar(figsize=(7.5, 4), color=[PLAT["Amazon"], PLAT["Flipkart"], PLAT["Meesho"]], width=0.75, rot=0)
label_bars(ax, "{:.0f}%"); ax.set_title("% of High Misled_Risk by how often people shop on each platform")
ax.set_ylabel("% High risk"); ax.grid(axis="x", visible=False)
finish("11_risk_vs_frequency")

# %% 18. Scatter: Misled_Score vs overall feature interest (jittered)
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(7, 4.4))
jx = enc["Misled_Score"] + rng.uniform(-0.18, 0.18, len(enc))
jy = enc["feature_interest"] + rng.uniform(-0.18, 0.18, len(enc))
ax.scatter(jx, jy, s=22, alpha=0.5, color=PLAT["Flipkart"], edgecolor="white", linewidth=0.4)
m, b = np.polyfit(enc["Misled_Score"], enc["feature_interest"], 1)
xs = np.array([0, 6]); ax.plot(xs, m * xs + b, color=INK, lw=1.6)
r = enc["Misled_Score"].corr(enc["feature_interest"])
ax.set_xlabel("Misled_Score"); ax.set_ylabel("Feature-interest score (0-12)")
ax.set_title(f"Misled_Score vs interest in review features (r = {r:.2f})"); ax.set_xticks(range(7))
finish("12_scatter_score_vs_interest")

# %% 19. Correlation heatmap
cols = ["amz_freq", "flip_freq", "mee_freq", "amz_ui", "amz_coupon", "flip_coupon", "amz_acct", "flip_guest", "mee_login",
        "mee_ui", "mee_reviews", "amz_sent", "amz_words", "flip_sent", "flip_words", "mee_sent", "mee_words",
        "amz_ref", "flip_ref", "mee_ref", "feature_interest", "Misled_Score"]
corr = enc[cols].astype(float).corr()
fig, ax = plt.subplots(figsize=(10, 8.5))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6.5},
            linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.7, "label": "Pearson r"}, ax=ax)
ax.set_title("Correlation heatmap (ordinal-encoded survey answers)"); ax.grid(False)
plt.xticks(rotation=60, ha="right")
finish("13_correlation_heatmap")
print("\nCorrelation of each variable with Misled_Score (source columns excluded):")
print(corr["Misled_Score"].drop("Misled_Score").round(2).sort_values(ascending=False))

# %% 20. Optional: is Misled_Risk related to the primary platform? (chi-square test)
try:
    from scipy.stats import chi2_contingency
    chi2, p, dof, _ = chi2_contingency(pd.crosstab(enc["platform"], enc["Misled_Risk"]))
    print(f"Chi-square = {chi2:.2f}, dof = {dof}, p-value = {p:.4f} ->",
          "significant association" if p < 0.05 else "no significant association at 5% level")
except ImportError:
    print("scipy not installed - skip the chi-square test (pip install scipy)")

print(f"\nDone. Charts saved in ./{OUTDIR}/")
