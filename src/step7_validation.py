"""
STEP 7 - Validation Tests
==========================
Replicates Section 4 of the paper using cross-sectional data.

The paper runs FIVE validation tests (Table 3 Panels A-E).
Since we have 1 year of data across 30 companies, we adapt each test
to the cross-sectional setting:

  Test 1 (Panel B equivalent): Agility vs Board Size & Independence
          Prediction: Smaller boards → higher agility (Lehn 2018, 2021)

  Test 2 (Panel D equivalent): Agility vs Financial Constraints
          Prediction: More agile firms are less financially constrained

  Test 3 (Panel E equivalent): Agility vs Stock Performance (BSE returns)
          Prediction: More agile firms had better returns during COVID (2020)
          or recent market downturns

  Test 4: Sector-level analysis (replaces Figure 3)
          Which Nifty 50 sectors are most/least agile?

  Test 5: Correlation analysis
          How does agility correlate with firm size, leverage, ROA?

DATA YOU NEED TO PROVIDE:
  data/financial/company_financials.csv  — see format below

  Required columns:
    company       — must match names in agility_scores.csv exactly
    board_size    — total number of directors on board
    pct_ind_dir   — % independent directors (0-100)
    leverage      — total debt / total assets
    log_assets    — log(total assets in crore)
    roa           — return on assets (net profit / total assets)
    sales_growth  — YoY revenue growth (%)
    stock_return  — annual stock return (%) for the year of the MD&A
    sector        — BSE/NSE sector classification

  Data sources:
    - BSE/NSE annual filings
    - Screener.in → company pages → financial data
    - Moneycontrol.com → company pages → balance sheet
    - BSE → corporate governance reports → board composition

Requirements:
    pip install pandas numpy matplotlib seaborn statsmodels scipy
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

AGILITY_CSV    = "outputs/agility_scores.csv"
FINANCIAL_CSV  = "data/financial/company_financials.csv"
OUT_FOLDER     = "outputs/validation"
FIGURES_FOLDER = "outputs/figures"

os.makedirs(OUT_FOLDER,     exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"primary": "#1f4e79", "secondary": "#c00000", "accent": "#375623"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def print_section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def save_fig(name: str):
    path = os.path.join(FIGURES_FOLDER, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {path}")


def ols_summary(y_col: str, x_cols: list, df: pd.DataFrame, label: str):
    """Run OLS regression and print a clean summary table."""
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  statsmodels not installed. pip install statsmodels")
        return None

    sub = df[[y_col] + x_cols].dropna()
    if len(sub) < 5:
        print(f"  Not enough data for regression ({len(sub)} obs).")
        return None

    X = sm.add_constant(sub[x_cols])
    y = sub[y_col]
    model = sm.OLS(y, X).fit()

    print(f"\n  OLS: {y_col} ~ {' + '.join(x_cols)}")
    print(f"  N={len(sub)}  R²={model.rsquared:.3f}  Adj-R²={model.rsquared_adj:.3f}")
    print(f"  {'Variable':<25} {'Coef':>10} {'Std Err':>10} {'t':>8} {'p':>8}")
    print(f"  {'-'*63}")
    for var in model.params.index:
        coef = model.params[var]
        se   = model.bse[var]
        t    = model.tvalues[var]
        p    = model.pvalues[var]
        sig  = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"  {var:<25} {coef:>10.4f} {se:>10.4f} {t:>8.3f} {p:>8.4f} {sig}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading agility scores …")
agility_df = pd.read_csv(AGILITY_CSV)
print(f"  {len(agility_df)} companies loaded from agility_scores.csv")
print(f"  Columns: {list(agility_df.columns)}")

# Check if financial data exists
has_financial = os.path.exists(FINANCIAL_CSV)
if has_financial:
    print("\nLoading financial data …")
    fin_df = pd.read_csv(FINANCIAL_CSV)
    print(f"  {len(fin_df)} rows, columns: {list(fin_df.columns)}")
    df = agility_df.merge(fin_df, on="company", how="left")
    print(f"  Merged dataset: {len(df)} rows, {df.notna().sum().sum()} non-null values")
else:
    print(f"\n  ⚠ Financial data file not found: {FINANCIAL_CSV}")
    print("  ⚠ Running ONLY the agility-only tests (Tests 4 and 5 partial).")
    print("  ⚠ Create the CSV with board/financial data to run full validation.")
    df = agility_df.copy()


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — Agility Distribution & Descriptive Statistics
# (Always runs — just needs agility scores)
# ═════════════════════════════════════════════════════════════════════════════
print_section("TEST 1: Agility Score Distribution")

desc = df["agility_score"].describe()
print(f"\n  {desc.to_string()}")

# Divide into quartiles
df["agility_quartile"] = pd.qcut(
    df["agility_score"], q=4,
    labels=["Q1 (least agile)", "Q2", "Q3", "Q4 (most agile)"],
    duplicates="drop"
)
print("\n  Companies per quartile:")
print(df["agility_quartile"].value_counts().sort_index().to_string())

# ── Figure 1: Distribution histogram ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df["agility_score"], bins=10, color=COLORS["primary"],
             edgecolor="white", alpha=0.85)
axes[0].axvline(df["agility_score"].median(), color=COLORS["secondary"],
                linestyle="--", linewidth=2, label="Median")
axes[0].set_title("Distribution of Agility Scores", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Agility Score")
axes[0].set_ylabel("Number of Companies")
axes[0].legend()

axes[1].barh(
    df.sort_values("agility_score")["company"],
    df.sort_values("agility_score")["agility_score"],
    color=COLORS["primary"], alpha=0.8
)
axes[1].set_title("Agility Score by Company", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Agility Score")

plt.tight_layout()
save_fig("test1_agility_distribution.png")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — Correlation Matrix
# (Always runs)
# ═════════════════════════════════════════════════════════════════════════════
print_section("TEST 2: Correlation Analysis")

available_cols = [c for c in
    ["agility_score", "threat_score", "response_score", "leverage",
     "log_assets", "roa", "sales_growth", "stock_return", "board_size",
     "pct_ind_dir", "total_tokens"]
    if c in df.columns]

if len(available_cols) >= 3:
    corr = df[available_cols].corr()
    print("\n  Pearson correlation matrix:")
    print(corr.round(3).to_string())

    # Save correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, cbar_kws={"shrink": 0.8},
                linewidths=0.5, square=True)
    ax.set_title("Correlation Matrix — Agility & Firm Characteristics",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    save_fig("test2_correlation_matrix.png")

    # Save correlation table
    corr.round(4).to_csv(os.path.join(OUT_FOLDER, "correlation_matrix.csv"))
    print(f"\n  Saved → {OUT_FOLDER}/correlation_matrix.csv")
else:
    print("  Not enough columns for correlation matrix.")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — Agility vs Corporate Governance (Panel B equivalent)
# Needs: board_size, pct_ind_dir in financial CSV
# ═════════════════════════════════════════════════════════════════════════════
print_section("TEST 3: Agility vs Corporate Governance (Lehn 2018/2021)")
print("  Prediction: Smaller board size → higher agility (negative coef)")
print("  Prediction: Fewer independent directors → higher agility (negative coef)")

gov_cols = ["board_size", "pct_ind_dir"]
available_gov = [c for c in gov_cols if c in df.columns]

if available_gov:
    for gov_var in available_gov:
        ols_summary("agility_score", [gov_var, "leverage", "log_assets"],
                    df, f"Agility ~ {gov_var}")

    if "board_size" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = df[["board_size", "agility_score"]].dropna()
        ax.scatter(valid["board_size"], valid["agility_score"],
                   color=COLORS["primary"], alpha=0.8, s=80)
        # Add company labels
        for _, row in valid.iterrows():
            company_short = df.loc[df["agility_score"] == row["agility_score"],
                                    "company"].values[0][:8]
            ax.annotate(company_short, (row["board_size"], row["agility_score"]),
                       fontsize=7, alpha=0.7)
        # Trend line
        z = np.polyfit(valid["board_size"], valid["agility_score"], 1)
        p = np.poly1d(z)
        ax.plot(sorted(valid["board_size"]), p(sorted(valid["board_size"])),
                color=COLORS["secondary"], linestyle="--", linewidth=2)
        ax.set_xlabel("Board Size (number of directors)")
        ax.set_ylabel("Agility Score")
        ax.set_title("Corporate Agility vs Board Size\n(Prediction: negative relationship)",
                     fontsize=12, fontweight="bold")
        save_fig("test3_agility_board_size.png")
else:
    print("\n  ⚠ board_size / pct_ind_dir not found in financial CSV.")
    print("  Add these columns to data/financial/company_financials.csv")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4 — Agility vs Financial Constraints (Panel D equivalent)
# Needs: leverage, roa in financial CSV
# ═════════════════════════════════════════════════════════════════════════════
print_section("TEST 4: Agility vs Financial Constraints")
print("  Prediction: More agile firms have lower leverage (less constrained)")

fin_constraint_cols = [c for c in ["leverage", "roa", "log_assets"] if c in df.columns]
if len(fin_constraint_cols) >= 2:
    ols_summary("agility_score", fin_constraint_cols, df,
                "Agility ~ Financial Characteristics")

    if "leverage" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = df[["leverage", "agility_score"]].dropna()
        ax.scatter(valid["leverage"], valid["agility_score"],
                   color=COLORS["accent"], alpha=0.8, s=80)
        z = np.polyfit(valid["leverage"], valid["agility_score"], 1)
        p = np.poly1d(z)
        ax.plot(sorted(valid["leverage"]), p(sorted(valid["leverage"])),
                color=COLORS["secondary"], linestyle="--", linewidth=2)
        ax.set_xlabel("Leverage (Debt / Total Assets)")
        ax.set_ylabel("Agility Score")
        ax.set_title("Corporate Agility vs Financial Leverage",
                     fontsize=12, fontweight="bold")
        save_fig("test4_agility_leverage.png")
else:
    print("\n  ⚠ leverage / roa not found in financial CSV.")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5 — Agility vs Stock Returns (Panel E equivalent)
# Needs: stock_return in financial CSV
# ═════════════════════════════════════════════════════════════════════════════
print_section("TEST 5: Agility vs Stock Returns")
print("  Prediction: Higher agility → better stock returns")

if "stock_return" in df.columns:
    ols_summary("stock_return", ["agility_score", "leverage", "log_assets"],
                df, "Return ~ Agility + controls")

    fig, ax = plt.subplots(figsize=(8, 5))
    valid = df[["agility_score", "stock_return"]].dropna()
    ax.scatter(valid["agility_score"], valid["stock_return"],
               color=COLORS["primary"], alpha=0.8, s=80)
    z = np.polyfit(valid["agility_score"], valid["stock_return"], 1)
    p = np.poly1d(z)
    ax.plot(sorted(valid["agility_score"]), p(sorted(valid["agility_score"])),
            color=COLORS["secondary"], linestyle="--", linewidth=2)
    ax.set_xlabel("Agility Score")
    ax.set_ylabel("Annual Stock Return (%)")
    ax.set_title("Agility vs Stock Return",
                 fontsize=12, fontweight="bold")
    save_fig("test5_agility_stock_return.png")
else:
    print("\n  ⚠ stock_return not found in financial CSV.")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6 — Sector Analysis (replaces Figure 3 of the paper)
# Needs: sector column in financial CSV
# ═════════════════════════════════════════════════════════════════════════════
print_section("TEST 6: Sector-Level Agility Analysis (Figure 3 equivalent)")

if "sector" in df.columns:
    sector_agility = (df.groupby("sector")["agility_score"]
                        .agg(["mean", "median", "count"])
                        .sort_values("mean", ascending=False))

    print("\n  Agility by Sector:")
    print(sector_agility.round(4).to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = [COLORS["primary"] if i < len(sector_agility) // 2
                  else COLORS["secondary"]
                  for i in range(len(sector_agility))]
    sector_agility["mean"].plot(kind="barh", ax=ax, color=colors_bar,
                                 edgecolor="white", alpha=0.85)
    ax.axvline(df["agility_score"].mean(), color="black",
               linestyle="--", linewidth=1.5, label="Overall mean")
    ax.set_title("Mean Agility Score by Sector\n(Blue = more agile, Red = less agile)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Agility Score")
    ax.legend()
    plt.tight_layout()
    save_fig("test6_sector_agility.png")

    sector_agility.to_csv(os.path.join(OUT_FOLDER, "sector_agility.csv"))
else:
    print("\n  ⚠ sector column not found in financial CSV.")
    print("  Add sector classification to data/financial/company_financials.csv")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7 — Agility Quartile Performance Comparison
# ═════════════════════════════════════════════════════════════════════════════
print_section("TEST 7: Performance by Agility Quartile")

perf_cols = [c for c in ["stock_return", "roa", "sales_growth"] if c in df.columns]

if perf_cols and "agility_quartile" in df.columns:
    quartile_perf = df.groupby("agility_quartile")[perf_cols].mean()
    print("\n  Mean performance metrics by agility quartile:")
    print(quartile_perf.round(4).to_string())

    if len(perf_cols) > 0:
        fig, axes = plt.subplots(1, len(perf_cols),
                                  figsize=(5 * len(perf_cols), 5))
        if len(perf_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, perf_cols):
            vals = [df[df["agility_quartile"] == q][col].mean()
                    for q in ["Q1 (least agile)", "Q2", "Q3", "Q4 (most agile)"]
                    if q in df["agility_quartile"].values]
            qnames = [q for q in ["Q1 (least agile)", "Q2", "Q3", "Q4 (most agile)"]
                      if q in df["agility_quartile"].values]
            bar_colors = [COLORS["secondary"]] + \
                         [COLORS["primary"]] * (len(qnames) - 1)
            ax.bar(range(len(vals)), vals, color=bar_colors, edgecolor="white")
            ax.set_xticks(range(len(qnames)))
            ax.set_xticklabels(qnames, rotation=25, ha="right", fontsize=9)
            ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
            ax.set_ylabel("Mean Value")
        plt.suptitle("Firm Performance by Agility Quartile",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_fig("test7_quartile_performance.png")
else:
    print("\n  ⚠ No performance columns (stock_return/roa/sales_growth) found.")
    print("  Showing agility quartile breakdown:")
    if "agility_quartile" in df.columns:
        for q in df["agility_quartile"].cat.categories:
            companies = df[df["agility_quartile"] == q]["company"].tolist()
            print(f"  {q}: {companies}")


# ─────────────────────────────────────────────────────────────────────────────
# Save complete merged dataset for further analysis
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv(os.path.join(OUT_FOLDER, "full_dataset.csv"), index=False)
print(f"\n  Full dataset saved → {OUT_FOLDER}/full_dataset.csv")

print("\n" + "=" * 65)
print("ALL VALIDATION TESTS COMPLETE")
print("=" * 65)
print(f"\nFigures saved in : {FIGURES_FOLDER}/")
print(f"Tables saved in  : {OUT_FOLDER}/")

if not has_financial:
    print("\n" + "!" * 65)
    print("  TO RUN FULL VALIDATION — CREATE THIS FILE:")
    print(f"  {FINANCIAL_CSV}")
    print()
    print("  Required columns (one row per company):")
    print("  company, board_size, pct_ind_dir, leverage, log_assets,")
    print("  roa, sales_growth, stock_return, sector")
    print()
    print("  Data sources:")
    print("  - Screener.in (free) → financials + ratios")
    print("  - BSE annual reports → board composition")
    print("  - NSE/Yahoo Finance → stock returns")
    print("!" * 65)
    