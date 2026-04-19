"""
FIFA 22 Player Market Value Predictor
======================================
INST414 – Module 6: Supervised Learning
Run this script in the same folder as players_22.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # remove this line if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('players_22.csv', low_memory=False)
print(f"Raw dataset: {df.shape[0]:,} players, {df.shape[1]} columns")

# ── 2. CLEAN DATA ─────────────────────────────────────────────────────────────
print("\nCleaning...")

# Keep only players with a valid market value (free agents have 0)
df = df[df['value_eur'].notna() & (df['value_eur'] > 0)].copy()
print(f"After dropping zero/null values: {len(df):,} players")

# Keep only the columns we need
KEEP_COLS = [
    'short_name', 'age', 'overall', 'potential',
    'value_eur', 'wage_eur',
    'player_positions',          # e.g. "ST, CF"
    'league_name',
    'nationality_name',
    'international_reputation',  # 1–5 stars
    'weak_foot',                 # 1–5 stars
    'skill_moves',               # 1–5 stars
    'work_rate',                 # e.g. "High/Medium"
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'height_cm', 'weight_kg',
]
# Some columns may be named differently — this handles it gracefully
KEEP_COLS = [c for c in KEEP_COLS if c in df.columns]
df = df[KEEP_COLS].copy()

# Extract primary position (first listed)
if 'player_positions' in df.columns:
    df['position'] = df['player_positions'].str.split(',').str[0].str.strip()
else:
    df['position'] = 'Unknown'

# Split work_rate into attack / defense
if 'work_rate' in df.columns:
    df['wr_att'] = df['work_rate'].str.split('/').str[0].str.strip()
    df['wr_def'] = df['work_rate'].str.split('/').str[1].str.strip()
else:
    df['wr_att'] = 'Medium'
    df['wr_def'] = 'Medium'

# Drop rows still missing key fields
REQUIRED = ['age', 'overall', 'potential', 'value_eur',
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
df.dropna(subset=REQUIRED, inplace=True)
df.reset_index(drop=True, inplace=True)   # ← fix: keeps iloc aligned with X_test.index
print(f"After dropping rows with missing key stats: {len(df):,} players")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("\nEngineering features...")

df['log_value']      = np.log1p(df['value_eur'])       # log-scale target
df['age_sq']         = df['age'] ** 2                  # age curve
df['potential_gap']  = df['potential'] - df['overall'] # growth headroom
df['overall_x_rep']  = df['overall'] * df['international_reputation']
df['att_ability']    = (df['pace'] + df['shooting'] + df['dribbling']) / 3
df['def_ability']    = (df['defending'] + df['physic']) / 2

# Encode categoricals
for col in ['position', 'league_name', 'nationality_name', 'wr_att', 'wr_def']:
    if col in df.columns:
        df[col + '_enc'] = LabelEncoder().fit_transform(df[col].astype(str))
    else:
        df[col + '_enc'] = 0

FEATURES = [
    'age', 'age_sq', 'overall', 'potential', 'potential_gap',
    'international_reputation', 'weak_foot', 'skill_moves',
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'height_cm', 'weight_kg',
    'overall_x_rep', 'att_ability', 'def_ability',
    'position_enc', 'league_name_enc', 'nationality_name_enc',
    'wr_att_enc', 'wr_def_enc'
]
FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES]
y = df['log_value']

# ── 4. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 5. TRAIN RANDOM FOREST ───────────────────────────────────────────────────
print("\nTraining Random Forest... (this takes ~30 seconds)")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print("Done.")

# ── 6. EVALUATE ───────────────────────────────────────────────────────────────
y_pred_log = rf.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_true     = np.expm1(y_test)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
cv   = cross_val_score(rf, X, y, cv=5, scoring='r2', n_jobs=-1)

print("\n── Model Performance ──────────────────────────────")
print(f"  Test  R²  : {r2:.4f}")
print(f"  Test  MAE : €{mae:>15,.0f}")
print(f"  Test  RMSE: €{rmse:>15,.0f}")
print(f"  CV R² (5-fold): {cv.mean():.4f} ± {cv.std():.4f}")
print("───────────────────────────────────────────────────")

# ── 7. FEATURE IMPORTANCES ────────────────────────────────────────────────────
fi = (pd.DataFrame({'feature': FEATURES, 'importance': rf.feature_importances_})
        .sort_values('importance', ascending=False))
print("\n── Top 10 Feature Importances ─────────────────────")
print(fi.head(10).to_string(index=False))

# ── 8. WORST 5 PREDICTIONS ───────────────────────────────────────────────────
results = df.iloc[X_test.index].copy()
results['pred_value'] = y_pred
results['true_value'] = y_true
results['abs_error']  = (y_true - y_pred).abs()
results['pct_error']  = results['abs_error'] / y_true * 100

worst5_cols = ['short_name', 'age', 'position', 'overall',
               'international_reputation', 'league_name',
               'true_value', 'pred_value', 'pct_error']
worst5_cols = [c for c in worst5_cols if c in results.columns]
worst5 = results.nlargest(5, 'abs_error')[worst5_cols]

print("\n── 5 Samples the Model Got Most Wrong ─────────────")
pd.set_option('display.float_format', '€{:,.0f}'.format)
print(worst5.to_string(index=False))
pd.reset_option('display.float_format')

# Save worst5 to CSV so you can inspect it easily
worst5.to_csv('worst5_predictions.csv', index=False)
print("\nSaved: worst5_predictions.csv")

# ── 9. FIGURES ────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

# Figure 1 — Feature Importances
fig, ax = plt.subplots(figsize=(10, 6))
top10  = fi.head(10)
colors = ['#1d4ed8' if i < 3 else '#60a5fa' for i in range(10)]
ax.barh(top10['feature'][::-1], top10['importance'][::-1], color=colors[::-1])
ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)', fontsize=12)
ax.set_title('Top 10 Features — FIFA 22 Player Market Value Prediction',
             fontsize=13, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
for i, (val, name) in enumerate(zip(top10['importance'][::-1],
                                     top10['feature'][::-1])):
    ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('fig1_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig1_feature_importance.png")

# Figure 2 — Predicted vs Actual
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_true / 1e6, y_pred / 1e6,
           alpha=0.25, s=20, c='#2563eb', edgecolors='none')
mx = max(y_true.max(), y_pred.max()) / 1e6
ax.plot([0, mx], [0, mx], 'r--', lw=1.5, label='Perfect prediction')
ax.set_xlabel('Actual Market Value (€M)', fontsize=12)
ax.set_ylabel('Predicted Market Value (€M)', fontsize=12)
ax.set_title('Predicted vs. Actual Player Market Value', fontsize=13, fontweight='bold')
ax.text(0.05, 0.92, f'R² = {r2:.3f}', transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#eff6ff', edgecolor='#93c5fd'))
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('fig2_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig2_pred_vs_actual.png")

# Figure 3 — Market Value by League (top 10 leagues by median value)
top_leagues = (df.groupby('league_name')['value_eur']
                 .median()
                 .nlargest(10)
                 .index)
df_top = df[df['league_name'].isin(top_leagues)]
order  = (df_top.groupby('league_name')['value_eur']
                .median()
                .sort_values(ascending=False)
                .index)

fig, ax = plt.subplots(figsize=(11, 6))
sns.boxplot(data=df_top, x='league_name', y='value_eur',
            order=order, palette='Blues_r', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
ax.set_xlabel('League', fontsize=12)
ax.set_ylabel('Market Value (€)', fontsize=12)
ax.set_title('Player Market Value by Top 10 Leagues', fontsize=13, fontweight='bold')
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f'€{x/1e6:.0f}M'))
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('fig3_value_by_league.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_value_by_league.png")

# Figure 4 — Overall Rating vs Value (coloured by Age)
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(df['overall'], df['value_eur'] / 1e6,
                c=df['age'], cmap='RdYlGn_r',
                alpha=0.3, s=15, edgecolors='none')
cb = plt.colorbar(sc, ax=ax)
cb.set_label('Age', fontsize=10)
ax.set_xlabel('Overall Rating', fontsize=12)
ax.set_ylabel('Market Value (€M)', fontsize=12)
ax.set_title('Overall Rating vs. Market Value (coloured by Age)',
             fontsize=13, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('fig4_overall_vs_value.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_overall_vs_value.png")

print("\n✅ All done! Check your folder for the 4 figures and worst5_predictions.csv")
print("\nCopy-paste the numbers below into the next step:")
print(f"  R²   = {r2:.4f}")
print(f"  MAE  = €{mae:,.0f}")
print(f"  RMSE = €{rmse:,.0f}")
print(f"  CV R² = {cv.mean():.4f} ± {cv.std():.4f}")