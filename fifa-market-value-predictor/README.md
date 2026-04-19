# FIFA 22 Player Market Value Predictor

**INST414 – Data Science Techniques | Module 6: Supervised Learning**

## Question
Can a player's measurable on-pitch attributes reliably predict their transfer market value — and which attributes matter most?

## Dataset
[FIFA 22 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset) via Kaggle  
- 19,239 players, 110 columns  
- Target variable: `value_eur` (transfer market value in euros)  
- After cleaning: 17,041 players used for modeling

## Model
**Random Forest Regressor** (scikit-learn)  
- 200 trees, max depth 12, min samples per leaf 3  
- Target log-transformed (`log1p`) to handle right skew  
- 80/20 train/test split, 5-fold cross-validation

## Results

| Metric | Score |
|--------|-------|
| Test R² | 0.9868 |
| Test MAE | €153,149 |
| Test RMSE | €964,043 |
| CV R² (5-fold) | 0.31 ± 0.59 |

## Top Features
| Feature | Importance |
|---------|------------|
| overall | 0.714 |
| potential | 0.149 |
| overall × reputation | 0.100 |
| age² | 0.017 |
| age | 0.015 |

## Files
| File | Description |
|------|-------------|
| `fifa_analysis.py` | Full analysis script — run this to reproduce all results |
| `worst5_predictions.csv` | The 5 players the model got most wrong |
| `fig1_feature_importance.png` | Top 10 feature importances |
| `fig2_pred_vs_actual.png` | Predicted vs. actual values scatter plot |
| `fig3_value_by_league.png` | Market value distribution by league |
| `fig4_overall_vs_value.png` | Overall rating vs. value colored by age |

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python fifa_analysis.py
```
Place `players_22.csv` (from Kaggle) in the same folder before running.

## Key Finding
`overall` rating alone drives 71% of the model's predictions — but the CV score (0.31 ± 0.59) reveals the model overfits to EA Sports' own rating system, which partially circular with market value. Non-stat factors like brand value, contract length, and transfer timing explain the largest errors.
