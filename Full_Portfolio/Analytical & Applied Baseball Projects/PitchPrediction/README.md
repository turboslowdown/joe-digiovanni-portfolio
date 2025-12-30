# MLB Pitch Type Prediction

Multiclass classification model predicting the next pitch type in an MLB at-bat using Statcast data and LightGBM. Focuses on pitcher-batter matchups, game context, and pitch sequencing rather than aggregate player statistics.

## Performance

- **Test Log Loss:** 1.16 (baseline: 2.08 uniform, ~0.9 majority class)
- **Top-1 Accuracy:** 48%
- **Top-3 Accuracy:** 91%
- **Train/Val Gap:** 0.15 (1.00 train, 1.16 validation)

Temporal validation split: 70% train, 10% validation, 20% test (chronological by game date).

## Methodology

### Feature Engineering

**Context Features:**
- Count state (balls, strikes, two-strikes, full count flags)
- Game state (inning, base runners, score differential, times through order)
- Platoon advantage

**Tendency Features:**
- Pitcher usage rates: global, situational (by count state), and transition (after previous pitch type)
- Pitcher-batter matchup history (usage and whiff rates)
- Catcher usage tendencies (`fielder_2`)
- Batter whiff rates by pitch type and vertical zone (high/low)
- All historical features use expanding windows with `shift(1)` to prevent leakage

**Sequence Features:**
- Previous 3 pitches: type, result, zone, velocity
- Back-to-back pitch indicator

**Categorical Features:**
- Pitcher ID, batter ID, catcher ID
- Previous pitch types, results, zones (encoded as categories)

### Model

LightGBM gradient boosting with:
- Multiclass objective (8 pitch types: FF, SL, SI, CH, FC, ST, CU, FS)
- Regularization: `learning_rate=0.03`, `num_leaves=31`, `feature_fraction=0.7`, L1/L2 penalties
- Early stopping on validation set (50 rounds)
- Categorical feature handling for IDs and sequence features

### Validation

- Temporal split (no future data leakage)
- Top-K accuracy by count situation
- Calibration checks (predicted vs observed probabilities)
- Performance breakdown by game state

## Project Structure

```
├── data/
│   ├── raw/              # Statcast parquet files
│   └── processed/        # Feature-engineered datasets
├── src/
│   ├── features/
│   │   ├── context_sequential.py  # Game state & pitch sequences
│   │   └── tendencies.py          # Rolling usage rates & matchups
│   ├── visualization/
│   │   ├── make_plots.py          # Confusion matrices, feature importance
│   │   └── output/
│   ├── ingestion.py               # Data fetch & filtering
│   └── eda.py                     # Exploratory analysis
└── models/
    └── train_model.py             # Training & validation pipeline
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch Statcast data
python src/ingestion.py

# Generate features (run in order)
python src/features/context_sequential.py
python src/features/tendencies.py

# Train model
python models/train_model.py

# Generate visualizations
python src/visualization/make_plots.py
```

## Key Insights

1. **Situational usage > global usage**: Count-specific features (`usage_situation_*`) consistently outperform global rates, indicating pitchers adjust strategy by count.

2. **Catcher signal is meaningful**: Catcher ID and usage tendencies rank in top 20 features, confirming pitch-calling is collaborative.

3. **Fastball as default**: Model correctly identifies fastball as the base pitch type in neutral counts. Class balancing degraded performance.

4. **Pitcher/batter IDs dominate**: Raw IDs are top features, suggesting strong individual tendencies that aren't fully captured by aggregated statistics.

## Technical Stack

- Python 3.10+
- LightGBM (gradient boosting)
- pandas, numpy (data processing)
- pybaseball (Statcast API)
- scikit-learn (metrics)

## Future Work

- Stuff metrics for fatigue detection
- Umpire-specific strike zone features
- Adversarial modeling (batter expectation)