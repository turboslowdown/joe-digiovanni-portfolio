import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

#set aesthetic style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'

def get_project_root():
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / 'data').exists():
            return parent
    return path.parent.parent

def prepare_data(df):
    top_pitches = ['FF', 'SL', 'SI', 'CH', 'FC', 'ST', 'CU', 'FS']
    df = df[df['pitch_type'].isin(top_pitches)].copy()
    
    target_encoder = LabelEncoder()
    df['target'] = target_encoder.fit_transform(df['pitch_type'])
    
    context_cols = [
        'balls', 'strikes', 'outs_when_up', 'inning', 'on_1b', 'on_2b', 'on_3b', 
        'score_diff', 'at_bat_number', 'pitch_number', 'platoon_advantage',
        'times_faced', 'is_two_strikes', 'is_full_count', 'is_first_pitch'
    ]
    
    id_cols = ['pitcher', 'batter', 'fielder_2']
    
    catcher_usage_cols = [c for c in df.columns if 'catcher_usage_' in c]
    pitcher_usage_cols = [c for c in df.columns if 'usage_global_' in c or 'usage_situation_' in c or 'usage_after_prev_' in c or 'matchup_usage_' in c]
    whiff_cols = [c for c in df.columns if 'whiff_rate_' in c or 'batter_whiff_' in c]
    
    tendency_cols = pitcher_usage_cols + catcher_usage_cols + whiff_cols + [
        'season_pitch_count', 'game_pitch_count', 
        'pitcher_chase_rate', 'batter_chase_rate', 'batter_season_pa'
    ]
    
    seq_cols = [
        'prev_pitch_1', 'prev_pitch_2', 'prev_pitch_3', 
        'prev_result_1', 'prev_result_2', 'prev_result_3',
        'prev_zone_1', 'prev_zone_2', 'prev_zone_3',
        'prev_velo_1', 'prev_velo_2', 'prev_velo_3', 'is_back_to_back'
    ]
    
    feature_cols = context_cols + id_cols + tendency_cols + seq_cols
    
    cat_features = [
        'pitcher', 'batter', 'fielder_2',
        'prev_pitch_1', 'prev_pitch_2', 'prev_pitch_3',
        'prev_result_1', 'prev_result_2', 'prev_result_3',
        'prev_zone_1', 'prev_zone_2', 'prev_zone_3'
    ]
    
    for col in cat_features:
        df[col] = df[col].fillna('None').astype(str).astype('category')
        
    return df, feature_cols, cat_features, target_encoder

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    """
    shows where the model gets confused. Normalized by row (Recall).
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Proportion of Actual Pitches'})
    
    plt.title('Confusion Matrix (Normalized by Actual Pitch)', fontsize=16, pad=20)
    plt.xlabel('Predicted Pitch', fontsize=12)
    plt.ylabel('Actual Pitch', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    print("Saved: confusion_matrix.png")

def plot_feature_importance(model, features, output_dir):
    """
    horizontal bar chart of top decision drivers
    """
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=importance, palette='viridis')
    
    plt.title('Top 20 Model Decision Drivers (Gain)', fontsize=16, pad=20)
    plt.xlabel('Importance Score (Gain)', fontsize=12)
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300)
    print("Saved: feature_importance.png")

def plot_model_confidence(y_true, preds_prob, output_dir):
    """
    histogram of how confident the model is when it is Right vs Wrong
    """
    # Get max probability (confidence) for each prediction
    confidence = np.max(preds_prob, axis=1)
    preds_class = np.argmax(preds_prob, axis=1)
    is_correct = (y_true == preds_class)
    
    df_conf = pd.DataFrame({'Confidence': confidence, 'Correct': is_correct})
    df_conf['Outcome'] = df_conf['Correct'].map({True: 'Correct Prediction', False: 'Incorrect Prediction'})
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_conf, x='Confidence', hue='Outcome', 
                 bins=30, kde=True, palette={'Correct Prediction': '#2ecc71', 'Incorrect Prediction': '#e74c3c'},
                 element='step', stat='density', common_norm=False)
    
    plt.title('Model Confidence Distribution', fontsize=16, pad=20)
    plt.xlabel('Predicted Probability (Confidence)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(0, 1)
    plt.legend(title='Outcome', loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_dist.png', dpi=300)
    print("Saved: confidence_dist.png")

def main():
    root = get_project_root()
    data_path = root / 'data' / 'processed' / 'pitch_data_features.parquet'
    output_dir = root / 'src' / 'visualization' / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_parquet(data_path)
    df, features, cat_features, encoder = prepare_data(df)
    
    # Temporal Split (Same as train_model)
    df = df.sort_values('game_date')
    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.8)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    X_train, y_train = train_df[features], train_df['target']
    X_val, y_val = val_df[features], val_df['target']
    X_test, y_test = test_df[features], test_df['target']
    
    #re-train for plotting
    print("Training model for visualization...")
    params = {
        'objective': 'multiclass',
        'num_class': len(encoder.classes_),
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_features)
    
    #mini model for plotting
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500, 
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )
    
    print("Generating predictions...")
    preds_prob = model.predict(X_test)
    preds_class = np.argmax(preds_prob, axis=1)
    
    print("Creating plots...")
    plot_confusion_matrix(y_test, preds_class, encoder.classes_, output_dir)
    plot_feature_importance(model, features, output_dir)
    plot_model_confidence(y_test, preds_prob, output_dir)
    
    print(f"\n✅ Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()