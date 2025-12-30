import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

def get_project_root():
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / 'data').exists():
            return parent
    return path.parent.parent

#prepare data for training
def prepare_data(df):
    #filter to top 8 pitches
    top_pitches = ['FF', 'SL', 'SI', 'CH', 'FC', 'ST', 'CU', 'FS']
    df = df[df['pitch_type'].isin(top_pitches)].copy()
    
    target_encoder = LabelEncoder()
    df['target'] = target_encoder.fit_transform(df['pitch_type'])
    
    # context features
    context_cols = [
        'balls', 'strikes', 'outs_when_up', 'inning', 'on_1b', 'on_2b', 'on_3b', 
        'score_diff', 'at_bat_number', 'pitch_number', 'platoon_advantage',
        'times_faced', 'is_two_strikes', 'is_full_count', 'is_first_pitch'
    ]
    
    # ids & usage
    id_cols = ['pitcher', 'batter', 'fielder_2']
    
    catcher_usage_cols = [c for c in df.columns if 'catcher_usage_' in c]
    
    pitcher_usage_cols = [
        c for c in df.columns 
        if 'usage_global_' in c 
        or 'usage_situation_' in c 
        or 'usage_after_prev_' in c
        or 'matchup_usage_' in c
    ]
    
    #whiff rate features
    whiff_cols = [c for c in df.columns if 'whiff_rate_' in c or 'batter_whiff_' in c]
    
    #tendency features
    tendency_cols = pitcher_usage_cols + catcher_usage_cols + whiff_cols + [
        'season_pitch_count', 'game_pitch_count', 
        'pitcher_chase_rate', 'batter_chase_rate', 'batter_season_pa'
    ]
    
    #sequence features
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

# check robustness and generalizability
def comprehensive_validation(model, test_df, y_test, preds_prob, encoder):
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("="*60)
    
    print("\n[1] Top-K Accuracy:")
    for k in [1, 2, 3]:
        top_k_preds = np.argsort(preds_prob, axis=1)[:, -k:]
        correct = np.array([y_test.iloc[i] in top_k_preds[i] for i in range(len(y_test))])
        print(f"  Top-{k}: {correct.mean()*100:.2f}%")
        
    print("\n[2] Performance by Count:")
    test_df = test_df.copy()
    test_df['count_str'] = test_df['balls'].astype(str) + '-' + test_df['strikes'].astype(str)
    
    for count in ['0-0', '0-2', '1-2', '2-2', '3-2']:
        mask = test_df['count_str'] == count
        if mask.sum() > 50:
            subset_loss = log_loss(y_test[mask], preds_prob[mask])
            print(f"  {count}: {subset_loss:.4f} (n={mask.sum()})")

    print("\n[3] Calibration Check (Correlations):")
    for i, pitch in enumerate(encoder.classes_):
        y_binary = (y_test == i).astype(int)
        pred_binary = preds_prob[:, i]
        if y_binary.sum() > 50:
            corr = np.corrcoef(y_binary, pred_binary)[0,1]
            print(f"  {pitch}: {corr:.3f}")

def train():
    root = get_project_root()
    data_path = root / 'data' / 'processed' / 'pitch_data_features.parquet'
    
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        return

    df = pd.read_parquet(data_path)
    df, features, cat_features, encoder = prepare_data(df)
    
    # temporal split
    df = df.sort_values('game_date')
    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.8)
    
    # split data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Split Sizes | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    X_train, y_train = train_df[features], train_df['target']
    X_val, y_val = val_df[features], val_df['target']
    X_test, y_test = test_df[features], test_df['target']
    
    # training parameters
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
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    preds_prob = model.predict(X_test)
    print(f"\nFINAL TEST LOG LOSS: {log_loss(y_test, preds_prob):.4f}")
    
    comprehensive_validation(model, test_df, y_test, preds_prob, encoder)
    
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Decision Drivers:")
    print(importance.head(15))

if __name__ == "__main__":
    train()