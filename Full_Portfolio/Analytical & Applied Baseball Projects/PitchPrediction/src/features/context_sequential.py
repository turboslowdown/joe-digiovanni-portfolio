import pandas as pd
import numpy as np
from pathlib import Path

def get_project_root():
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / 'data').exists():
            return parent
    return path.parent.parent

def create_context_features(df):
    df = df.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])

    # standard context
    df['on_1b'] = df['on_1b'].notna().astype(int)
    df['on_2b'] = df['on_2b'].notna().astype(int)
    df['on_3b'] = df['on_3b'].notna().astype(int)
    df['score_diff'] = df['bat_score'] - df['fld_score']
    df['platoon_advantage'] = (df['p_throws'] != df['stand']).astype(int)

    # times through order
    df['matchup_id'] = df['game_pk'].astype(str) + '_' + df['pitcher'].astype(str) + '_' + df['batter'].astype(str)
    df['times_faced'] = df.groupby(['matchup_id'])['at_bat_number'].transform(lambda x: x.factorize()[0] + 1)
    df = df.drop(columns=['matchup_id'])

    # outcome logic
    swing_descs = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt', 'foul', 'foul_tip', 'foul_bunt', 'in_play_hit_foul', 'in_play_no_out', 'in_play_out']
    whiff_descs = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt']
    df['is_swing'] = df['description'].isin(swing_descs).astype(int)
    df['is_whiff'] = df['description'].isin(whiff_descs).astype(int)

    conditions = [
        df['type'] == 'B',
        (df['type'] == 'S') & (df['is_swing'] == 0),
        df['description'].str.contains('foul'),
        df['type'] == 'X',
        df['is_whiff'] == 1
    ]
    choices = ['Ball', 'CalledStrike', 'Foul', 'InPlay', 'Whiff']
    df['pitch_result_code'] = np.select(conditions, choices, default='Unknown')

    df['is_outside'] = ((df['plate_x'].abs() > 0.833) | (df['plate_z'] > df['sz_top']) | (df['plate_z'] < df['sz_bot'])).astype(int)
    df['is_chase'] = ((df['is_outside'] == 1) & (df['is_swing'] == 1)).astype(int)
    
    # granular count features
    df['is_two_strikes'] = (df['strikes'] == 2).astype(int)
    df['is_full_count'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)
    
    # 0-0 count is a distinct pitching state
    df['is_first_pitch'] = ((df['balls'] == 0) & (df['strikes'] == 0)).astype(int)
    
    return df

def create_sequence_features(df):
    group_cols = ['game_pk', 'at_bat_number']
    
    common_pitches = ['FF', 'SL', 'SI', 'CH', 'FC', 'ST', 'CU', 'FS']
    df['pitch_type_clean'] = df['pitch_type'].where(df['pitch_type'].isin(common_pitches), 'Other')
    
    for i in range(1, 4):
        df[f'prev_pitch_{i}'] = df.groupby(group_cols)['pitch_type_clean'].shift(i).fillna('None')
        df[f'prev_result_{i}'] = df.groupby(group_cols)['pitch_result_code'].shift(i).fillna('None')
        df[f'prev_zone_{i}'] = df.groupby(group_cols)['zone'].shift(i).fillna(0).astype(int).astype(str)
        df[f'prev_velo_{i}'] = df.groupby(group_cols)['release_speed'].shift(i).fillna(0)
        
    df['is_back_to_back'] = (df['prev_pitch_1'] == df['prev_pitch_2']).astype(int)
    df.loc[df['pitch_number'] <= 2, 'is_back_to_back'] = 0
    df = df.drop(columns=['pitch_type_clean'])
    
    return df

def main():
    root = get_project_root()
    input_path = root / 'data' / 'processed' / 'pitch_data_base.parquet'
    output_path = root / 'data' / 'processed' / 'pitch_data_context.parquet'
    
    if not input_path.exists():
        print("Input file not found.")
        return

    df = pd.read_parquet(input_path)
    df = create_context_features(df)
    df = create_sequence_features(df)
    df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    main()