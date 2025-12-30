import pandas as pd
import numpy as np
from pathlib import Path

def get_project_root():
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / 'data').exists():
            return parent
    return path.parent.parent

def calculate_pitcher_tendencies(df):
    df = df.sort_values(['pitcher', 'game_date', 'at_bat_number', 'pitch_number'])
    df['season_pitch_count'] = df.groupby(['pitcher', 'game_year']).cumcount()
    df['game_pitch_count'] = df.groupby(['pitcher', 'game_pk']).cumcount()
    
    conditions = [df['strikes'] == 2, df['balls'] > df['strikes']]
    choices = ['two_strikes', 'behind']
    df['count_state'] = np.select(conditions, choices, default='neutral_ahead')

    top_pitches = ['FF', 'SL', 'SI', 'CH', 'FC', 'ST', 'CU', 'FS']
    for p in top_pitches:
        df[f'is_{p}'] = (df['pitch_type'] == p).astype(int)

    df['prev_pitch_raw'] = df.groupby(['pitcher', 'game_year'])['pitch_type'].shift(1).fillna('None')

    print("  > Calculating Pitcher Tendencies...")
    for p_type in top_pitches:
        # global usage
        df[f'usage_global_{p_type}'] = (
            df.groupby(['pitcher', 'game_year'])[f'is_{p_type}']
            .transform(lambda x: x.shift(1).expanding().mean())
        ).fillna(0)

        # situational usage
        df[f'usage_situation_{p_type}'] = (
            df.groupby(['pitcher', 'game_year', 'count_state'])[f'is_{p_type}']
            .transform(lambda x: x.shift(1).expanding().mean())
        ).fillna(0)
        
        # transition usage
        df[f'usage_after_prev_{p_type}'] = (
            df.groupby(['pitcher', 'game_year', 'prev_pitch_raw'])[f'is_{p_type}']
            .transform(lambda x: x.shift(1).expanding().mean())
        ).fillna(0)

        # pivoted whiffs
        is_p_type = df['pitch_type'] == p_type
        df[f'temp_whiff_{p_type}'] = np.where(is_p_type, df['is_whiff'], np.nan)
        df[f'whiff_rate_{p_type}'] = (
            df.groupby(['pitcher', 'game_year'])[f'temp_whiff_{p_type}']
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        df[f'whiff_rate_{p_type}'] = df.groupby(['pitcher', 'game_year'])[f'whiff_rate_{p_type}'].ffill().fillna(0)
        df = df.drop(columns=[f'temp_whiff_{p_type}'])

    outside_mask = df['is_outside'] == 1
    df.loc[outside_mask, 'temp_chase'] = (
        df[outside_mask].groupby('pitcher')['is_chase']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df['pitcher_chase_rate'] = df.groupby('pitcher')['temp_chase'].ffill().fillna(0)
    
    drop_cols = [f'is_{p}' for p in top_pitches] + ['temp_chase', 'prev_pitch_raw']
    df = df.drop(columns=drop_cols)
    return df

def calculate_catcher_tendencies(df):
    df = df.sort_values(['fielder_2', 'game_date', 'at_bat_number', 'pitch_number'])
    top_pitches = ['FF', 'SL', 'SI', 'CH', 'FC', 'ST', 'CU', 'FS']
    for p in top_pitches:
        df[f'is_{p}'] = (df['pitch_type'] == p).astype(int)
        
    print("  > Calculating Catcher Tendencies...")
    for p_type in top_pitches:
        df[f'catcher_usage_{p_type}'] = (
            df.groupby(['fielder_2', 'game_year'])[f'is_{p_type}']
            .transform(lambda x: x.shift(1).expanding().mean())
        ).fillna(0)
    df = df.drop(columns=[f'is_{p}' for p in top_pitches])
    return df

def calculate_matchup_features(df):
    # specific pitcher-batter history
    print("  > Calculating Matchup History...")
    df = df.sort_values(['pitcher', 'batter', 'game_date', 'at_bat_number', 'pitch_number'])
    top_pitches = ['FF', 'SL', 'SI', 'CH', 'FC', 'ST', 'CU', 'FS']
    
    for p in top_pitches:
        df[f'is_{p}'] = (df['pitch_type'] == p).astype(int)
        
    for p_type in top_pitches:
        # matchup usage (what does THIS pitcher throw THIS batter?)
        df[f'matchup_usage_{p_type}'] = (
            df.groupby(['pitcher', 'batter'])[f'is_{p_type}']
            .transform(lambda x: x.shift(1).expanding().mean())
        ).fillna(0)
    
    df = df.drop(columns=[f'is_{p}' for p in top_pitches])
    return df

def calculate_batter_tendencies(df):
    df = df.sort_values(['batter', 'game_date', 'at_bat_number', 'pitch_number'])
    top_pitches = ['FF', 'SL', 'SI', 'CH', 'FC', 'ST', 'CU', 'FS']
    
    print("  > Calculating Batter Stats...")
    for p_type in top_pitches:
        is_p_type = df['pitch_type'] == p_type
        df[f'temp_bat_whiff_{p_type}'] = np.where(is_p_type, df['is_whiff'], np.nan)
        df[f'batter_whiff_{p_type}'] = (
            df.groupby(['batter', 'p_throws'])[f'temp_bat_whiff_{p_type}']
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        df[f'batter_whiff_{p_type}'] = df.groupby(['batter', 'p_throws'])[f'batter_whiff_{p_type}'].ffill().fillna(0)
        df = df.drop(columns=[f'temp_bat_whiff_{p_type}'])
    
    # macro-zone features
    df['is_high'] = (df['plate_z'] > 2.5).astype(int)
    
    high_mask = df['is_high'] == 1
    df.loc[high_mask, 'temp_high_whiff'] = (
        df[high_mask].groupby('batter')['is_whiff']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df['batter_whiff_high'] = df.groupby('batter')['temp_high_whiff'].ffill().fillna(0)
    
    low_mask = df['is_high'] == 0
    df.loc[low_mask, 'temp_low_whiff'] = (
        df[low_mask].groupby('batter')['is_whiff']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df['batter_whiff_low'] = df.groupby('batter')['temp_low_whiff'].ffill().fillna(0)
    
    # Chase
    outside_mask = df['is_outside'] == 1
    df.loc[outside_mask, 'temp_chase'] = (
        df[outside_mask].groupby('batter')['is_chase']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df['batter_chase_rate'] = df.groupby('batter')['temp_chase'].ffill().fillna(0)
    
    df['batter_pa_count'] = df.groupby(['batter', 'game_year', 'game_pk', 'at_bat_number']).ngroup()
    df['batter_season_pa'] = df.groupby(['batter', 'game_year'])['batter_pa_count'].transform(lambda x: x.factorize()[0])
    
    drop_cols = ['temp_chase', 'temp_high_whiff', 'temp_low_whiff', 'is_high']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

def main():
    root = get_project_root()
    input_path = root / 'data' / 'processed' / 'pitch_data_context.parquet'
    output_path = root / 'data' / 'processed' / 'pitch_data_features.parquet'
    
    if not input_path.exists():
        print("Input file not found.")
        return

    df = pd.read_parquet(input_path)
    
    df = calculate_pitcher_tendencies(df)
    df = calculate_catcher_tendencies(df)
    df = calculate_matchup_features(df)
    df = calculate_batter_tendencies(df)
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('None').astype(str)
    
    df.to_parquet(output_path, index=False)
    print("Features Updated.")

if __name__ == "__main__":
    main()