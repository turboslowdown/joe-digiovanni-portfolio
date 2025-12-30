#imports
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pybaseball import statcast
from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parent.parent

def setup_directories():
    root = get_project_root()
    dirs = [
        root / 'data' / 'raw',
        root / 'data' / 'processed',
        root / 'src' / 'features',
        root / 'models'
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def fetch_year(year):
    start_date = f"{year}-03-28"
    end_date = f"{year}-11-01"
    
    try:
        data = statcast(start_dt=start_date, end_dt=end_date)
        return data
    except Exception:
        return None

#remove pitchers with less than 100 pitches and less than 90 mph (postition players)
def filter_pitchers(df, min_pitches=100, min_vclo=90):
    stats = df.groupby(['pitcher', 'game_year']).agg(
        n_pitches=('release_speed', 'count'),
        max_v=('release_speed', 'max')
    ).reset_index()
    
    valid_ids = stats[(stats['n_pitches'] >= min_pitches) & (stats['max_v'] >= min_vclo)]
    return df.merge(valid_ids[['pitcher', 'game_year']], on=['pitcher', 'game_year'], how='inner')

def main():
    setup_directories()
    root = get_project_root()
    
    years = [2024] 
    collected = []
    
    for y in years:
        raw_data = fetch_year(y)
        if raw_data is not None:
            raw_data = raw_data.dropna(subset=['pitch_type', 'release_speed'])
            filtered_data = filter_pitchers(raw_data)
            
            output_path = root / 'data' / 'raw' / f'statcast_{y}.parquet'
            filtered_data.to_parquet(output_path, index=False)
            collected.append(filtered_data)

    if collected:
        full_df = pd.concat(collected)
        processed_path = root / 'data' / 'processed' / 'pitch_data_base.parquet'
        full_df.to_parquet(processed_path, index=False)

if __name__ == "__main__":
    main()