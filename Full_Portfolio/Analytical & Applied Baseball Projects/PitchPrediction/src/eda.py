import pandas as pd
from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parent.parent

#check for data file and run basic eda
def run_basic_eda():
    root = get_project_root()
    data_path = root / 'data' / 'processed' / 'pitch_data_base.parquet'
    
    if not data_path.exists():
        print("Data file not found. Run ingestion first.")
        return

    df = pd.read_parquet(data_path)
    
    print(" Dataset Overview ")
    print(f"Total Pitches: {len(df)}")
    print(f"Unique Pitchers: {df['pitcher'].nunique()}")
    print(f"Unique Batters: {df['batter'].nunique()}")
    
    print("\n Pitch Type Distribution ")
    print(df['pitch_type'].value_counts(normalize=True).round(4) * 100)
    
    print("\n Velocity Stats (Check Filter) ")
    print(df['release_speed'].describe())
    
    print("\n Handedness (Platoon) ")
    print(df.groupby(['p_throws', 'stand']).size().unstack())

    print("\n FULL COLUMN LIST ")
    cols = df.columns.tolist()
    for i in range(0, len(cols), 4):
        print(", ".join(cols[i:i+4]))

if __name__ == "__main__":
    run_basic_eda()