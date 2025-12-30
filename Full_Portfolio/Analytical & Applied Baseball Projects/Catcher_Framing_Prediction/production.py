import sys
import subprocess
import os
import pandas as pd
import numpy as np
import joblib

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    "pandas>=2.0.3",
    "numpy>=1.24.3",
    "joblib>=1.2.0",
    "scikit-learn==1.3.2"
]

print("Checking dependencies...")
for pkg in required_packages:
    try:
        install(pkg)
    except:
        pass

# create sample data for testing
df = pd.read_csv('ML_TAKES_ENCODED.csv')

MODEL_PATH = 'framing_model.pkl'
INPUT_DATA_PATH = 'new_data.csv'
OUTPUT_PREDS = 'pitch_level_predictions.csv'
OUTPUT_METRICS = 'new_output.csv'

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(INPUT_DATA_PATH):
        print("Error: Missing model file or new_data.csv.")
        return

    print(f"Loading {MODEL_PATH}...")
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print(f"Loading {INPUT_DATA_PATH}...")
    df = pd.read_csv(INPUT_DATA_PATH)

    print("Processing data...")
    df['PITCHCALL'] = df['PITCHCALL'].astype(str).str.strip()
    df_takes = df[df['PITCHCALL'].isin(['StrikeCalled', 'BallCalled'])].copy()

    if df_takes.empty:
        print("No takes found in data.")
        return

    df_takes = df_takes.dropna(subset=['TOP_ZONE', 'BOT_ZONE', 'PLATELOCHEIGHT', 'PLATELOCSIDE']).copy()

    # filter out sensor errors and corrupt data
    initial_count = len(df_takes)
    
    df_takes = df_takes[
        (df_takes['RELSPEED'].between(50, 106)) & 
        (df_takes['PLATELOCHEIGHT'].between(0.25, 6.5)) & 
        (df_takes['PLATELOCSIDE'].between(-3.5, 3.5)) & 
        (df_takes['INDUCEDVERTBREAK'].between(-75, 75)) & 
        (df_takes['HORZBREAK'].between(-75, 75))
    ].copy()
    
    dropped = initial_count - len(df_takes)
    if dropped > 0:
        print(f"Filter: Dropped {dropped} rows ({dropped/initial_count:.1%}) of non-frameable/corrupt data.")

    df_takes['PLATELOCHEIGHT_NORM'] = (
        (df_takes['PLATELOCHEIGHT'] - df_takes['BOT_ZONE']) / 
        (df_takes['TOP_ZONE'] - df_takes['BOT_ZONE'])
    )

    # handle year column if missing
    if 'Year' not in df_takes.columns:
        if 'GAME_YEAR' in df_takes.columns:
            df_takes['Year'] = df_takes['GAME_YEAR']
        elif 'GAME_DATE' in df_takes.columns:
             df_takes['Year'] = pd.to_datetime(df_takes['GAME_DATE']).dt.year
        else:
            df_takes['Year'] = 2023

    print("Predicting...")
    try:
        probs = pipeline.predict_proba(df_takes)[:, 1]
    except KeyError as e:
        print(f"Error: Missing columns in new data: {e}")
        return
        
    df_takes['CS_PROB'] = probs
    df_takes['IS_STRIKE'] = np.where(df_takes['PITCHCALL'] == 'StrikeCalled', 1, 0)

    print(f"Exporting {OUTPUT_PREDS}...")
    pitch_out = df_takes[['PITCH_ID', 'IS_STRIKE', 'CS_PROB']].copy()
    pitch_out.to_csv(OUTPUT_PREDS, index=False)

    print(f"Exporting {OUTPUT_METRICS}...")
    
    df_takes['STRIKES_ADDED'] = df_takes['IS_STRIKE'] - df_takes['CS_PROB']

    metrics = df_takes.groupby(['CATCHER_ID', 'Year']).agg(
        Opportunities=('PITCH_ID', 'count'),
        Actual_Called_Strikes=('IS_STRIKE', 'sum'),
        Sum_Strikes_Added=('STRIKES_ADDED', 'sum')
    ).reset_index()

    metrics['Called Strikes "added"'] = metrics['Sum_Strikes_Added'].round(4)
    metrics['Called Strikes "added" per 100 opportunities'] = (
        (metrics['Sum_Strikes_Added'] / metrics['Opportunities']) * 100
    ).round(4)

    final_output = metrics[[
        'CATCHER_ID', 'Year', 'Opportunities', 
        'Actual_Called_Strikes', 
        'Called Strikes "added"', 
        'Called Strikes "added" per 100 opportunities'
    ]].rename(columns={'Actual_Called_Strikes': 'Actual Called Strikes'})

    final_output.to_csv(OUTPUT_METRICS, index=False)
    print("Done.")

if __name__ == "__main__":
    main()