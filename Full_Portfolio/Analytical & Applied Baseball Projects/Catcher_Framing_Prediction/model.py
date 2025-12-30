import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, log_loss

DATA_PATH = 'ML_TAKES_ENCODED.csv'
MODEL_OUTPUT_PATH = 'framing_model.pkl'

NUMERIC_FEATURES = [
    'PLATELOCSIDE', 
    'PLATELOCHEIGHT_NORM', 
    'INDUCEDVERTBREAK', 
    'HORZBREAK', 
    'RELSPEED'
]

CATEGORICAL_FEATURES = [
    'BATTERSIDE', 
    'PITCHERTHROWS', 
    'AUTOPITCHTYPE'
]

def main():
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
        return

    print("Preprocessing...")
    df['PITCHCALL'] = df['PITCHCALL'].astype(str).str.strip()
    df = df[df['PITCHCALL'].isin(['StrikeCalled', 'BallCalled'])].copy()

    df = df.dropna(subset=['TOP_ZONE', 'BOT_ZONE', 'PLATELOCHEIGHT', 'PLATELOCSIDE', 'GAMEID']).copy()

    df['IS_STRIKE'] = np.where(df['PITCHCALL'] == 'StrikeCalled', 1, 0)

    # normalize height to 0-1 range
    df['PLATELOCHEIGHT_NORM'] = (df['PLATELOCHEIGHT'] - df['BOT_ZONE']) / (df['TOP_ZONE'] - df['BOT_ZONE'])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['IS_STRIKE']
    groups = df['GAMEID']

    # split by game to prevent data leakage
    # test set will contain entirely different games than train set
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    print(f"Train set: {len(X_train)} pitches from {groups.iloc[train_idx].nunique()} games")
    print(f"Test set: {len(X_test)} pitches from {groups.iloc[test_idx].nunique()} games")

    print("Building model...")

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL_FEATURES),
            ('num', 'passthrough', NUMERIC_FEATURES)
        ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    clf = HistGradientBoostingClassifier(
        random_state=42,
        categorical_features=[0, 1, 2], 
        max_iter=100
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    print("Training...")
    model_pipeline.fit(X_train, y_train)

    probs = model_pipeline.predict_proba(X_test)[:, 1]
    preds = model_pipeline.predict(X_test)
    loss = log_loss(y_test, probs)
    print(f"Log Loss: {loss:.4f}")
    print(classification_report(y_test, preds, digits=4))

    joblib.dump(model_pipeline, MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()