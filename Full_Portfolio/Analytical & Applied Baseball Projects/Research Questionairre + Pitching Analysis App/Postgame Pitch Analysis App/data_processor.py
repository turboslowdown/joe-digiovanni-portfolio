import pandas as pd
import numpy as np

# Strike zone dimensions - using standard MLB zone
# Height: 1.6 to 3.4 feet (varies by batter but this is typical)
# Width: 17" plate + ball radius gives us about +/- 0.83 ft
ZONE_TOP = 3.4
ZONE_BOT = 1.6
ZONE_X_HALF = 0.83 

def load_data(filepath):
    """
    Loads the pitch tracking CSV and builds the key boolean flags we need.
    
    Had to deal with some encoding issues with this dataset, so this tries a few
    common ones. It also filters out undefined pitch types/calls since they're
    not useful. 
    
    Returns a dataframe with the boolean flags for swings, whiffs, strikes, in-zone pitches, and chases.
    """
    df = None
    for enc in ['latin-1', 'cp1252', 'utf-8-sig']:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
            
    if df is None:
        raise ValueError("Couldn't read the file - encoding issue.")

    # Get rid of undefined pitches/calls
    df = df[df['TaggedPitchType'] != 'Undefined'].copy()
    df = df[df['PitchCall'] != 'Undefined'].copy()
    
    # Build swing/miss flags based on PitchCall
    # Swings include swinging strikes, fouls, and balls in play
    swing_codes = ['StrikeSwinging', 'FoulBall', 'InPlay']
    strike_codes = ['StrikeCalled', 'StrikeSwinging', 'FoulBall', 'InPlay']
    
    df['is_swing'] = df['PitchCall'].isin(swing_codes)
    df['is_whiff'] = df['PitchCall'] == 'StrikeSwinging'  # define whiff as swing and miss
    df['is_strike'] = df['PitchCall'].isin(strike_codes)
    
    # Check if the pitch in the strike zone
    # Using plate location coordinates from tracking data
    df['in_zone'] = (
        (df['PlateLocSide'].abs() <= ZONE_X_HALF) & 
        (df['PlateLocHeight'] >= ZONE_BOT) & 
        (df['PlateLocHeight'] <= ZONE_TOP)
    )
    
    # Chase = swing at a pitch outside the zone
    # Note: chase is estimated since the dataset lacks strike zone data
    df['is_chase'] = df['is_swing'] & (~df['in_zone'])

    return df

def get_pitcher_summary(df, pitcher_name):
    """
    Pulls out one pitcher's data and aggregates by pitch type.
    
    Returns a summary table with usage, velo, spin, movement, and rates.
    """
    p_df = df[df['Pitcher'] == pitcher_name].copy()
    
    if p_df.empty: 
        return pd.DataFrame()

    # Aggregate the key metrics by pitch type
    metrics = {
        'Count': ('TaggedPitchType', 'count'),
        'Velo': ('RelSpeed', 'mean'),
        'Spin': ('SpinRate', 'mean'),
        'H-Break': ('HorzBreak', 'mean'),
        'V-Break': ('InducedVertBreak', 'mean'),
        'Whiffs': ('is_whiff', 'sum'),
        'Swings': ('is_swing', 'sum'),
        'Chases': ('is_chase', 'sum'),
        'OutOfZone': ('in_zone', lambda x: (~x).sum()),  # count of out-of-zone pitches
        'Strikes': ('is_strike', 'sum'),
    }

    grouped = p_df.groupby('TaggedPitchType').agg(**metrics).reset_index()
    
    # Calculate rate stats
    grouped['Usage%'] = (grouped['Count'] / grouped['Count'].sum()) * 100
    grouped['Strike%'] = (grouped['Strikes'] / grouped['Count']) * 100
    
    # Whiff% and Chase% need to handle cases where denominator is 0
    grouped['Whiff%'] = np.where(grouped['Swings'] > 0, 
                                 (grouped['Whiffs'] / grouped['Swings']) * 100, 0)
    grouped['Chase%'] = np.where(grouped['OutOfZone'] > 0, 
                                 (grouped['Chases'] / grouped['OutOfZone']) * 100, 0)
    
    # Return just the columns we want to show
    cols = ['TaggedPitchType', 'Count', 'Usage%', 'Velo', 'Spin', 
            'H-Break', 'V-Break', 'Strike%', 'Whiff%', 'Chase%']
            
    return grouped[cols].round(1)