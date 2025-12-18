# Post-Game Pitching Report

## App Description
   - Interactive dashboard for analyzing pitcher performance from a single game's pitch tracking data
   - Features: pitch mix analysis with usage percentages, movement profiles (horizontal vs vertical break), pitch location plots with strike zone overlay, and interactive filtering by outcome, batter handedness, and count situation

## Design Choices

### Data Processing
   - Encoding handling: Attempts multiple encodings (latin-1, cp1252, utf-8-sig) to handle CSV encoding variations; may fail on non-standard formats
   - Filters out "Undefined" pitch types and pitch calls as they are not useful
   - Strike zone: Uses fixed approximated dimensions (1.6-3.4 ft height, ±0.83 ft width) based on standard MLB zone. This is an approximation since the dataset lacks sz_top/sz_bottom fields. Zones vary by batter, but provides a consistent visual reference.
   - Metric calculations: Builds boolean flags from PitchCall codes to identify swings, whiffs, strikes, in-zone pitches, and chases

### Visualizations
   - Color scheme: Roughly matches Baseball Savant's pitch type colors for familiarity (cutter changed from brown to pink for better visual distinction)
   - Fixed axis ranges: Movement plots use -24 to 24 inches on both axes to enable consistent comparison across pitchers. Variable ranges would make comparisons misleading.
   - Strike zone overlay: Red rectangle on location plot provides visual reference for zone boundaries
   - Plotly: Chosen for interactive hover details and dashboard-friendly output
   - Legend labels: Include pitch counts to indicate sample sizes for each pitch type

### User Interface
   - Streamlit framework: Selected for rapid development and built-in interactivity
   - Wide layout: Side-by-side plots optimize screen space for comparative analysis
   - Data caching: @st.cache_data decorator prevents data reloading on filter changes, improving performance
   - Filter organization: Grouped by category (outcome, batter, count) to keep UI manageable
   - Any issues caused by resizing the page are solved by refreshing

### Code Structure
   - Separation of concerns: Data processing isolated in separate module (data_processor.py) for maintainability
   - Modular functions: Reusable data loading and aggregation functions

## Assumptions

### Data Format
   - CSV file with standard pitch tracking columns: Pitcher, TaggedPitchType, PitchCall, RelSpeed, SpinRate, HorzBreak, InducedVertBreak, PlateLocSide, PlateLocHeight, Balls, Strikes, BatterSide, etc.
   - Assumes pitch tracking data is reasonably accurate (classifications, locations, and TaggedPitchType field are correct)
   - BatterSide field handled flexibly to accommodate syntax variations of future datasets

### Metric Definitions
   - Swing: StrikeSwinging, FoulBall, or InPlay
   - Strike: Called strike, swinging strike, foul, or ball in play
   - Whiff: StrikeSwinging (swing and miss)
   - Chase: Any swing at a pitch outside the strike zone
   - Count situations: Ahead = more balls than strikes, Behind = more strikes than balls, Even = equal

## Running Locally
   - Prerequisites: Python 3.7+ with required packages (streamlit, pandas, numpy, plotly)
   - Installation: `pip install streamlit pandas numpy plotly`
   - Ensure CSV data file (`CWS_ML_Analyst_Dataset.csv`) is in the project directory
   - Run: `streamlit run postgame_pitch_app.py`
   - Application opens in default web browser (typically http://localhost:8501)

