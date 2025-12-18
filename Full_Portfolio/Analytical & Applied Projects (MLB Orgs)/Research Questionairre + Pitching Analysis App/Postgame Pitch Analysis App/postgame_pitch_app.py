import streamlit as st
import plotly.express as px
import data_processor as dp
import pandas as pd
import numpy as np

# Set up the page layout
st.set_page_config(page_title="Pitching Analysis", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    h3 {
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #666;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    .stButton > button {
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .dataframe {
        font-size: 0.85rem;
    }
    .dataframe th {
        padding: 0.4rem 0.6rem;
    }
    .dataframe td {
        padding: 0.3rem 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Color scheme roughly matching Baseball Savant's pitch type colors
# Makes it easier to compare with their visualizations
SAVANT_COLORS = {
    'Cutter': '#FF82FF',
    'Curveball': '#00B5FF',
    'Sweeper': 'gold',
    'Slider': '#FFB300',  
    'Sinker': '#FF7518', 
    'Fastball': '#E81038',
    'ChangeUp': '#38CC0A', 
    'Splitter': 'teal',
    'Knuckle': 'purple'
}

@st.cache_data
def get_data():
    """Load the dataset - cache it so we don't reload on every interaction."""
    return dp.load_data("CWS_ML_Analyst_Dataset.csv")

df = get_data()

# Build the pitcher dropdown list
# Show pitcher name, team, and pitch count so it's easier to pick
pitcher_counts = df.groupby('Pitcher').size().sort_values(ascending=False)
pitcher_list = pitcher_counts.index.tolist()
pitcher_teams = df.groupby('Pitcher')['PitcherTeam'].first()

pitcher_display = []
for p in pitcher_list:
    team = pitcher_teams.get(p, '')
    count = pitcher_counts[p]
    pitcher_display.append(f"{p} ({team}) - {count} pitches")


col1, col2 = st.columns([3, 1])
with col2:
    selected_pitcher_idx = st.selectbox(
        "Select Pitcher",
        range(len(pitcher_list)),
        format_func=lambda x: pitcher_display[x],
        key="pitcher_select"
    )
    selected_pitcher = pitcher_list[selected_pitcher_idx]

with col1:
    st.title(f"Post-Game Pitching Report: {selected_pitcher}")

pitcher_df = df[df['Pitcher'] == selected_pitcher].copy()

if len(pitcher_df) > 0:
    # Quick summary stats at the top
    total_pitches = len(pitcher_df)
    max_velo = pitcher_df['RelSpeed'].max()
    k_rate = (pitcher_df['is_strike'].sum() / total_pitches) * 100 if total_pitches > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pitches", total_pitches)
    with col2:
        st.metric("Peak Velocity", f"{max_velo:.1f} mph")
    with col3:
        st.metric("Strike %", f"{k_rate:.1f}%")

st.subheader("Pitch Mix & Performance")

# Start with all pitches, then filter based on selections
mix_filtered_df = pitcher_df.copy()
mix_available_pitches = sorted(pitcher_df['TaggedPitchType'].unique())
mix_selected_pitches = mix_available_pitches

# Filter controls for the pitch mix table
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Outcome**")
    mix_outcome_options = ["All Pitches", "Swing & Misses", "Base Hits", "Strikes", "Balls"]
    mix_selected_outcome = st.selectbox(
        "Filter by outcome",
        mix_outcome_options,
        index=0,
        label_visibility="collapsed",
        key="mix_outcome"
    )
with col2:
    st.markdown("**Batter Handedness**")
    if 'BatterSide' in pitcher_df.columns:
        mix_batter_options = ["All Batters", "Right Handed", "Left Handed"]
        mix_selected_batter = st.selectbox(
            "Filter by batter side",
            mix_batter_options,
            index=0,
            label_visibility="collapsed",
            key="mix_batter"
        )
    else:
        mix_selected_batter = "All Batters"
with col3:
    st.markdown("**Count Situation**")
    mix_count_options = ["All Counts", "Ahead in Count", "Behind in Count", "Even Count", "2 Strikes"]
    mix_selected_count = st.selectbox(
        "Filter by count",
        mix_count_options,
        index=0,
        label_visibility="collapsed",
        key="mix_count"
    )

# Apply outcome filter
if mix_selected_outcome == "Swing & Misses":
    mix_filtered_df = mix_filtered_df[mix_filtered_df['is_whiff'] == True]
elif mix_selected_outcome == "Base Hits":
    if 'HitType' in mix_filtered_df.columns:
        mix_filtered_df = mix_filtered_df[mix_filtered_df['HitType'].notna() & (mix_filtered_df['HitType'] != '')]
elif mix_selected_outcome == "Strikes":
    mix_filtered_df = mix_filtered_df[mix_filtered_df['is_strike'] == True]
elif mix_selected_outcome == "Balls":
    mix_filtered_df = mix_filtered_df[mix_filtered_df['is_strike'] == False]

# Apply batter handedness filter
# check for multiple variations for future datasets that may change syntax
if mix_selected_batter == "Right Handed" and 'BatterSide' in mix_filtered_df.columns:
    mix_filtered_df = mix_filtered_df[mix_filtered_df['BatterSide'].isin(['Right', 'R', 'RIGHT', 'right'])]
elif mix_selected_batter == "Left Handed" and 'BatterSide' in mix_filtered_df.columns:
    mix_filtered_df = mix_filtered_df[mix_filtered_df['BatterSide'].isin(['Left', 'L', 'LEFT', 'left'])]

# Apply count situation filter
if mix_selected_count == "Ahead in Count" and 'Balls' in mix_filtered_df.columns and 'Strikes' in mix_filtered_df.columns:
    mix_filtered_df = mix_filtered_df[mix_filtered_df['Balls'] > mix_filtered_df['Strikes']]
elif mix_selected_count == "Behind in Count" and 'Balls' in mix_filtered_df.columns and 'Strikes' in mix_filtered_df.columns:
    mix_filtered_df = mix_filtered_df[mix_filtered_df['Balls'] < mix_filtered_df['Strikes']]
elif mix_selected_count == "Even Count" and 'Balls' in mix_filtered_df.columns and 'Strikes' in mix_filtered_df.columns:
    mix_filtered_df = mix_filtered_df[mix_filtered_df['Balls'] == mix_filtered_df['Strikes']]
elif mix_selected_count == "2 Strikes" and 'Strikes' in mix_filtered_df.columns:
    mix_filtered_df = mix_filtered_df[mix_filtered_df['Strikes'] == 2]

if len(mix_filtered_df) > 0:
    # Aggregate stats by pitch type
    p_df = mix_filtered_df.copy()
    metrics = {
        'Count': ('TaggedPitchType', 'count'),
        'Velo': ('RelSpeed', 'mean'),
        'Spin': ('SpinRate', 'mean'),
        'H-Break': ('HorzBreak', 'mean'),
        'V-Break': ('InducedVertBreak', 'mean'),
        'Whiffs': ('is_whiff', 'sum'),
        'Swings': ('is_swing', 'sum'),
        'Chases': ('is_chase', 'sum'),
        'OutOfZone': ('in_zone', lambda x: (~x).sum()),  # count pitches outside zone
        'Strikes': ('is_strike', 'sum'),
    }
    grouped = p_df.groupby('TaggedPitchType').agg(**metrics).reset_index()
    
    # Calculate percentages
    grouped['Usage%'] = (grouped['Count'] / grouped['Count'].sum()) * 100
    grouped['Strike%'] = (grouped['Strikes'] / grouped['Count']) * 100
    grouped['Whiff%'] = np.where(grouped['Swings'] > 0, 
                                 (grouped['Whiffs'] / grouped['Swings']) * 100, 0)
    grouped['Chase%'] = np.where(grouped['OutOfZone'] > 0, 
                                 (grouped['Chases'] / grouped['OutOfZone']) * 100, 0)
    
    # Format and sort by usage
    cols = ['TaggedPitchType', 'Count', 'Usage%', 'Velo', 'Spin', 
            'H-Break', 'V-Break', 'Strike%', 'Whiff%', 'Chase%']
    summary_table = grouped[cols].round(1)
    summary_table = summary_table.sort_values('Usage%', ascending=False)
    
    # Display table and pie chart side by side
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(summary_table, use_container_width=True, hide_index=True)
    with col2:
        if len(grouped) > 0:
            fig_pie = px.pie(
                grouped,
                values='Count',
                names='TaggedPitchType',
                color='TaggedPitchType',
                color_discrete_map=SAVANT_COLORS
            )
            fig_pie.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True, key="usage_pie")
else:
    st.info("No pitch type data available for the selected filters.")

# Movement and location plots section
plot_filtered_df = pitcher_df.copy()

if len(plot_filtered_df) > 0:
    st.subheader("Pitch Movement & Location")
    
    plot_available_pitches = sorted(pitcher_df['TaggedPitchType'].unique())
    
    # Filter controls for the plots
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Pitch Types**")
        plot_selected_pitches = st.multiselect(
            "Select pitch types",
            plot_available_pitches,
            default=plot_available_pitches,
            label_visibility="collapsed",
            key="plot_pitches"
        )
    with col2:
        st.markdown("**Outcome**")
        outcome_options = ["All Pitches", "Swing & Misses", "Base Hits", "Strikes", "Balls"]
        plot_selected_outcome = st.selectbox(
            "Filter by outcome",
            outcome_options,
            index=0,
            label_visibility="collapsed",
            key="plot_outcome"
        )
    with col3:
        st.markdown("**Batter Handedness**")
        if 'BatterSide' in pitcher_df.columns:
            batter_options = ["All Batters", "Right Handed", "Left Handed"]
            plot_selected_batter = st.selectbox(
                "Filter by batter side",
                batter_options,
                index=0,
                label_visibility="collapsed",
                key="plot_batter"
            )
        else:
            plot_selected_batter = "All Batters"
    with col4:
        st.markdown("**Count Situation**")
        count_options = ["All Counts", "Ahead in Count", "Behind in Count", "Even Count", "2 Strikes"]
        plot_selected_count = st.selectbox(
            "Filter by count",
            count_options,
            index=0,
            label_visibility="collapsed",
            key="plot_count"
        )

    # Filter by selected pitch types
    if len(plot_selected_pitches) > 0:
        plot_filtered_df = plot_filtered_df[plot_filtered_df['TaggedPitchType'].isin(plot_selected_pitches)]

    # Apply outcome filter
    if plot_selected_outcome == "Swing & Misses":
        plot_filtered_df = plot_filtered_df[plot_filtered_df['is_whiff'] == True]
    elif plot_selected_outcome == "Base Hits":
        if 'HitType' in plot_filtered_df.columns:
            plot_filtered_df = plot_filtered_df[plot_filtered_df['HitType'].notna() & (plot_filtered_df['HitType'] != '')]
    elif plot_selected_outcome == "Strikes":
        plot_filtered_df = plot_filtered_df[plot_filtered_df['is_strike'] == True]
    elif plot_selected_outcome == "Balls":
        plot_filtered_df = plot_filtered_df[plot_filtered_df['is_strike'] == False]

    # Apply batter handedness filter
    if plot_selected_batter == "Right Handed" and 'BatterSide' in plot_filtered_df.columns:
        plot_filtered_df = plot_filtered_df[plot_filtered_df['BatterSide'].isin(['Right', 'R', 'RIGHT', 'right'])]
    elif plot_selected_batter == "Left Handed" and 'BatterSide' in plot_filtered_df.columns:
        plot_filtered_df = plot_filtered_df[plot_filtered_df['BatterSide'].isin(['Left', 'L', 'LEFT', 'left'])]

    # Apply count situation filter
    if plot_selected_count == "Ahead in Count" and 'Balls' in plot_filtered_df.columns and 'Strikes' in plot_filtered_df.columns:
        plot_filtered_df = plot_filtered_df[plot_filtered_df['Balls'] > plot_filtered_df['Strikes']]
    elif plot_selected_count == "Behind in Count" and 'Balls' in plot_filtered_df.columns and 'Strikes' in plot_filtered_df.columns:
        plot_filtered_df = plot_filtered_df[plot_filtered_df['Balls'] < plot_filtered_df['Strikes']]
    elif plot_selected_count == "Even Count" and 'Balls' in plot_filtered_df.columns and 'Strikes' in plot_filtered_df.columns:
        plot_filtered_df = plot_filtered_df[plot_filtered_df['Balls'] == plot_filtered_df['Strikes']]
    elif plot_selected_count == "2 Strikes" and 'Strikes' in plot_filtered_df.columns:
        plot_filtered_df = plot_filtered_df[plot_filtered_df['Strikes'] == 2]

    if len(plot_filtered_df) > 0:
        # Movement plot - shows horizontal vs vertical break
        y_col = "InducedVertBreak"
        y_title = "Induced Vertical Break (in)"
        pitch_counts = plot_filtered_df['TaggedPitchType'].value_counts().to_dict()

        # Create unique keys for plot caching
        filter_key = f"{selected_pitcher}_{str(sorted(plot_selected_pitches))}_{plot_selected_outcome}_{plot_selected_batter}_{plot_selected_count}"
        movement_key = f"movement_plot_{hash(filter_key)}"

        fig_move = px.scatter(
            plot_filtered_df,
            x="HorzBreak",
            y=y_col,
            color="TaggedPitchType",
            hover_data=["RelSpeed", "SpinRate", "PitchCall", "HorzBreak", y_col],
            title="Movement Profile",
            color_discrete_map=SAVANT_COLORS,
            labels={
                "HorzBreak": "Horizontal Break (in)",
                y_col: y_title,
                "TaggedPitchType": "Pitch Type"
            }
        )
        # Fixed axis ranges so plots are comparable across pitchers
        x_axis_min = -24
        x_axis_max = 24
        y_axis_min = -24
        y_axis_max = 24
        
        # Add center lines for reference
        fig_move.add_hline(y=0, line_width=1, line_dash="dot", line_color="grey")
        fig_move.add_vline(x=0, line_width=1, line_dash="dot", line_color="grey")
        
        fig_move.update_xaxes(range=[x_axis_min, x_axis_max], title="Horizontal Break (in)", fixedrange=True)
        fig_move.update_yaxes(
            range=[y_axis_min, y_axis_max],
            title=y_title,
            scaleanchor="x",
            scaleratio=1,
            fixedrange=True
        )

        fig_move.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          "Horizontal Break: %{x:.1f} in<br>" +
                          "Induced Vertical Break: %{y:.1f} in<br>" +
                          "Velocity: %{customdata[0]:.1f} mph<br>" +
                          "Spin Rate: %{customdata[1]:.0f} rpm<br>" +
                          "Result: %{customdata[2]}<extra></extra>"
        )

        fig_move.update_layout(
            height=600,
            legend=dict(
                title="Pitch Type (Count)",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(128,128,128,0.3)",
                borderwidth=1
            )
        )

        # Add pitch counts to legend labels
        for trace in fig_move.data:
            pitch_type = trace.name
            count = pitch_counts.get(pitch_type, 0)
            trace.name = f"{pitch_type} ({count})"
        
        # Location plot shows where pitches crossed the plate
        location_key = f"location_plot_{hash(filter_key)}"
        
        fig_loc = px.scatter(
            plot_filtered_df,
            x="PlateLocSide",
            y="PlateLocHeight",
            color="TaggedPitchType",
            symbol="PitchCall",
            hover_data=["RelSpeed", "ExitSpeed", "Angle", "HorzBreak", "VertBreak", "PitchCall"],
            title="Pitch Locations",
            color_discrete_map=SAVANT_COLORS,
            labels={
                "PlateLocSide": "Plate Side (ft)",
                "PlateLocHeight": "Plate Height (ft)",
                "TaggedPitchType": "Pitch Type",
                "PitchCall": "Result"
            }
        )

        # Fixed ranges for consistent scaling
        # Plate width is about 17 inches = 1.42 ft, so -2 to 2 gives us margin
        # Strike zone height is typically 1.6 to 3.4 ft, so 0 to 5 covers it
        loc_x_axis_min = -2.0
        loc_x_axis_max = 2.0
        loc_y_axis_min = 0.0
        loc_y_axis_max = 5.0
        
        # Draw strike zone box with buffer for baseball diameter
        # Plate width: 17" = 0.708 ft on each side of center
        # Baseball diameter: ~2.9 inches = 0.242 ft, radius = 0.121 ft
        # Buffer expands box by ball radius on all sides to account for ball clipping zone edge
        ball_radius_ft = 0.121  # half of 2.9 inch baseball diameter
        plate_half_width = 0.708
        zone_bottom = 1.6
        zone_top = 3.4
        
        fig_loc.add_shape(
            type="rect",
            x0=-(plate_half_width + ball_radius_ft), 
            y0=zone_bottom - ball_radius_ft, 
            x1=plate_half_width + ball_radius_ft, 
            y1=zone_top + ball_radius_ft,
            line=dict(color="Red", width=2),
            fillcolor="Red", opacity=0.1
        )
        
        fig_loc.update_xaxes(range=[loc_x_axis_min, loc_x_axis_max], title="Plate Side (ft)", fixedrange=True)
        fig_loc.update_yaxes(range=[loc_y_axis_min, loc_y_axis_max], title="Plate Height (ft)", scaleanchor="x", scaleratio=1, fixedrange=True)
        
        # Marker size isroughly proportional to a baseball
        # Baseball is about 2.9 inches diameter, adjusted for visibility
        baseball_diameter_px = 12
        
        fig_loc.update_traces(
            marker=dict(
                size=baseball_diameter_px,
                line=dict(
                    width=1.5,
                )
            ),
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          "Location: (%{x:.2f}, %{y:.2f}) ft<br>" +
                          "Velocity: %{customdata[0]:.1f} mph<br>" +
                          "Exit Speed: %{customdata[1]:.0f} mph<br>" +
                          "Launch Angle: %{customdata[2]:.1f}°<br>" +
                          "H-Break: %{customdata[3]:.1f} in<br>" +
                          "V-Break: %{customdata[4]:.1f} in<br>" +
                          "Result: %{customdata[5]}<extra></extra>"
        )

        fig_loc.update_layout(
            height=600,
            legend=dict(
                title="Pitch Type (Count)",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(128,128,128,0.3)",
                borderwidth=1
            )
        )

        # Add pitch counts to legend labels
        for trace in fig_loc.data:
            pitch_type = trace.name
            count = pitch_counts.get(pitch_type, 0)
            trace.name = f"{pitch_type} ({count})"

        # Display both plots side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_move, use_container_width=True, key=movement_key)
        with col2:
            st.plotly_chart(fig_loc, use_container_width=True, key=location_key)
    else:
        st.info("No data available to display. Please adjust your filters.")
else:
    st.info("No data available to display. Please adjust your filters.")

if len(pitcher_df) > 0:
    with st.expander("See Raw Data"):
        st.dataframe(pitcher_df)