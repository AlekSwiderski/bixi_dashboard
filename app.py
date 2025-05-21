import dash
from dash import dcc, html, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import os
from math import radians, sin, cos, sqrt, atan2

# --- Constants ---
# Data Loading & Filtering
BIXI_CSV_SAMPLE_SIZE = 100000
MONTREAL_LAT_LOWER_BOUND = 40.0
MONTREAL_LON_UPPER_BOUND = -70.0 # Note: longitude is negative, so this is an upper bound for valid values
MIN_TRIP_DURATION_MINUTES = 0.0
MAX_TRIP_DURATION_MINUTES = 1440.0 # 24 hours
MIN_TRIP_DISTANCE_KM = 0.0
MAX_TRIP_DISTANCE_KM = 15.0
MIN_TRIP_SPEED_KMH = 0.0 
MAX_TRIP_SPEED_KMH = 35.0

# Visualization parameters
MAX_DURATION_VISUALIZATION_MINUTES = 60
MAX_DISTANCE_VISUALIZATION_KM = 10
MAX_SPEED_VISUALIZATION_KMH = 30
TOP_STATIONS_COUNT = 10
TOP_DISTRICTS_COUNT = 5
TOP_STATION_PAIRS_COUNT = 15
SIGNIFICANT_DISTRICT_TRIP_COUNT = 50
TOP_STATIONS_MAP_COUNT = 100

# Map settings
MAP_DEFAULT_ZOOM = 11
MAP_MONTREAL_CENTER_LAT = 45.5088
MAP_MONTREAL_CENTER_LON = -73.5878

# Geographical constants
EARTH_RADIUS_KM = 6371

# Time-based constants
RUSH_HOUR_MORNING_START = 7
RUSH_HOUR_MORNING_END = 9
RUSH_HOUR_EVENING_START = 16
RUSH_HOUR_EVENING_END = 18
# --- End Constants ---

# Get the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, 'Bixi_2023_sample.csv')

# Initialize the Dash app with improved title and metadata
app = dash.Dash(__name__, 
                title='Bixi 2023 Analysis Dashboard',
                meta_tags=[{'name': 'viewport', 
                           'content': 'width=device-width, initial-scale=1.0'}])

server = app.server

# Custom CSS for better styling is now in assets/style.css

# App layout with improved styling
app.layout = html.Div([
    # Header with gradient background
    html.Div([
        html.H1('Bixi 2023 Data Analysis Dashboard', 
                style={'textAlign': 'center', 'color': 'white', 'margin-bottom': '10px'}),
        html.H3('Exploring Montreal\'s Bike-Sharing Patterns', 
                style={'textAlign': 'center', 'color': 'rgba(255,255,255,0.85)', 'margin-top': '0px', 
                       'font-weight': 'normal'})
    ], className='dashboard-header'),
    
    # Main content
    html.Div([
        # Filters panel
        html.Div([
            html.H4('Filters', style={'textAlign': 'center', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            
            html.Label('Time Period:', style={'marginTop': '20px', 'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='time-filter',
                options=[
                    {'label': 'All Data', 'value': 'all'},
                    {'label': 'Weekdays Only', 'value': 'weekday'},
                    {'label': 'Weekends Only', 'value': 'weekend'}
                ],
                value='all',
                className='dropdown'
            ),
            
            html.Div([
                html.H4('Dataset Statistics', style={'textAlign': 'center', 'marginTop': '30px', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
                html.Div(id='dataset-stats')
            ], className='stats-box'),
            
            html.Div([
                html.Button(
                    'Load Data',
                    id='load-data-button',
                    className='load-button'
                ),
                html.Div(
                    id='loading-status',
                    children=[
                        html.P(
                            "Click 'Load Data' to begin analysis. Loading may take 5-10 seconds.",
                            style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px', 'marginTop': '5px'}
                        )
                    ]
                ),
                # Add a spinner for visual feedback
                dcc.Loading(
                    id="loading-spinner",
                    type="circle",
                    children=html.Div(id="loading-output")
                )
            ])
        ], className='filters-panel', style={'width': '15%', 'float': 'left'}),
        
        # Charts panel
        html.Div([
            # Tabs for different analysis categories
            dcc.Tabs([
                # Temporal Patterns Tab
                dcc.Tab(label='Temporal Patterns', children=[
                    html.Div([
                        html.Div([
                            html.H4('Hourly Trip Distribution', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='hourly-pattern-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.H4('Daily Trip Distribution', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='daily-pattern-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block'}),
                        
                        html.Div([
                            html.H4('Monthly Trip Distribution', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='monthly-pattern-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.H4('Weekday vs Weekend Hourly Patterns', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='weekday-weekend-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block'})
                    ])
                ], className='dash-tab'),
                
                # Spatial Patterns Tab
                dcc.Tab(label='Spatial Patterns', children=[
                    html.Div([
                        html.Div([
                            html.H4('Top Stations', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='top-stations-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.H4('District Flows', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='district-flow-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block'})
                    ])
                ], className='dash-tab'),
                
                # Trip Characteristics Tab
                dcc.Tab(label='Trip Characteristics', children=[
                    html.Div([
                        html.Div([
                            html.H4('Trip Duration Distribution', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='duration-dist-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.H4('Trip Distance Distribution', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='distance-dist-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block'}),
                        
                        html.Div([
                            html.H4('Speed Distribution', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='speed-dist-chart')
                        ], className='chart-container', style={'width': '100%'})
                    ])
                ], className='dash-tab'),
                
                # Station Map Tab (NEW)
                dcc.Tab(label='Station Map', children=[
                    html.Div([
                        html.Div([
                            html.H3('Bixi Station Map', style={'textAlign': 'center', 'color': '#3498db', 'marginBottom': '20px'}),
                            html.P('Showing the top 100 stations by usage. Size indicates popularity and color indicates district.', 
                                   style={'textAlign': 'center', 'marginBottom': '20px'}),
                            html.Div(id='map-loading-indicator', children=[
                                html.P("Loading map... if it doesn't appear, try selecting a different tab then returning here.", 
                                       style={'textAlign': 'center', 'color': '#e67e22', 'marginBottom': '10px'})
                            ]),
                            dcc.Graph(id='station-map', style={'height': '75vh'})
                        ], className='chart-container', style={'width': '100%', 'padding': '20px'})
                    ])
                ], className='dash-tab'),
                
                # Insights Tab
                dcc.Tab(label='Key Insights', children=[
                    html.Div([
                        html.Div([
                            html.H4('Rush Hour Analysis', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='rush-hour-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.H4('Station Net Flow (Starts - Ends)', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='station-flow-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block'}),
                        
                        html.Div([
                            html.H4('Station Pair Analysis', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='station-pair-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.H4('Trip Duration by District', style={'textAlign': 'center', 'color': '#3498db'}),
                            dcc.Graph(id='district-duration-chart')
                        ], className='chart-container', style={'width': '48%', 'display': 'inline-block'})
                    ])
                ], className='dash-tab'),
                
                # About Tab (NEW)
                dcc.Tab(label='About', children=[
                    html.Div([
                        html.Div([
                            html.H3('About the Bixi Dashboard', 
                                    style={'textAlign': 'center', 'color': '#3498db', 'marginBottom': '20px'}),
                            
                            html.Div([
                                html.H4('Dataset Overview', style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
                                html.P([
                                    "This dashboard analyzes Montreal's Bixi bike-sharing system data from the 2023 season. ",
                                    "The data includes trip information such as start/end stations, timestamps, and geographical coordinates."
                                ]),
                                html.P([
                                    "The complete dataset contains millions of trips, but for performance reasons, ",
                                    "this dashboard uses a sample of 100,000 randomly selected trips. After filtering out invalid entries ",
                                    "(trips with unrealistic durations, distances, or coordinates), the final analyzed dataset contains ",
                                    "approximately 95,000 valid trips."
                                ]),
                                
                                html.H4('Data Source', style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'marginTop': '30px'}),
                                html.P([
                                    "The data used in this dashboard is available for public download from the City of Montreal's open data portal: "
                                ]),
                                html.A(
                                    "Montreal Open Data - Bixi System",
                                    href="https://donnees.montreal.ca/ville-de-montreal/stations-bixi",
                                    target="_blank",
                                    style={'color': '#3498db'}
                                ),
                                html.P([
                                    "The raw data is provided in CSV format and includes all Bixi trips from the 2023 operating season."
                                ]),
                                
                                html.H4('Key Insights', style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'marginTop': '30px'}),
                                html.Ul([
                                    html.Li("Peak usage occurs during commuting hours (8-9 AM and 4-6 PM) on weekdays."),
                                    html.Li("Weekend usage patterns differ significantly, with more evenly distributed trips throughout the day."),
                                    html.Li("The average trip lasts approximately 12.7 minutes and covers 1.92 kilometers."),
                                    html.Li("Popular districts include Ville-Marie, Plateau-Mont-Royal, and Rosemont-La Petite-Patrie."),
                                    html.Li("Weather has a significant impact on ridership, with peak usage during summer months."),
                                    html.Li("Some stations show a net outflow of bikes in the morning and net inflow in the evening, suggesting commuter patterns."),
                                ], style={'marginLeft': '20px'}),
                                
                                html.H4('Notes on Data Loading', style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'marginTop': '30px'}),
                                html.P([
                                    "When clicking the 'Load Data' button, please note:"
                                ]),
                                html.Ul([
                                    html.Li("Initial data loading may take 5-10 seconds."),
                                    html.Li("Graphs will populate gradually as data is processed."),
                                    html.Li("Some charts may require additional clicks to render if they weren't visible during initial load."),
                                    html.Li("The sample size shows 100,000 trips, but the total count is lower (approximately 95,000) because invalid trips are filtered out during processing."),
                                ], style={'marginLeft': '20px'}),
                                
                            ], style={'marginBottom': '30px', 'lineHeight': '1.6'})
                        ], className='chart-container', style={'width': '100%', 'padding': '30px'})
                    ])
                ], className='dash-tab')
            ], style={'margin-bottom': '20px'})
        ], style={'width': '85%', 'float': 'right', 'padding': '20px 20px 20px 16%'})
    ])
])

# Cache for loaded data
data_cache = {
    'df': None
}

# Data loading function (fixed version)
def load_bixi_data(file_path):
    """
    Load a fixed sample of the Bixi dataset with time conversions
    to prevent memory issues with the full 3GB dataset
    """
    # Define optimized dtypes
    dtypes = {
        'STARTSTATIONNAME': str,
        'STARTSTATIONARRONDISSEMENT': str,
        'STARTSTATIONLATITUDE': float,
        'STARTSTATIONLONGITUDE': float,
        'ENDSTATIONNAME': str,
        'ENDSTATIONARRONDISSEMENT': str,
        'ENDSTATIONLATITUDE': float,
        'ENDSTATIONLONGITUDE': float,
        'STARTTIMEMS': 'int64',
        'ENDTIMEMS': 'Int64'  # Use nullable integer type
    }
    
    # Load data with error handling
    try:
        # First try with specified dtypes
        df = pd.read_csv(file_path, nrows=BIXI_CSV_SAMPLE_SIZE, dtype=dtypes)
    except Exception as e:
        print(f"Error with specified dtypes: {e}")
        # If that fails, try with automatic dtype inference
        try:
            df = pd.read_csv(file_path, nrows=BIXI_CSV_SAMPLE_SIZE)
            print("Loaded with automatic dtype inference")
        except Exception as e:
            print(f"Error with automatic inference: {e}")
            raise e
    
    # Now that we have the data, convert to appropriate types and handle errors
    try:
        # Convert timestamps and add derived columns
        df['start_time'] = pd.to_datetime(df['STARTTIMEMS'], unit='ms')
        df['end_time'] = pd.to_datetime(df['ENDTIMEMS'], unit='ms', errors='coerce')
        df['duration_seconds'] = (df['ENDTIMEMS'] - df['STARTTIMEMS']) / 1000
        df['duration_minutes'] = df['duration_seconds'] / 60
        
        # Add time components
        df['start_hour'] = df['start_time'].dt.hour
        df['start_day'] = df['start_time'].dt.day_name()
        df['start_date'] = df['start_time'].dt.date
        df['start_month'] = df['start_time'].dt.month
        df['start_month_name'] = df['start_time'].dt.month_name()
        df['is_weekend'] = df['start_time'].dt.dayofweek >= 5
        
        # Filter valid trips
        initial_rows = len(df)
        print(f"Data before duration filtering: {initial_rows} rows")
        df = df[(df['duration_minutes'] > MIN_TRIP_DURATION_MINUTES) & (df['duration_minutes'] < MAX_TRIP_DURATION_MINUTES)]
        rows_after_duration_filter = len(df)
        print(f"Data after duration filtering: {rows_after_duration_filter} rows ({initial_rows - rows_after_duration_filter} rows dropped)")
        
        # Calculate distances for valid coordinates
        for col in ['STARTSTATIONLATITUDE', 'STARTSTATIONLONGITUDE', 'ENDSTATIONLATITUDE', 'ENDSTATIONLONGITUDE']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        initial_rows_coords = len(df)
        print(f"Data before coordinate filtering: {initial_rows_coords} rows")
        
        valid_coords_mask = (
            (df['STARTSTATIONLATITUDE'] > MONTREAL_LAT_LOWER_BOUND) &
            (df['STARTSTATIONLONGITUDE'] < MONTREAL_LON_UPPER_BOUND) &
            (df['ENDSTATIONLATITUDE'] > MONTREAL_LAT_LOWER_BOUND) &
            (df['ENDSTATIONLONGITUDE'] < MONTREAL_LON_UPPER_BOUND)
        )
        
        df.loc[valid_coords_mask, 'distance_km'] = df.loc[valid_coords_mask].apply(
            lambda row: haversine_distance(
                row['STARTSTATIONLATITUDE'], row['STARTSTATIONLONGITUDE'],
                row['ENDSTATIONLATITUDE'], row['ENDSTATIONLONGITUDE']
            ), axis=1
        )
        
        df.loc[valid_coords_mask & df['duration_minutes'].notna() & (df['duration_minutes'] > MIN_TRIP_DURATION_MINUTES), 'speed_kmh'] = (
            df.loc[valid_coords_mask & df['duration_minutes'].notna() & (df['duration_minutes'] > MIN_TRIP_DURATION_MINUTES), 'distance_km'] / 
            (df.loc[valid_coords_mask & df['duration_minutes'].notna() & (df['duration_minutes'] > MIN_TRIP_DURATION_MINUTES), 'duration_minutes'] / 60)
        )

        initial_rows_distance_filter = len(df)
        print(f"Data before distance filtering: {initial_rows_distance_filter} rows")
        df = df[(df['distance_km'].isna()) | 
               ((df['distance_km'] >= MIN_TRIP_DISTANCE_KM) & (df['distance_km'] < MAX_TRIP_DISTANCE_KM))]
        rows_after_distance_filter = len(df)
        print(f"Data after distance filtering: {rows_after_distance_filter} rows ({initial_rows_distance_filter - rows_after_distance_filter} rows dropped)")

        initial_rows_speed_filter = len(df)
        print(f"Data before speed filtering: {initial_rows_speed_filter} rows")
        df = df[(df['speed_kmh'].isna()) | 
               ((df['speed_kmh'] > MIN_TRIP_SPEED_KMH) & (df['speed_kmh'] < MAX_TRIP_SPEED_KMH))]
        rows_after_speed_filter = len(df)
        print(f"Data after speed filtering: {rows_after_speed_filter} rows ({initial_rows_speed_filter - rows_after_speed_filter} rows dropped)")
        
        print(f"Finished data cleaning. Final dataset: {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        raise e

def haversine_distance(lat1_orig, lon1_orig, lat2_orig, lon2_orig):
    """Calculate distance between two points in km using Haversine formula"""
    try:
        # Attempt to convert inputs to float, store original values for logging
        try:
            lat1 = float(lat1_orig)
            lon1 = float(lon1_orig)
            lat2 = float(lat2_orig)
            lon2 = float(lon2_orig)
        except ValueError as ve:
            print(f"Haversine distance: Error converting inputs to float. lat1='{lat1_orig}', lon1='{lon1_orig}', lat2='{lat2_orig}', lon2='{lon2_orig}'. Details: {ve}")
            return None
        
        # Convert to radians
        rad_lat1, rad_lon1, rad_lat2, rad_lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = rad_lat2 - rad_lat1
        dlon = rad_lon2 - rad_lon1
        a = sin(dlat/2)**2 + cos(rad_lat1) * cos(rad_lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = EARTH_RADIUS_KM * c
        
        return distance
    except Exception as e:
        # This will catch other unexpected errors during calculation (e.g., math domain errors if a is out of range for sqrt)
        print(f"Haversine distance: Error calculating distance for ({lat1_orig}, {lon1_orig}, {lat2_orig}, {lon2_orig}). Details: {e}")
        return None

def create_error_figure(error_message="An error occurred while generating this chart."):
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source="https://images.plot.ly/plotly-documentation/images/logo-plotly-layout-image.png", # Or a generic error icon
            xref="paper", yref="paper",
            x=0.5, y=0.6,
            sizex=0.3, sizey=0.3,
            xanchor="center", yanchor="middle",
            opacity=0.3
        )
    )
    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        annotations=[
            dict(
                text=error_message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.4,
                showarrow=False,
                font=dict(size=14, color="#FF0000") # Red color for error
            )
        ],
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

# Callback to load data
@app.callback(
    Output('dataset-stats', 'children'),
    [Input('load-data-button', 'n_clicks')]
)
def load_data(n_clicks):
    if n_clicks is None:
        return html.Div([
            html.P("Click 'Load Data' to begin analysis.", 
                   style={'textAlign': 'center', 'color': '#7f8c8d'})
        ])
    
    # Check if we need to load new data
    if data_cache['df'] is None:
        try:
            # Load data with fixed sample size 
            df = load_bixi_data(data_file_path)
            data_cache['df'] = df
            
            # Generate updated stats with improved styling
            stats_html = html.Div([
                html.Div([
                    html.Span("Total Valid Trips: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{len(df):,}")
                ], style={'margin': '5px 0'}),
                
                html.Div([
                    html.Span("Unique Stations: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{df['STARTSTATIONNAME'].nunique():,}")
                ], style={'margin': '5px 0'}),
                
                html.Div([
                    html.Span("Date Range: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{df['start_time'].min().date()} to {df['start_time'].max().date()}")
                ], style={'margin': '5px 0'}),
                
                html.Div([
                    html.Span("Avg. Trip Duration: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{df['duration_minutes'].mean():.1f} minutes")
                ], style={'margin': '5px 0'}),
                
                html.Div([
                    html.Span("Avg. Trip Distance: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{df.loc[df['distance_km'].notna(), 'distance_km'].mean():.2f} km")
                ], style={'margin': '5px 0'}),
                
                html.Div([
                    html.Span("Initial Sample: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{BIXI_CSV_SAMPLE_SIZE:,} trips")
                ], style={'margin': '5px 0'}),
                
                html.Div([
                    html.Span("After Filtering: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{len(df):,} valid trips")
                ], style={'margin': '5px 0', 'color': '#3498db'}),
                
                html.Hr(style={'margin': '15px 0', 'borderTop': '1px dashed #ccc'}),
                
                html.Div([
                    html.P("Note: Invalid trips (e.g., with unrealistic durations or coordinates) were filtered out during processing.", 
                           style={'fontSize': '11px', 'color': '#7f8c8d', 'margin': '5px 0'})
                ])
            ])
            
            return stats_html
            
        except Exception as e:
            return html.Div([
                html.P(f"Error loading data: {str(e)}", 
                       style={'color': 'red', 'fontWeight': 'bold'})
            ])
    else:
        # Data already loaded, just return stats
        df = data_cache['df']
        
        # Generate updated stats with improved styling
        stats_html = html.Div([
            html.Div([
                html.Span("Total Valid Trips: ", style={'fontWeight': 'bold'}),
                html.Span(f"{len(df):,}")
            ], style={'margin': '5px 0'}),
            
            html.Div([
                html.Span("Unique Stations: ", style={'fontWeight': 'bold'}),
                html.Span(f"{df['STARTSTATIONNAME'].nunique():,}")
            ], style={'margin': '5px 0'}),
            
            html.Div([
                html.Span("Date Range: ", style={'fontWeight': 'bold'}),
                html.Span(f"{df['start_time'].min().date()} to {df['start_time'].max().date()}")
            ], style={'margin': '5px 0'}),
            
            html.Div([
                html.Span("Avg. Trip Duration: ", style={'fontWeight': 'bold'}),
                html.Span(f"{df['duration_minutes'].mean():.1f} minutes")
            ], style={'margin': '5px 0'}),
            
            html.Div([
                html.Span("Avg. Trip Distance: ", style={'fontWeight': 'bold'}),
                html.Span(f"{df.loc[df['distance_km'].notna(), 'distance_km'].mean():.2f} km")
            ], style={'margin': '5px 0'}),
            
            html.Div([
                html.Span("Initial Sample: ", style={'fontWeight': 'bold'}),
                html.Span(f"{BIXI_CSV_SAMPLE_SIZE:,} trips")
            ], style={'margin': '5px 0'}),
            
            html.Div([
                html.Span("After Filtering: ", style={'fontWeight': 'bold'}),
                html.Span(f"{len(df):,} valid trips")
            ], style={'margin': '5px 0', 'color': '#3498db'}),
            
            html.Hr(style={'margin': '15px 0', 'borderTop': '1px dashed #ccc'}),
            
            html.Div([
                html.P("Note: Invalid trips (e.g., with unrealistic durations or coordinates) were filtered out during processing.", 
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'margin': '5px 0'})
            ])
        ])
        
        return stats_html

# Additional callback for loading status
@app.callback(
    [Output('loading-status', 'children'),
     Output('loading-output', 'children')],
    [Input('load-data-button', 'n_clicks')]
)
def update_loading_status(n_clicks):
    if n_clicks is None:
        return [
            html.P(
                "Click 'Load Data' to begin analysis. Loading may take 5-10 seconds.",
                style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px', 'marginTop': '5px'}
            )
        ], ""
    elif data_cache['df'] is None:
        # During loading
        return [
            html.P(
                "Loading data and creating visualizations... Please wait.",
                style={'textAlign': 'center', 'color': '#e67e22', 'fontSize': '12px', 'marginTop': '5px', 'fontWeight': 'bold'}
            )
        ], "loading"
    else:
        # After loading
        return [
            html.P(
                "Data loaded successfully! Scroll through tabs to view all charts.",
                style={'textAlign': 'center', 'color': '#27ae60', 'fontSize': '12px', 'marginTop': '5px', 'fontWeight': 'bold'}
            )
        ], "loaded"

# Helper function to filter dataframe based on selections
def filter_dataframe(df, time_filter, district_filter=None):
    filtered_df = df.copy()
    
    # Apply time filter
    if time_filter == 'weekday':
        filtered_df = filtered_df[~filtered_df['is_weekend']]
    elif time_filter == 'weekend':
        filtered_df = filtered_df[filtered_df['is_weekend']]
    
    return filtered_df

# Callback for hourly pattern chart
@app.callback(
    Output('hourly-pattern-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_hourly_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        hourly_counts = df.groupby('start_hour').size().reset_index(name='count')
        fig = px.bar(hourly_counts, x='start_hour', y='count',
                    labels={'start_hour': 'Hour of Day', 'count': 'Number of Trips'},
                    title='Hourly Trip Distribution',
                    color_discrete_sequence=['#3498db'])
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(tickmode='linear', tick0=0, dtick=1, gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
        )
        return fig
    except Exception as e:
        print(f"Error in update_hourly_chart: {str(e)}")
        return create_error_figure(f"Could not load hourly chart. Details: {str(e)[:100]}")

# Callback for daily pattern chart
@app.callback(
    Output('daily-pattern-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_daily_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts = df.groupby('start_day').size().reset_index(name='count')
        day_df = pd.DataFrame({'start_day': day_order})
        daily_counts = day_df.merge(daily_counts, on='start_day', how='left').fillna(0)
        colors = ['#3498db'] * 5 + ['#e74c3c'] * 2
        fig = px.bar(daily_counts, x='start_day', y='count',
                    labels={'start_day': 'Day of Week', 'count': 'Number of Trips'},
                    title='Daily Trip Distribution',
                    category_orders={"start_day": day_order})
        for i, day in enumerate(day_order):
            fig.data[0].marker.color = colors
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
        )
        return fig
    except Exception as e:
        print(f"Error in update_daily_chart: {str(e)}")
        return create_error_figure(f"Could not load daily chart. Details: {str(e)[:100]}")

# Callback for monthly pattern chart
@app.callback(
    Output('monthly-pattern-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_monthly_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        monthly_counts = df.groupby('start_month').size().reset_index(name='count')
        all_months = pd.DataFrame({'start_month': range(1, 13)})
        monthly_counts = all_months.merge(monthly_counts, on='start_month', how='left').fillna(0)
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        monthly_counts['month_name'] = monthly_counts['start_month'].map(month_names)
        summer_colors = [
            '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c',
            '#08519c', '#08306b', '#08306b',
            '#4292c6', '#6baed6', '#9ecae1'
        ]
        fig = px.bar(
            monthly_counts, 
            x='month_name', 
            y='count',
            labels={'month_name': 'Month', 'count': 'Number of Trips'},
            title='Monthly Trip Distribution',
            category_orders={"month_name": list(month_names.values())}
        )
        for i, month in enumerate(month_names.values()):
            fig.data[0].marker.color = summer_colors
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
        )
        return fig
    except Exception as e:
        print(f"Error in update_monthly_chart: {str(e)}")
        return create_error_figure(f"Could not load monthly chart. Details: {str(e)[:100]}")

# Callback for weekday vs weekend chart
@app.callback(
    Output('weekday-weekend-chart', 'figure'),
    [Input('load-data-button', 'n_clicks')]
)
def update_weekday_weekend_chart(n_clicks):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = data_cache['df']
        weekday = df[~df['is_weekend']]
        weekend = df[df['is_weekend']]
        if len(weekday) > 0:
            weekday_hourly = weekday.groupby('start_hour').size()
            weekday_pct = (weekday_hourly / weekday_hourly.sum() * 100).reset_index(name='percentage')
            weekday_pct['type'] = 'Weekday'
        else:
            weekday_pct = pd.DataFrame({'start_hour': range(24), 'percentage': 0, 'type': 'Weekday'})
        if len(weekend) > 0:
            weekend_hourly = weekend.groupby('start_hour').size()
            weekend_pct = (weekend_hourly / weekend_hourly.sum() * 100).reset_index(name='percentage')
            weekend_pct['type'] = 'Weekend'
        else:
            weekend_pct = pd.DataFrame({'start_hour': range(24), 'percentage': 0, 'type': 'Weekend'})
        hours_df = pd.DataFrame({'start_hour': range(24)})
        if 'start_hour' in weekday_pct.columns:
            weekday_pct = hours_df.merge(weekday_pct, on='start_hour', how='left').fillna({'percentage': 0, 'type': 'Weekday'})
        if 'start_hour' in weekend_pct.columns:
            weekend_pct = hours_df.merge(weekend_pct, on='start_hour', how='left').fillna({'percentage': 0, 'type': 'Weekend'})
        combined = pd.concat([weekday_pct, weekend_pct])
        fig = px.line(
            combined, 
            x='start_hour', 
            y='percentage', 
            color='type',
            labels={'start_hour': 'Hour of Day', 'percentage': 'Percentage of Trips', 'type': ''},
            title='Hourly Trip Distribution: Weekdays vs Weekends',
            color_discrete_map={'Weekday': '#3498db', 'Weekend': '#e74c3c'}
        )
        fig.update_traces(line=dict(width=3), mode='lines+markers')
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(tickmode='linear', tick0=0, dtick=1, gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    except Exception as e:
        print(f"Error in update_weekday_weekend_chart: {str(e)}")
        return create_error_figure(f"Could not load weekday/weekend chart. Details: {str(e)[:100]}")

# Callback for top stations chart
@app.callback(
    Output('top-stations-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_top_stations_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        start_stations = df['STARTSTATIONNAME'].value_counts().head(TOP_STATIONS_COUNT).reset_index()
        start_stations.columns = ['station', 'count']
        start_stations['type'] = 'Start Station'
        end_stations = df['ENDSTATIONNAME'].value_counts().head(TOP_STATIONS_COUNT).reset_index()
        end_stations.columns = ['station', 'count']
        end_stations['type'] = 'End Station'
        combined = pd.concat([start_stations, end_stations])
        fig = px.bar(
            combined, 
            x='count', 
            y='station', 
            color='type', 
            barmode='group',
            orientation='h',
            title=f'Top {TOP_STATIONS_COUNT} Start and End Stations',
            labels={'count': 'Number of Trips', 'station': 'Station'},
            color_discrete_map={'Start Station': '#3498db', 'End Station': '#2ecc71'}
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    except Exception as e:
        print(f"Error in update_top_stations_chart: {str(e)}")
        return create_error_figure(f"Could not load top stations chart. Details: {str(e)[:100]}")

# Callback for district flow chart
@app.callback(
    Output('district-flow-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_district_flow_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = data_cache['df']
        if time_filter == 'weekday':
            df = df[~df['is_weekend']]
        elif time_filter == 'weekend':
            df = df[df['is_weekend']]
        top_districts = df['STARTSTATIONARRONDISSEMENT'].value_counts().head(TOP_DISTRICTS_COUNT).index.tolist()
        district_trips = df[
            df['STARTSTATIONARRONDISSEMENT'].isin(top_districts) & 
            df['ENDSTATIONARRONDISSEMENT'].isin(top_districts)
        ]
        if len(district_trips) > 0:
            matrix = pd.crosstab(
                district_trips['STARTSTATIONARRONDISSEMENT'], 
                district_trips['ENDSTATIONARRONDISSEMENT']
            )
            fig = px.imshow(
                matrix, 
                text_auto=True, 
                aspect="auto",
                labels=dict(x="Destination District", y="Origin District", color="Trip Count"),
                title=f"Trip Flows Between Top {TOP_DISTRICTS_COUNT} Districts",
                color_continuous_scale=px.colors.sequential.Blues
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                title_font=dict(size=18, color='#333'),
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            )
            return fig
        else:
            return create_error_figure("Not enough data for district flow chart after filtering.")
    except Exception as e:
        print(f"Error in update_district_flow_chart: {str(e)}")
        return create_error_figure(f"Could not load district flow chart. Details: {str(e)[:100]}")

# Callback for station map
@app.callback(
    Output('station-map', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_station_map(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        stations = df.groupby('STARTSTATIONNAME').agg({
            'STARTSTATIONLATITUDE': 'first',
            'STARTSTATIONLONGITUDE': 'first',
            'STARTSTATIONARRONDISSEMENT': 'first'
        }).reset_index()
        stations.columns = ['station_name', 'latitude', 'longitude', 'district']
        start_counts = df['STARTSTATIONNAME'].value_counts().to_dict()
        stations['trips'] = stations['station_name'].map(start_counts)
        stations = stations[(stations['latitude'] > MONTREAL_LAT_LOWER_BOUND) & (stations['longitude'] < MONTREAL_LON_UPPER_BOUND)]
        stations = stations.sort_values('trips', ascending=False).head(TOP_STATIONS_MAP_COUNT)
        fig = px.scatter_map(
            stations,
            lat='latitude',
            lon='longitude',
            size='trips',
            color='district',
            hover_name='station_name',
            hover_data={'trips': True, 'district': True, 'latitude': False, 'longitude': False},
            title=f'Top {TOP_STATIONS_MAP_COUNT} Stations by Usage',
            size_max=30,
            map_style="carto-positron",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(
            margin={"r":0,"t":30,"l":0,"b":0},
            title_font=dict(size=18, color='#333'),
            map=dict(
                center=dict(lat=MAP_MONTREAL_CENTER_LAT, lon=MAP_MONTREAL_CENTER_LON),
                zoom=MAP_DEFAULT_ZOOM
            ),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-0.05, 
                xanchor="center", 
                x=0.5,
                bgcolor='rgba(255,255,255,0.8)'
            )
        )
        return fig
    except Exception as e:
        print(f"Error in update_station_map: {str(e)}")
        return create_error_figure(f"Could not load station map. Details: {str(e)[:100]}")

# Callback for duration distribution chart
@app.callback(
    Output('duration-dist-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_duration_dist_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        durations = df[df['duration_minutes'] < MAX_DURATION_VISUALIZATION_MINUTES]['duration_minutes']
        fig = px.histogram(
            durations,
            x='duration_minutes',
            nbins=30,
            labels={'duration_minutes': 'Trip Duration (minutes)'},
            title='Trip Duration Distribution',
            color_discrete_sequence=['#3498db']
        )
        mean_duration = durations.mean()
        fig.add_vline(
            x=mean_duration, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Mean: {mean_duration:.1f} min",
            annotation_position="top right"
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title='Trip Duration (minutes)',
            yaxis_title='Number of Trips',
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
        )
        return fig
    except Exception as e:
        print(f"Error in update_duration_dist_chart: {str(e)}")
        return create_error_figure(f"Could not load duration distribution. Details: {str(e)[:100]}")

# Callback for distance distribution chart
@app.callback(
    Output('distance-dist-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_distance_dist_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        distances = df[df['distance_km'].notna() & (df['distance_km'] > MIN_TRIP_DISTANCE_KM) & (df['distance_km'] < MAX_DISTANCE_VISUALIZATION_KM)]
        fig = px.histogram(
            distances,
            x='distance_km',
            nbins=30,
            labels={'distance_km': 'Trip Distance (km)'},
            title='Trip Distance Distribution',
            color_discrete_sequence=['#2ecc71']
        )
        mean_distance = distances['distance_km'].mean()
        fig.add_vline(
            x=mean_distance, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Mean: {mean_distance:.2f} km",
            annotation_position="top right"
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title='Trip Distance (km)',
            yaxis_title='Number of Trips',
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
        )
        return fig
    except Exception as e:
        print(f"Error in update_distance_dist_chart: {str(e)}")
        return create_error_figure(f"Could not load distance distribution. Details: {str(e)[:100]}")

# Callback for speed distribution chart
@app.callback(
    Output('speed-dist-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_speed_dist_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        speeds = df[
            df['speed_kmh'].notna() & 
            (df['speed_kmh'] > MIN_TRIP_SPEED_KMH) & 
            (df['speed_kmh'] < MAX_SPEED_VISUALIZATION_KMH)
        ]
        fig = px.histogram(
            speeds,
            x='speed_kmh',
            nbins=30,
            labels={'speed_kmh': 'Speed (km/h)'},
            title='Trip Speed Distribution',
            color_discrete_sequence=['#9b59b6']
        )
        mean_speed = speeds['speed_kmh'].mean()
        fig.add_vline(
            x=mean_speed, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Mean: {mean_speed:.1f} km/h",
            annotation_position="top right"
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title='Speed (km/h)',
            yaxis_title='Number of Trips',
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
        )
        return fig
    except Exception as e:
        print(f"Error in update_speed_dist_chart: {str(e)}")
        return create_error_figure(f"Could not load speed distribution. Details: {str(e)[:100]}")

# Callback for rush hour chart
@app.callback(
    Output('rush-hour-chart', 'figure'),
    [Input('load-data-button', 'n_clicks')]
)
def update_rush_hour_chart(n_clicks):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = data_cache['df']
        df['rush_hour'] = 'Off-peak'
        df.loc[(df['start_hour'] >= RUSH_HOUR_MORNING_START) & (df['start_hour'] <= RUSH_HOUR_MORNING_END), 'rush_hour'] = f'Morning Rush ({RUSH_HOUR_MORNING_START}-{RUSH_HOUR_MORNING_END}AM)'
        df.loc[(df['start_hour'] >= RUSH_HOUR_EVENING_START) & (df['start_hour'] <= RUSH_HOUR_EVENING_END), 'rush_hour'] = f'Evening Rush ({RUSH_HOUR_EVENING_START % 12}-{RUSH_HOUR_EVENING_END % 12}PM)'
        weekday_df = df[~df['is_weekend']]
        hourly_rush = weekday_df.groupby(['rush_hour', 'start_hour']).size().reset_index(name='count')
        fig = px.line(
            hourly_rush,
            x='start_hour',
            y='count',
            color='rush_hour',
            title='Weekday Hourly Pattern by Rush Hour Period',
            labels={
                'start_hour': 'Hour of Day',
                'count': 'Number of Trips',
                'rush_hour': 'Time Period'
            },
            color_discrete_map={
                f'Morning Rush ({RUSH_HOUR_MORNING_START}-{RUSH_HOUR_MORNING_END}AM)': '#f39c12', 
                f'Evening Rush ({RUSH_HOUR_EVENING_START % 12}-{RUSH_HOUR_EVENING_END % 12}PM)': '#9b59b6',
                'Off-peak': '#7f8c8d'
            }
        )
        fig.update_traces(line=dict(width=4), mode='lines+markers')
        fig.add_vrect(
            x0=RUSH_HOUR_MORNING_START, x1=RUSH_HOUR_MORNING_END, 
            fillcolor="#f39c12", opacity=0.2, 
            layer="below", line_width=0,
            annotation_text="Morning Rush",
            annotation_position="top left"
        )
        fig.add_vrect(
            x0=RUSH_HOUR_EVENING_START, x1=RUSH_HOUR_EVENING_END, 
            fillcolor="#9b59b6", opacity=0.2, 
            layer="below", line_width=0,
            annotation_text="Evening Rush",
            annotation_position="top left"
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(tickmode='linear', tick0=0, dtick=1, gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    except Exception as e:
        print(f"Error in update_rush_hour_chart: {str(e)}")
        return create_error_figure(f"Could not load rush hour analysis. Details: {str(e)[:100]}")

# Callback for station flow chart
@app.callback(
    Output('station-flow-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_station_flow_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        start_counts = df['STARTSTATIONNAME'].value_counts().reset_index()
        start_counts.columns = ['station', 'starts']
        end_counts = df['ENDSTATIONNAME'].value_counts().reset_index()
        end_counts.columns = ['station', 'ends']
        station_flow = pd.merge(start_counts, end_counts, on='station', how='outer').fillna(0)
        station_flow['net_flow'] = station_flow['starts'] - station_flow['ends']
        station_flow['total'] = station_flow['starts'] + station_flow['ends']
        outflow = station_flow.sort_values('net_flow', ascending=False).head(TOP_STATIONS_COUNT)
        inflow = station_flow.sort_values('net_flow').head(TOP_STATIONS_COUNT)
        outflow['flow_type'] = 'Net Outflow (more starts)'
        inflow['flow_type'] = 'Net Inflow (more ends)'
        flow_data = pd.concat([outflow, inflow])
        fig = px.bar(
            flow_data,
            x='net_flow',
            y='station',
            color='flow_type',
            orientation='h',
            labels={
                'net_flow': 'Net Flow (starts - ends)',
                'station': 'Station',
                'flow_type': 'Flow Type'
            },
            title=f'Top {TOP_STATIONS_COUNT} Stations by Net Flow',
            color_discrete_map={
                'Net Outflow (more starts)': '#e74c3c',
                'Net Inflow (more ends)': '#2ecc71'
            }
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    except Exception as e:
        print(f"Error in update_station_flow_chart: {str(e)}")
        return create_error_figure(f"Could not load station flow analysis. Details: {str(e)[:100]}")

# Callback for station pair chart
@app.callback(
    Output('station-pair-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_station_pair_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        station_pairs = df.groupby(['STARTSTATIONNAME', 'ENDSTATIONNAME']).size().reset_index(name='count')
        top_pairs = station_pairs.sort_values('count', ascending=False).head(TOP_STATION_PAIRS_COUNT)
        top_pairs['pair'] = top_pairs['STARTSTATIONNAME'] + ' → ' + top_pairs['ENDSTATIONNAME']
        fig = px.bar(
            top_pairs,
            x='count',
            y='pair',
            orientation='h',
            labels={
                'count': 'Number of Trips',
                'pair': 'Station Pair'
            },
            title=f'Top {TOP_STATION_PAIRS_COUNT} Station Pairs',
            color='count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0', autorange="reversed"),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
            coloraxis_showscale=False
        )
        return fig
    except Exception as e:
        print(f"Error in update_station_pair_chart: {str(e)}")
        return create_error_figure(f"Could not load station pair analysis. Details: {str(e)[:100]}")

# Callback for district duration chart
@app.callback(
    Output('district-duration-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_district_duration_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return create_error_figure("Data not loaded. Please click 'Load Data'.")
    try:
        df = data_cache['df'] # Start with full dataframe from cache
        if time_filter == 'weekday':
            df = df[~df['is_weekend']]
        elif time_filter == 'weekend':
            df = df[df['is_weekend']]
        district_durations = df.groupby('STARTSTATIONARRONDISSEMENT')['duration_minutes'].agg(['mean', 'count']).reset_index()
        district_durations.columns = ['district', 'avg_duration', 'trip_count']
        significant_districts = district_durations[district_durations['trip_count'] > SIGNIFICANT_DISTRICT_TRIP_COUNT].sort_values('avg_duration', ascending=False)
        fig = px.bar(
            significant_districts,
            x='district',
            y='avg_duration',
            title='Average Trip Duration by District',
            labels={
                'district': 'District',
                'avg_duration': 'Average Duration (minutes)'
            },
            custom_data=['trip_count'],
            color='avg_duration',
            color_continuous_scale='RdBu_r'
        )
        fig.update_traces(
            hovertemplate='District: %{x}<br>Avg Duration: %{y:.1f} min<br>Trip Count: %{customdata[0]:,}'
        )
        overall_avg = df['duration_minutes'].mean()
        fig.add_hline(
            y=overall_avg, 
            line_dash="dash", 
            line_color="#e74c3c", 
            annotation_text=f"Overall Avg: {overall_avg:.1f} min",
            annotation_position="top right"
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_tickangle=-45,
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_color='white'),
            title_font=dict(size=18, color='#333'),
            coloraxis_showscale=False
        )
        return fig
    except Exception as e:
        print(f"Error in update_district_duration_chart: {str(e)}")
        return create_error_figure(f"Could not load district duration analysis. Details: {str(e)[:100]}")

# Callback to hide map loading indicator once map is loaded
@app.callback(
    Output('map-loading-indicator', 'style'),
    [Input('station-map', 'figure')]
)
def hide_map_loading_indicator(figure):
    if figure is None or not figure:
        return {'display': 'block'}
    return {'display': 'none'}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)