import dash
from dash import dcc, html, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import os
from math import radians, sin, cos, sqrt, atan2

# Get the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, 'Bixi_2023_sample.csv')

# Initialize the Dash app with improved title and metadata
app = dash.Dash(__name__,
                title='Bixi 2023 Analysis Dashboard',
                meta_tags=[{'name': 'viewport',
                           'content': 'width=device-width, initial-scale=1.0'}])

server = app.server

# Pre-load data at startup for better UX
print("Loading Bixi data at startup...")
STARTUP_DATA = None



# Custom CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html lang="en">
    <head>
        {%metas%}
        <title>{%title%}</title>

        <!-- SEO Meta Tags -->
        <meta name="description" content="Interactive dashboard analyzing Montreal's Bixi bike-sharing system data from 2023. Explore trip patterns, popular stations, and usage trends.">
        <meta name="keywords" content="Bixi, Montreal, bike sharing, data visualization, cycling, transportation, dashboard">
        <meta name="author" content="Alek Swiderski">

        <!-- Open Graph Meta Tags -->
        <meta property="og:title" content="Bixi 2023 Analysis Dashboard">
        <meta property="og:description" content="Interactive dashboard exploring Montreal's bike-sharing patterns and trends.">
        <meta property="og:type" content="website">

        <!-- Twitter Card Meta Tags -->
        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:title" content="Bixi 2023 Analysis Dashboard">
        <meta name="twitter:description" content="Interactive dashboard exploring Montreal's bike-sharing patterns and trends.">

        <!-- Favicon -->
        <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚲</text></svg>">

        {%favicon%}
        {%css%}
        <style>
            * {
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', 'Roboto', sans-serif;
                background-color: #f5f7fa;
                margin: 0;
                color: #2c3e50;
            }

            .dashboard-header {
                background: linear-gradient(135deg, #e41e31, #c41e3a);
                color: white;
                padding: 25px 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }

            .kpi-container {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 15px;
                padding: 20px;
                background: linear-gradient(135deg, #2c3e50, #34495e);
            }

            .kpi-card {
                background: white;
                border-radius: 10px;
                padding: 20px 25px;
                text-align: center;
                min-width: 160px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.3s;
            }

            .kpi-card:hover {
                transform: translateY(-3px);
            }

            .kpi-value {
                font-size: 28px;
                font-weight: bold;
                color: #e41e31;
                margin: 5px 0;
            }

            .kpi-label {
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .kpi-subtitle {
                font-size: 11px;
                color: #999;
                margin-top: 3px;
            }

            .filters-panel {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 20px;
                margin: 20px;
                margin-bottom: 0;
            }

            .stats-box {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                margin-top: 20px;
                border-left: 4px solid #e41e31;
            }

            .chart-container {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                padding: 15px;
                transition: transform 0.3s;
            }

            .chart-container:hover {
                transform: translateY(-3px);
            }

            .tab-content {
                padding: 20px;
            }

            /* Loading spinner */
            ._dash-loading {
                margin: 50px auto;
            }

            /* Responsive */
            @media (max-width: 768px) {
                .kpi-card {
                    min-width: 140px;
                    padding: 15px;
                }
                .kpi-value {
                    font-size: 22px;
                }
            }

            /* Better scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #f1f1f1;
            }
            ::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout with improved styling
app.layout = html.Div([
    # Header with gradient background
    html.Div([
        html.H1('Bixi 2023 Data Analysis Dashboard',
                style={'textAlign': 'center', 'color': 'white', 'margin': '0 0 5px 0', 'fontSize': '2rem'}),
        html.P('Exploring Montreal\'s Bike-Sharing Patterns',
               style={'textAlign': 'center', 'color': 'rgba(255,255,255,0.85)', 'margin': '0',
                      'fontSize': '1rem'})
    ], className='dashboard-header'),

    # KPI Cards Row
    html.Div(id='kpi-cards', className='kpi-container'),

    # Filters bar (horizontal, not sidebar)
    html.Div([
        html.Div([
            html.Label('Filter by:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='time-filter',
                options=[
                    {'label': 'All Data', 'value': 'all'},
                    {'label': 'Weekdays Only', 'value': 'weekday'},
                    {'label': 'Weekends Only', 'value': 'weekend'}
                ],
                value='all',
                style={'width': '200px', 'display': 'inline-block'}
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'gap': '10px'}),

        # Hidden elements for backward compatibility
        html.Div(id='dataset-stats', style={'display': 'none'}),
        html.Button('Load Data', id='load-data-button', style={'display': 'none'}),
        html.Div(id='loading-status', style={'display': 'none'}),
        html.Div(id='loading-output', style={'display': 'none'})
    ], className='filters-panel'),

    # Charts panel - full width now
    dcc.Loading(
        id="main-loading",
        type="default",
        children=[
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
        ], className='tab-content')
    ])
])

# Cache for loaded data
data_cache = {
    'df': None
}

# KPI Card helper function
def create_kpi_card(label, value, subtitle=""):
    return html.Div([
        html.Div(label, className='kpi-label'),
        html.Div(value, className='kpi-value'),
        html.Div(subtitle, className='kpi-subtitle') if subtitle else None
    ], className='kpi-card')

# Data loading function (fixed version)
def load_bixi_data(file_path):
    """
    Load a fixed sample of the Bixi dataset with time conversions
    to prevent memory issues with the full 3GB dataset
    """
    # Define the fixed sample size
    SAMPLE_SIZE = 100000
    
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
        'ENDTIMEMS': float
    }
    
    # Load data with error handling
    try:
        # First try with specified dtypes
        df = pd.read_csv(file_path, nrows=SAMPLE_SIZE, dtype=dtypes)
    except Exception as e:
        print(f"Error with specified dtypes: {e}")
        # If that fails, try with automatic dtype inference
        try:
            df = pd.read_csv(file_path, nrows=SAMPLE_SIZE)
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
        df = df[(df['duration_minutes'] > 0) & (df['duration_minutes'] < 1440)]  # Max 24 hours
        
        # Debug info
        print(f"Loaded {len(df)} valid trips from {file_path}")
        
        # Calculate distances for valid coordinates
        # Make sure latitude and longitude are numeric
        for col in ['STARTSTATIONLATITUDE', 'STARTSTATIONLONGITUDE', 'ENDSTATIONLATITUDE', 'ENDSTATIONLONGITUDE']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        valid_coords = (
            (df['STARTSTATIONLATITUDE'] > 40) &  # Valid Montreal latitude 
            (df['STARTSTATIONLONGITUDE'] < -70) & # Valid Montreal longitude
            (df['ENDSTATIONLATITUDE'] > 40) & 
            (df['ENDSTATIONLONGITUDE'] < -70)
        )
        
        df.loc[valid_coords, 'distance_km'] = df.loc[valid_coords].apply(
            lambda row: haversine_distance(
                row['STARTSTATIONLATITUDE'], row['STARTSTATIONLONGITUDE'],
                row['ENDSTATIONLATITUDE'], row['ENDSTATIONLONGITUDE']
            ), axis=1
        )
        
        # Speed calculation
        df.loc[valid_coords, 'speed_kmh'] = (
            df.loc[valid_coords, 'distance_km'] / (df.loc[valid_coords, 'duration_minutes'] / 60)
        )
        
        # Filter realistic speeds and distances
        df = df[(df['distance_km'].isna()) | 
               ((df['distance_km'] >= 0) & (df['distance_km'] < 15))]
        
        df = df[(df['speed_kmh'].isna()) | 
               ((df['speed_kmh'] > 0) & (df['speed_kmh'] < 35))]
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        raise e

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km using Haversine formula"""
    try:
        R = 6371  # Earth radius in km

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c

        return distance
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return None

# AUTO-LOAD DATA AT STARTUP
try:
    print("Auto-loading Bixi data at startup...")
    data_cache['df'] = load_bixi_data(data_file_path)
    print(f"Loaded {len(data_cache['df']):,} valid trips successfully!")
except Exception as e:
    print(f"Error auto-loading data: {e}")
    data_cache['df'] = None

# Callback for KPI cards - updates on startup and filter change
@app.callback(
    Output('kpi-cards', 'children'),
    [Input('time-filter', 'value')]
)
def update_kpi_cards(time_filter):
    if data_cache['df'] is None:
        return [html.Div("Loading data...", style={'color': 'white', 'textAlign': 'center', 'width': '100%'})]

    df = data_cache['df']

    # Apply time filter
    if time_filter == 'weekday':
        df = df[~df['is_weekend']]
    elif time_filter == 'weekend':
        df = df[df['is_weekend']]

    # Calculate KPIs
    total_trips = len(df)
    unique_stations = df['STARTSTATIONNAME'].nunique()
    avg_duration = df['duration_minutes'].mean()
    avg_distance = df.loc[df['distance_km'].notna(), 'distance_km'].mean()
    avg_speed = df.loc[df['speed_kmh'].notna(), 'speed_kmh'].mean()

    # Date range
    date_min = df['start_time'].min().strftime('%b %d')
    date_max = df['start_time'].max().strftime('%b %d, %Y')

    return [
        create_kpi_card("Total Trips", f"{total_trips:,}", "valid trips analyzed"),
        create_kpi_card("Unique Stations", f"{unique_stations:,}", "across Montreal"),
        create_kpi_card("Avg Duration", f"{avg_duration:.1f} min", "per trip"),
        create_kpi_card("Avg Distance", f"{avg_distance:.2f} km", "per trip"),
        create_kpi_card("Avg Speed", f"{avg_speed:.1f} km/h", "cycling speed"),
        create_kpi_card("Date Range", f"{date_min} - {date_max}", "2023 season"),
    ]

# Callback to load data (kept for backward compatibility but auto-loads now)
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
                    html.Span("100,000 trips")
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
                html.Span("100,000 trips")
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
    
    # We've removed the district filter functionality
    
    return filtered_df

# Callback for hourly pattern chart
@app.callback(
    Output('hourly-pattern-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_hourly_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
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

# Callback for daily pattern chart
@app.callback(
    Output('daily-pattern-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_daily_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
    df = filter_dataframe(data_cache['df'], time_filter)
    
    # Order days correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_counts = df.groupby('start_day').size().reset_index(name='count')
    
    # Ensure all days are present
    day_df = pd.DataFrame({'start_day': day_order})
    daily_counts = day_df.merge(daily_counts, on='start_day', how='left').fillna(0)
    
    # Color array to highlight weekends
    colors = ['#3498db'] * 5 + ['#e74c3c'] * 2
    
    fig = px.bar(daily_counts, x='start_day', y='count',
                labels={'start_day': 'Day of Week', 'count': 'Number of Trips'},
                title='Daily Trip Distribution',
                category_orders={"start_day": day_order})
    
    # Update bar colors to highlight weekends
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

# Callback for monthly pattern chart
@app.callback(
    Output('monthly-pattern-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_monthly_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
    df = filter_dataframe(data_cache['df'], time_filter)
    
    try:
        # Count trips by month
        monthly_counts = df.groupby('start_month').size().reset_index(name='count')
        
        # Ensure all months are represented
        all_months = pd.DataFrame({'start_month': range(1, 13)})
        monthly_counts = all_months.merge(monthly_counts, on='start_month', how='left').fillna(0)
        
        # Add month names for display
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        monthly_counts['month_name'] = monthly_counts['start_month'].map(month_names)
        
        # Create a colorscale that peaks in summer months
        summer_colors = [
            '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c',  # blues for winter/spring
            '#08519c', '#08306b', '#08306b',  # dark blue for summer peak
            '#4292c6', '#6baed6', '#9ecae1'   # back to lighter blues for fall/winter
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
        print(f"Error in monthly chart: {e}")
        # Fallback - use month numbers if month names aren't available
        monthly_counts = df.groupby('start_month').size().reset_index(name='count')
        fig = px.bar(monthly_counts, x='start_month', y='count',
                    labels={'start_month': 'Month', 'count': 'Number of Trips'},
                    title='Monthly Trip Distribution')
        return fig

# Callback for weekday vs weekend chart
@app.callback(
    Output('weekday-weekend-chart', 'figure'),
    [Input('load-data-button', 'n_clicks')]
)
def update_weekday_weekend_chart(n_clicks):
    if data_cache['df'] is None:
        return go.Figure()
    
    try:
        # Use all data
        df = data_cache['df']
        
        # Get weekday and weekend data
        weekday = df[~df['is_weekend']]
        weekend = df[df['is_weekend']]
        
        # Hourly patterns with error handling
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
        
        # Ensure all hours are represented
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
        print(f"Error in weekday/weekend chart: {e}")
        # Return empty figure in case of error
        return go.Figure()

# Callback for top stations chart
@app.callback(
    Output('top-stations-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_top_stations_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        
        # Get top 10 start and end stations
        start_stations = df['STARTSTATIONNAME'].value_counts().head(10).reset_index()
        start_stations.columns = ['station', 'count']
        start_stations['type'] = 'Start Station'
        
        end_stations = df['ENDSTATIONNAME'].value_counts().head(10).reset_index()
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
            title='Top 10 Start and End Stations',
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
        print(f"Error in top stations chart: {e}")
        return go.Figure()

# Callback for district flow chart
@app.callback(
    Output('district-flow-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_district_flow_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
    try:
        # Apply time filter
        df = data_cache['df']
        if time_filter == 'weekday':
            df = df[~df['is_weekend']]
        elif time_filter == 'weekend':
            df = df[df['is_weekend']]
        
        # Get top 5 districts
        top_districts = df['STARTSTATIONARRONDISSEMENT'].value_counts().head(5).index.tolist()
        
        # Filter for major districts
        district_trips = df[
            df['STARTSTATIONARRONDISSEMENT'].isin(top_districts) & 
            df['ENDSTATIONARRONDISSEMENT'].isin(top_districts)
        ]
        
        # If we have district trips, create the flow matrix
        if len(district_trips) > 0:
            # Create district flow matrix
            matrix = pd.crosstab(
                district_trips['STARTSTATIONARRONDISSEMENT'], 
                district_trips['ENDSTATIONARRONDISSEMENT']
            )
            
            # Convert to plotly heatmap
            fig = px.imshow(
                matrix, 
                text_auto=True, 
                aspect="auto",
                labels=dict(x="Destination District", y="Origin District", color="Trip Count"),
                title="Trip Flows Between Top 5 Districts",
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
            return go.Figure()
    except Exception as e:
        print(f"Error in district flow chart: {e}")
        return go.Figure()

# Callback for station map
@app.callback(
    Output('station-map', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_station_map(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        
        # Create stations dataset
        stations = df.groupby('STARTSTATIONNAME').agg({
            'STARTSTATIONLATITUDE': 'first',
            'STARTSTATIONLONGITUDE': 'first',
            'STARTSTATIONARRONDISSEMENT': 'first'
        }).reset_index()
        
        stations.columns = ['station_name', 'latitude', 'longitude', 'district']
        
        # Calculate station popularity
        start_counts = df['STARTSTATIONNAME'].value_counts().to_dict()
        stations['trips'] = stations['station_name'].map(start_counts)
        
        # Filter for valid coordinates
        stations = stations[(stations['latitude'] > 40) & (stations['longitude'] < -70)]
        
        # Sort by popularity and take top 100
        stations = stations.sort_values('trips', ascending=False).head(100)
        
        # Revert to scatter_mapbox to ensure map loads properly
        # Note: This will show a deprecation warning but ensures functionality
        # TODO: Properly migrate to scatter_map following the guide at: https://plotly.com/python/mapbox-to-maplibre/
        fig = px.scatter_mapbox(
            stations,
            lat='latitude',
            lon='longitude',
            size='trips',
            color='district',
            hover_name='station_name',
            hover_data={'trips': True, 'district': True, 'latitude': False, 'longitude': False},
            title='Top 100 Stations by Usage',
            size_max=30,
            zoom=11,
            mapbox_style="carto-positron",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Update layout with better map configuration
        fig.update_layout(
            margin={"r":0,"t":30,"l":0,"b":0},
            title_font=dict(size=18, color='#333'),
            mapbox=dict(
                center=dict(lat=45.5088, lon=-73.5878),  # Center on Montreal
                zoom=11
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
        print(f"Error in station map: {e}")
        return go.Figure()

# Callback for duration distribution chart
@app.callback(
    Output('duration-dist-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_duration_dist_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        
        # Filter for reasonable durations for visualization
        durations = df[df['duration_minutes'] < 60]['duration_minutes']
        
        fig = px.histogram(
            durations,
            x='duration_minutes',
            nbins=30,
            labels={'duration_minutes': 'Trip Duration (minutes)'},
            title='Trip Duration Distribution',
            color_discrete_sequence=['#3498db']
        )
        
        # Add a line for the mean
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
        print(f"Error in duration distribution chart: {e}")
        return go.Figure()

# Callback for distance distribution chart
@app.callback(
    Output('distance-dist-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_distance_dist_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        
        # Filter for valid distances
        distances = df[df['distance_km'].notna() & (df['distance_km'] > 0) & (df['distance_km'] < 10)]
        
        fig = px.histogram(
            distances,
            x='distance_km',
            nbins=30,
            labels={'distance_km': 'Trip Distance (km)'},
            title='Trip Distance Distribution',
            color_discrete_sequence=['#2ecc71']
        )
        
        # Add a line for the mean
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
        print(f"Error in distance distribution chart: {e}")
        return go.Figure()

# [Removed callback for duration vs distance chart]

# Callback for speed distribution chart
@app.callback(
    Output('speed-dist-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_speed_dist_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        
        # Filter for valid speeds
        speeds = df[
            df['speed_kmh'].notna() & 
            (df['speed_kmh'] > 0) & 
            (df['speed_kmh'] < 30)
        ]
        
        fig = px.histogram(
            speeds,
            x='speed_kmh',
            nbins=30,
            labels={'speed_kmh': 'Speed (km/h)'},
            title='Trip Speed Distribution',
            color_discrete_sequence=['#9b59b6']
        )
        
        # Add a line for the mean
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
        print(f"Error in speed distribution chart: {e}")
        return go.Figure()

# Callback for rush hour chart
@app.callback(
    Output('rush-hour-chart', 'figure'),
    [Input('load-data-button', 'n_clicks')]
)
def update_rush_hour_chart(n_clicks):
    if data_cache['df'] is None:
        return go.Figure()
    
    try:
        df = data_cache['df']
        
        # Define rush hours
        df['rush_hour'] = 'Off-peak'
        df.loc[(df['start_hour'] >= 7) & (df['start_hour'] <= 9), 'rush_hour'] = 'Morning Rush (7-9AM)'
        df.loc[(df['start_hour'] >= 16) & (df['start_hour'] <= 18), 'rush_hour'] = 'Evening Rush (4-6PM)'
        
        # Get weekday data only for rush hour analysis
        weekday_df = df[~df['is_weekend']]
        
        # Group by rush hour and hour
        hourly_rush = weekday_df.groupby(['rush_hour', 'start_hour']).size().reset_index(name='count')
        
        # Create line chart with improved styling
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
                'Morning Rush (7-9AM)': '#f39c12', 
                'Evening Rush (4-6PM)': '#9b59b6',
                'Off-peak': '#7f8c8d'
            }
        )
        
        fig.update_traces(line=dict(width=4), mode='lines+markers')
        
        # Add shaded areas for rush hours
        fig.add_vrect(
            x0=7, x1=9, 
            fillcolor="#f39c12", opacity=0.2, 
            layer="below", line_width=0,
            annotation_text="Morning Rush",
            annotation_position="top left"
        )
        
        fig.add_vrect(
            x0=16, x1=18, 
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
        print(f"Error in rush hour chart: {e}")
        return go.Figure()

# Callback for station flow chart
@app.callback(
    Output('station-flow-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_station_flow_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        
        # Calculate station net flow
        start_counts = df['STARTSTATIONNAME'].value_counts().reset_index()
        start_counts.columns = ['station', 'starts']
        
        end_counts = df['ENDSTATIONNAME'].value_counts().reset_index()
        end_counts.columns = ['station', 'ends']
        
        station_flow = pd.merge(start_counts, end_counts, on='station', how='outer').fillna(0)
        station_flow['net_flow'] = station_flow['starts'] - station_flow['ends']
        station_flow['total'] = station_flow['starts'] + station_flow['ends']
        
        # Get top 10 stations by net outflow and inflow
        outflow = station_flow.sort_values('net_flow', ascending=False).head(10)
        inflow = station_flow.sort_values('net_flow').head(10)
        
        # Combine and add flow direction
        outflow['flow_type'] = 'Net Outflow (more starts)'
        inflow['flow_type'] = 'Net Inflow (more ends)'
        flow_data = pd.concat([outflow, inflow])
        
        # Create bar chart with improved styling
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
            title='Top Stations by Net Flow',
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
        print(f"Error in station flow chart: {e}")
        return go.Figure()

# Callback for station pair chart
@app.callback(
    Output('station-pair-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_station_pair_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
    try:
        df = filter_dataframe(data_cache['df'], time_filter)
        
        # Get top station pairs
        station_pairs = df.groupby(['STARTSTATIONNAME', 'ENDSTATIONNAME']).size().reset_index(name='count')
        top_pairs = station_pairs.sort_values('count', ascending=False).head(15)
        
        # Create combined pair name
        top_pairs['pair'] = top_pairs['STARTSTATIONNAME'] + ' → ' + top_pairs['ENDSTATIONNAME']
        
        # Create horizontal bar chart with gradient color
        fig = px.bar(
            top_pairs,
            x='count',
            y='pair',
            orientation='h',
            labels={
                'count': 'Number of Trips',
                'pair': 'Station Pair'
            },
            title='Top 15 Station Pairs',
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
        print(f"Error in station pair chart: {e}")
        return go.Figure()

# Callback for district duration chart
@app.callback(
    Output('district-duration-chart', 'figure'),
    [Input('load-data-button', 'n_clicks'),
     Input('time-filter', 'value')]
)
def update_district_duration_chart(n_clicks, time_filter):
    if data_cache['df'] is None:
        return go.Figure()
    
    try:
        df = data_cache['df']
        if time_filter == 'weekday':
            df = df[~df['is_weekend']]
        elif time_filter == 'weekend':
            df = df[df['is_weekend']]
        
        # Get average duration by district
        district_durations = df.groupby('STARTSTATIONARRONDISSEMENT')['duration_minutes'].agg(['mean', 'count']).reset_index()
        district_durations.columns = ['district', 'avg_duration', 'trip_count']
        
        # Filter for districts with significant number of trips
        significant_districts = district_durations[district_durations['trip_count'] > 50].sort_values('avg_duration', ascending=False)
        
        # Create bar chart with improved styling
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
        
        # Add overall average as reference line
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
        print(f"Error in district duration chart: {e}")
        return go.Figure()

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
    app.run(debug=True)