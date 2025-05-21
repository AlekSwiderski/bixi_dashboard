import pytest
from app import haversine_distance # Assumes app.py is in project root
import pandas as pd
from app import load_bixi_data
from io import StringIO
import os
from datetime import datetime

# Tests for haversine_distance
def test_haversine_valid_inputs():
    # Montreal to Quebec City (approximate)
    lat1, lon1 = 45.5017, -73.5673 
    lat2, lon2 = 46.8139, -71.2080
    # Expected distance for Montreal (45.5017, -73.5673) to Quebec City (46.8139, -71.2080)
    expected_distance = 233.0289 
    assert haversine_distance(lat1, lon1, lat2, lon2) == pytest.approx(expected_distance, abs=0.001)

def test_haversine_same_point():
    lat, lon = 45.5017, -73.5673
    assert haversine_distance(lat, lon, lat, lon) == pytest.approx(0.0)

def test_haversine_invalid_string_input(capsys):
    assert haversine_distance("not_a_float", -73.5673, 46.8139, -71.2080) is None
    captured = capsys.readouterr()
    assert "Error converting inputs to float" in captured.out
    assert "lat1='not_a_float'" in captured.out # Corrected: f-string uses variable name `lat1_orig`, so output is `lat1=...`
    assert "lon2='-71.208'" in captured.out 

def test_haversine_none_input(capsys):
    assert haversine_distance(None, -73.5673, 46.8139, -71.2080) is None
    captured = capsys.readouterr()
    # This case will raise TypeError when float(None) is attempted, caught by the generic Exception
    assert "Error calculating distance for (None, -73.5673, 46.8139, -71.208)" in captured.out 
    assert "float() argument must be a string or a real number, not 'NoneType'" in captured.out


# Tests for load_bixi_data
@pytest.fixture
def sample_csv_data():
    # CSV data as a string. Includes valid, invalid duration, invalid coords, invalid speed/distance
    csv_content = """STARTSTATIONNAME,STARTSTATIONARRONDISSEMENT,STARTSTATIONLATITUDE,STARTSTATIONLONGITUDE,ENDSTATIONNAME,ENDSTATIONARRONDISSEMENT,ENDSTATIONLATITUDE,ENDSTATIONLONGITUDE,STARTTIMEMS,ENDTIMEMS
StationA,DistrictA,45.5,-73.5,StationB,DistrictB,45.505,-73.505,1672531200000,1672531800000
StationC,DistrictC,45.6,-73.6,StationD,DistrictD,45.605,-73.605,1672534800000,1672534800000
StationE,DistrictE,45.7,-73.7,StationF,DistrictF,45.705,-73.705,1672538400000,1672538460000
StationG,DistrictG,30.0,-70.0,StationH,DistrictH,30.05,-70.05,1672542000000,1672542600000
StationI,DistrictI,45.8,-73.8,StationJ,DistrictJ,46.8,-74.8,1672545600000,1672546200000
StationK,DistrictK,45.9,-73.9,StationL,DistrictL,45.901,-73.901,1672549200000,1672578000000
"""
    # Trip 1 (StationA): Valid (10 min duration). Kept.
    # Trip 2 (StationC): Invalid duration (0 min). Dropped by duration filter.
    # Trip 3 (StationE): Valid duration (1 min). Speed ~37km/h. Dropped by speed filter.
    # Trip 4 (StationG): Invalid coordinates (lat 30.0). distance_km and speed_kmh will be NaN. Kept by filters.
    # Trip 5 (StationI): Valid coordinates, but very large distance (~137km). Dropped by distance filter.
    # Trip 6 (StationK): Valid coordinates, long duration (8 hours), very low speed. Kept.
    return csv_content

@pytest.fixture
def temp_csv_file(tmp_path, sample_csv_data):
    csv_file = tmp_path / "test_bixi_data.csv"
    csv_file.write_text(sample_csv_data)
    return str(csv_file)

def test_load_bixi_data_processing(temp_csv_file, capsys):
    df = load_bixi_data(temp_csv_file)
    
    # Expected 3 rows based on current logic (Trip 1 (A), Trip 4 (G), Trip 6 (K))
    assert len(df) == 3, "DataFrame should contain 3 rows after filtering"
    
    # Test specific columns and values for Trip 1 (StationA)
    trip1_df = df[df['STARTSTATIONNAME'] == 'StationA']
    assert not trip1_df.empty, "Trip 1 (StationA) should be in the DataFrame"
    
    assert trip1_df['start_time'].iloc[0] == pd.Timestamp('2023-01-01 00:00:00')
    assert trip1_df['end_time'].iloc[0] == pd.Timestamp('2023-01-01 00:10:00')
    assert trip1_df['duration_minutes'].iloc[0] == pytest.approx(10.0)
    assert trip1_df['start_hour'].iloc[0] == 0
    assert trip1_df['start_day'].iloc[0] == "Sunday" # 2023-01-01 is a Sunday
    assert 'distance_km' in df.columns
    assert 'speed_kmh' in df.columns
    
    # Check dtypes
    assert pd.api.types.is_datetime64_any_dtype(df['start_time'])
    assert pd.api.types.is_datetime64_any_dtype(df['end_time'])
    assert pd.api.types.is_float_dtype(df['duration_minutes'])
    assert pd.api.types.is_integer_dtype(df['start_hour']) # Pandas Int64Dtype can be pd.NA

    # Check presence of new columns from data loading
    expected_cols = ['start_time', 'end_time', 'duration_seconds', 'duration_minutes', 
                     'start_hour', 'start_day', 'start_date', 'start_month', 
                     'start_month_name', 'is_weekend', 'distance_km', 'speed_kmh']
    for col in expected_cols:
        assert col in df.columns, f"Column '{col}' should be in DataFrame"

    # Check logging output
    captured = capsys.readouterr()
    assert "Data before duration filtering: 6 rows" in captured.out
    assert "Data after duration filtering: 5 rows (1 rows dropped)" in captured.out
    assert "Data before coordinate filtering: 5 rows" in captured.out 
    assert "Data after distance filtering: 4 rows (1 rows dropped)" in captured.out
    assert "Data after speed filtering: 3 rows (1 rows dropped)" in captured.out
    assert "Finished data cleaning. Final dataset: 3 rows" in captured.out
