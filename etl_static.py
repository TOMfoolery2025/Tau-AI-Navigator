import pandas as pd
import os

GTFS_PATH = "/app/import_stage"

def load_static_lookups():
    print("📂 Loading Static GTFS Data (Multi-Modal)...")
    
    # GTFS Route Types Standard:
    # 0=Tram, 1=Subway, 3=Bus, 4=Ferry, 109=Train (HSL specific)
    MODE_MAP = {
        '0': 'TRAM', '900': 'TRAM',
        '1': 'METRO', '401': 'METRO',
        '3': 'BUS', '700': 'BUS',
        '4': 'FERRY', '1000': 'FERRY',
        '109': 'TRAIN', '100': 'TRAIN'
    }

    try:
        routes = pd.read_csv(os.path.join(GTFS_PATH, "routes.txt"), dtype=str)
        routes_dict = {}
        for _, row in routes.iterrows():
            r_id = row['route_id'].replace("HSL:", "")
            r_type_code = str(row.get('route_type', '3'))
            
            # Map code to string (e.g. "1" -> "METRO")
            mode_str = MODE_MAP.get(r_type_code, 'BUS') 
            
            routes_dict[r_id] = {
                "short": row['route_short_name'],
                "long": row['route_long_name'],
                "mode": mode_str 
            }
        print(f"   -> Loaded {len(routes_dict)} routes (Buses, Trams, Trains, Ferries).")
    except Exception as e:
        print(f"   ❌ Error loading routes.txt: {e}")
        routes_dict = {}

    try:
        trips = pd.read_csv(os.path.join(GTFS_PATH, "trips.txt"), dtype=str)
        trip_lookup = {}
        direction_lookup = {}
        
        for _, row in trips.iterrows():
            t_id = row['trip_id'].replace("HSL:", "")
            r_id = row['route_id'].replace("HSL:", "")
            d_id = row['direction_id']
            headsign = row['trip_headsign']
            
            trip_lookup[t_id] = headsign
            direction_lookup[(r_id, d_id)] = headsign
            
        print(f"   -> Loaded {len(trip_lookup)} trips.")
    except Exception as e:
        print(f"   ❌ Error loading trips.txt: {e}")
        trip_lookup = {}
        direction_lookup = {}

    return routes_dict, trip_lookup, direction_lookup