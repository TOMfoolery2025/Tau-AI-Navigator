import pandas as pd
import os
from collections import Counter

GTFS_PATH = "/app/import_stage"

def load_static_lookups():
    print("📂 Loading Static GTFS Data...")
    
    # 1. ROBUST MODE MAPPING
    MODE_MAP = {
        '0': 'TRAM', '900': 'TRAM', 
        '1': 'METRO', '400': 'METRO', '401': 'METRO',
        '2': 'TRAIN', '100': 'TRAIN', '109': 'TRAIN',
        '4': 'FERRY', '1000': 'FERRY',
        '3': 'BUS', '700': 'BUS', '701': 'BUS', '702': 'BUS', '704': 'BUS'
    }
    
    routes_dict = {}
    try:
        routes = pd.read_csv(os.path.join(GTFS_PATH, "routes.txt"), dtype=str)
        for _, row in routes.iterrows():
            r_id = row['route_id'].replace("HSL:", "").strip()
            r_type = str(row.get('route_type', '3'))
            routes_dict[r_id] = {
                "short": row.get('route_short_name', r_id),
                "long": row.get('route_long_name', ''),
                "mode": MODE_MAP.get(r_type, 'BUS')
            }
    except: pass

    # 2. SMART DIRECTION LOOKUP
    trip_lookup = {}
    dir_counters = {}     
    direction_lookup = {} 
    
    try:
        trips = pd.read_csv(os.path.join(GTFS_PATH, "trips.txt"), dtype=str)
        for _, row in trips.iterrows():
            t_id = row['trip_id'].replace("HSL:", "").strip()
            r_id = row['route_id'].replace("HSL:", "").strip()
            d_id = str(row.get('direction_id', '0'))
            headsign = row.get('trip_headsign', 'Unknown')
            
            trip_lookup[t_id] = headsign
            
            key = (r_id, d_id)
            if key not in dir_counters: dir_counters[key] = Counter()
            dir_counters[key][headsign] += 1
            
        for key, counter in dir_counters.items():
            direction_lookup[key] = counter.most_common(1)[0][0]
    except: pass

    return routes_dict, trip_lookup, direction_lookup