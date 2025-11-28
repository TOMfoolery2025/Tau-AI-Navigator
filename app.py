import streamlit as st
import os
import time
import json
import requests
import pandas as pd
import pydeck as pdk
import datetime
from groq import Groq
from google.transit import gtfs_realtime_pb2
from neo4j import GraphDatabase
from streamlit_js_eval import get_geolocation

# Import ETL scripts
from etl_neo4j import run_neo4j_import
from etl_rdf import generate_rdf_file
from etl_enrich import run_enrichment

# --- 1. CONFIG & CSS ---
st.set_page_config(layout="wide", page_title="Helsinki AI Navigator", page_icon="🇫🇮")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    /* ANIMATED GRADIENT TEXT */
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .hero-title {
        font-weight: 800; font-size: 3.5rem;
        background: linear-gradient(-45deg, #00c6ff, #0072ff, #00c6ff, #29ffc6);
        background-size: 300%;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: gradient 6s ease infinite;
        margin-bottom: 0px; letter-spacing: -1px;
    }

    /* GLASS CARDS */
    .glass-card {
        background: rgba(18, 18, 25, 0.7);
        backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px; padding: 25px; margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* WEATHER BADGE */
    .weather-badge {
        background: rgba(255, 255, 255, 0.1);
        padding: 5px 12px; border-radius: 20px;
        font-size: 0.9rem; color: #fff; font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
        display: inline-flex; align-items: center; gap: 5px;
    }

    /* METRICS */
    .metric-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px; padding: 15px; text-align: center;
        transition: transform 0.2s;
        height: 100%; /* Ensure uniform height */
    }
    .metric-container:hover { transform: translateY(-3px); background: rgba(255, 255, 255, 0.06); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
    .metric-sub { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }

    /* TIMELINE */
    .step-card { display: flex; align-items: flex-start; margin-bottom: 25px; position: relative; }
    .step-line { position: absolute; left: 14px; top: 30px; bottom: -35px; width: 2px; background: rgba(255, 255, 255, 0.1); }
    .step-icon {
        width: 30px; height: 30px; border-radius: 50%;
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; font-weight: bold; color: white; z-index: 2;
        box-shadow: 0 0 15px rgba(0, 114, 255, 0.5); flex-shrink: 0;
    }
    .step-content { margin-left: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIG ---
def clean_key(key):
    if key: return key.strip().replace('"', '').replace("'", "")
    return None

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123"))
GROQ_KEY = clean_key(os.getenv("GROQ_API_KEY"))
HSL_KEY = clean_key(os.getenv("DIGITRANSIT_API_KEY"))

@st.cache_resource
def get_driver():
    try: return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    except: return None
driver = get_driver()

def get_route_color(route_short_name):
    palette = {
        "1": [255, 87, 34], "2": [76, 175, 80], "3": [33, 150, 243], 
        "4": [255, 193, 7], "5": [156, 39, 176], "6": [233, 30, 99],
        "7": [0, 188, 212], "8": [63, 81, 181], "9": [205, 220, 57], "10": [121, 85, 72]
    }
    return palette.get(str(route_short_name), [0, 223, 216])

# --- 3. DATA FEATURES ---

def get_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=60.17&longitude=24.94&current=temperature_2m,weather_code&wind_speed_unit=kmh"
        res = requests.get(url).json()
        temp = res['current']['temperature_2m']
        code = res['current']['weather_code']
        condition = "Clear"
        if code > 3: condition = "Cloudy"
        if code > 50: condition = "Rainy"
        return {"temp": temp, "condition": condition}
    except: return {"temp": 10, "condition": "Unknown"}

def get_crowding_status():
    hour = datetime.datetime.now().hour
    if (7 <= hour <= 9) or (15 <= hour <= 18): return ("High", "🔴")
    elif (10 <= hour <= 14): return ("Medium", "🟡")
    else: return ("Low", "🟢")

def calculate_co2_savings(walking_dist_meters):
    """
    Calculates CO2 saved by taking Tram/Walk vs Car.
    Assumption: Typical Helsinki city tour leg is ~4.5 km.
    Car CO2: ~150g/km (City traffic)
    Tram CO2: 0g/km (HSL is green electricity)
    """
    trip_km = 4.5 + (walking_dist_meters / 1000.0)
    car_emissions = trip_km * 150 # grams
    return int(car_emissions)

# --- CORE LOGIC ---

def get_boarding_info(user_lat, user_lon, target_routes_list):
    query = """
    {
      nearest(lat: %f, lon: %f, maxDistance: 1000, filterByPlaceTypes: STOP) {
        edges {
          node {
            place {
              ... on Stop {
                name
                lat
                lon
                stoptimesWithoutPatterns(numberOfDepartures: 20) {
                  realtimeArrival
                  serviceDay
                  trip { route { shortName } }
                }
              }
            }
            distance
          }
        }
      }
    }
    """ % (user_lat, user_lon)
    headers = {"Content-Type": "application/json", "digitransit-subscription-key": HSL_KEY}
    try:
        resp = requests.post("https://api.digitransit.fi/routing/v2/hsl/gtfs/v1", json={"query": query}, headers=headers)
        data = resp.json()
        candidates = data.get('data', {}).get('nearest', {}).get('edges', [])
        best_option = None
        min_wait = 999
        for item in candidates:
            stop = item['node']['place']
            distance = item['node']['distance']
            for departure in stop['stoptimesWithoutPatterns']:
                route_name = departure['trip']['route']['shortName']
                if route_name in target_routes_list:
                    arrival_seconds = departure['realtimeArrival']
                    service_day = departure['serviceDay']
                    minutes_away = int(((service_day + arrival_seconds) - time.time()) / 60)
                    if 0 <= minutes_away < min_wait:
                        min_wait = minutes_away
                        walk_time = max(1, int(distance / 80))
                        best_option = {
                            "stop_name": stop['name'],
                            "distance": int(distance),
                            "walk_time": walk_time,
                            "minutes_away": minutes_away,
                            "lat": stop['lat'],
                            "lon": stop['lon'],
                            "route": route_name
                        }
        return best_option
    except: return None

def get_ai_tour(interest, weather):
    if not GROQ_KEY: return None
    client = Groq(api_key=GROQ_KEY)
    w_inst = "It is RAINING. Minimize walking." if "Rain" in weather['condition'] else "Weather is nice."
    prompt = f"""
    Create a 3-stop Helsinki tour based on: '{interest}'.
    Weather: {weather['condition']}, {weather['temp']}C. {w_inst}
    Prioritize TRAMS (4, 7, 2, 10). Output STRICT JSON.
    Schema:
    {{
        "title": "Tour Name",
        "stops": [
            {{ "name": "Stop Name", "desc": "Desc", "get_off_stop": "Exit Stop Name" }}
        ],
        "suggested_routes": ["4", "7"],
        "mode": "TRAM" 
    }}
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "JSON only."}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except: return None

def resolve_route_ids(route_names_list, mode):
    resolved_data = {}
    headers = {"Content-Type": "application/json", "digitransit-subscription-key": HSL_KEY}
    api_mode = mode.upper()
    for name in route_names_list:
        query = """{ routes(name: "%s", transportModes: %s) { gtfsId shortName } }""" % (name, api_mode)
        try:
            resp = requests.post("https://api.digitransit.fi/routing/v2/hsl/gtfs/v1", json={"query": query}, headers=headers)
            data = resp.json().get("data", {}).get("routes", [])
            for r in data:
                if r['shortName'] == name:
                    resolved_data[r['gtfsId']] = r['shortName']
                    break
        except: continue
    return resolved_data

def get_live_vehicles(route_id_map):
    feed = gtfs_realtime_pb2.FeedMessage()
    headers = {"digitransit-subscription-key": HSL_KEY}
    try:
        resp = requests.get("https://realtime.hsl.fi/realtime/vehicle-positions/v2/hsl", headers=headers, timeout=5)
        feed.ParseFromString(resp.content)
    except: return pd.DataFrame()

    vehicles = []
    targets_clean = {rid.replace("HSL:", ""): name for rid, name in route_id_map.items()}
    
    for entity in feed.entity:
        if entity.HasField('vehicle'):
            v = entity.vehicle
            raw_route_id = v.trip.route_id
            matched_name = None
            if raw_route_id:
                for target_id, short_name in targets_clean.items():
                    if target_id in raw_route_id:
                        matched_name = short_name
                        break
            if matched_name:
                base_color = get_route_color(matched_name)
                vehicles.append({
                    "lat": v.position.latitude,
                    "lon": v.position.longitude,
                    "id": v.vehicle.label,
                    "route_name": matched_name,
                    "color": base_color
                })
    return pd.DataFrame(vehicles)

# --- 4. UI ---

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Helsinki_vaakuna.svg/1200px-Helsinki_vaakuna.svg.png", width=50)
    st.markdown("### **System Status**")
    if driver: st.success("Database: Connected")
    else: st.error("Database: Offline")
    st.markdown("---")
    if st.button("🔄 Reload Graph"):
        if driver:
            run_neo4j_import(driver, HSL_KEY)
            st.toast("Knowledge Graph Updated!", icon="✅")
    if st.button("🔄 Reload Graph & Enrich"):
        if driver:
            # 1. Run Standard ETL
            run_neo4j_import(driver, HSL_KEY)
            
            # 2. Run RDF Generation
            rdf_file = generate_rdf_file(HSL_KEY)
            
            # 3. Run Semantic Enrichment (The "Winning" Step)
            run_enrichment(driver)
            
            st.toast("Knowledge Graph Enriched!", icon="🧠")

col_h1, col_h2 = st.columns([3, 1])
weather = get_weather()

with col_h1:
    st.markdown('<h1 class="hero-title">HELSINKI<br>AI NAVIGATOR</h1>', unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style="text-align:right; margin-top:20px;">
        <div class="weather-badge">
            <span>{weather['condition']}</span>
            <span style="color: #00dfd8;">{weather['temp']}°C</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<p style="font-size: 1.2rem; color: #aaa; margin-bottom: 40px;">Multimodal travel companion powered by Knowledge Graphs & Live Data.</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 2], gap="large")

with col_left:
    st.markdown("### 📍 **Journey Planner**")
    
    loc_data = get_geolocation()
    default_lat, default_lon = 60.1710, 24.9410
    
    if loc_data and 'coords' in loc_data:
        user_lat = loc_data['coords']['latitude']
        user_lon = loc_data['coords']['longitude']
        st.success(f"📍 GPS Active: {user_lat:.4f}, {user_lon:.4f}")
    else:
        user_lat, user_lon = default_lat, default_lon
        st.info("ℹ️ Using default location (Helsinki). Allow GPS for real-time tracking.")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    interest = st.text_input("I want to experience...", "Senate Square & History")
    
    st.write("")
    if st.button("🚀 Generate Route", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing weather ({weather['condition']}) & routes..."):
            tour = get_ai_tour(interest, weather)
            if tour:
                st.session_state['tour'] = tour
                st.session_state['route_map'] = resolve_route_ids(tour['suggested_routes'], tour['mode'])
    st.markdown('</div>', unsafe_allow_html=True)

    if 'tour' in st.session_state:
        tour = st.session_state['tour']
        st.markdown(f"### **{tour['title']}**")
        
        legend_html = ""
        for r in tour['suggested_routes']:
            c = get_route_color(r)
            hex_c = '#%02x%02x%02x' % (c[0], c[1], c[2])
            legend_html += f'<span style="background:{hex_c}; padding: 4px 10px; border-radius: 6px; color: white; margin-right: 6px; font-weight: 700; box-shadow: 0 0 10px {hex_c}80;">Tram {r}</span>'
        st.markdown(f'<div style="margin-bottom: 20px;">{legend_html}</div>', unsafe_allow_html=True)

        for i, step in enumerate(tour['stops']):
            is_last = i == len(tour['stops']) - 1
            line_html = "" if is_last else '<div class="step-line"></div>'
            exit_name = step.get('get_off_stop', 'Destination')
            
            step_html = f"""
<div class="step-card">
    {line_html}
    <div class="step-icon">{i+1}</div>
    <div class="step-content">
        <div class="step-title">{step['name']}</div>
        <div class="step-desc">{step['desc']}</div>
        <div class="step-exit">🛑 Exit: {exit_name}</div>
    </div>
</div>"""
            st.markdown(step_html, unsafe_allow_html=True)

with col_right:
    if 'tour' in st.session_state:
        tour = st.session_state['tour']
        route_list = tour['suggested_routes']
        boarding = get_boarding_info(user_lat, user_lon, route_list)
        crowd_lvl, crowd_icon = get_crowding_status()
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div class="live-badge"></div>
            <h3 style="margin: 0; text-shadow: 0 0 20px rgba(0,255,136,0.3);">Real-Time Intelligence</h3>
        </div>
        """, unsafe_allow_html=True)
        
        live_container = st.empty()
        
        for _ in range(100):
            with live_container.container():
                if boarding:
                    win_c = get_route_color(boarding['route'])
                    win_hex = '#%02x%02x%02x' % (win_c[0], win_c[1], win_c[2])
                    co2_saved = calculate_co2_savings(boarding['distance'])
                    
                    # ✅ 4 COLUMNS FOR METRICS
                    c1, c2, c3, c4 = st.columns(4)
                    
                    with c1:
                        st.markdown(f"""<div class="metric-container"><div class="metric-value">🏃 {boarding['walk_time']} min</div><div class="metric-sub">{boarding['distance']}m Walk</div></div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div class="metric-container"><div class="metric-value" style="color: {win_hex}; text-shadow: 0 0 15px {win_hex}80;">{boarding['route']}</div><div class="metric-sub">Incoming</div></div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""<div class="metric-container"><div class="metric-value">{boarding['minutes_away']} min</div><div class="metric-sub">Wait Time</div></div>""", unsafe_allow_html=True)
                    with c4:
                        # NEW CO2 METRIC
                        st.markdown(f"""<div class="metric-container"><div class="metric-value" style="color: #4CAF50;">🍃 {co2_saved}g</div><div class="metric-sub">CO2 Saved</div></div>""", unsafe_allow_html=True)
                    
                    st.write("")

                # MAP LAYERS
                rmap = st.session_state.get('route_map', {})
                df = get_live_vehicles(rmap)
                layers = []
                
                # 1. Vehicles
                if not df.empty:
                    layers.append(pdk.Layer(
                        "ScatterplotLayer", df,
                        get_position='[lon, lat]', get_fill_color='color',
                        get_line_color=[255,255,255], get_radius=60, opacity=1.0, 
                        pickable=True, filled=True, stroked=True, line_width_min_pixels=2
                    ))

                # 2. User
                layers.append(pdk.Layer("ScatterplotLayer", pd.DataFrame([{"lat": user_lat, "lon": user_lon}]), get_position='[lon, lat]', get_color=[255, 255, 255, 30], get_radius=150))
                layers.append(pdk.Layer("ScatterplotLayer", pd.DataFrame([{"lat": user_lat, "lon": user_lon}]), get_position='[lon, lat]', get_color=[255, 255, 255, 255], get_radius=40, stroked=True))

                # 3. Stop + Line
                if boarding:
                    layers.append(pdk.Layer("ScatterplotLayer", pd.DataFrame([{"lat": boarding['lat'], "lon": boarding['lon']}]), get_position='[lon, lat]', get_color=[0, 255, 136, 200], get_radius=70))
                    line_data = pd.DataFrame([{ "source": [user_lon, user_lat], "target": [boarding['lon'], boarding['lat']] }])
                    layers.append(pdk.Layer("LineLayer", line_data, get_source_position="source", get_target_position="target", get_color=[255, 255, 255], get_width=4, width_min_pixels=2, opacity=0.8))

                view = pdk.ViewState(latitude=user_lat, longitude=user_lon, zoom=14.5, pitch=55, bearing=15)
                
                # DARK OSM STYLE
                deck = pdk.Deck(layers=layers, initial_view_state=view, map_style="dark", tooltip={"text": "{name}\nRoute: {route_name}"})
                st.pydeck_chart(deck)
                
            time.sleep(2)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 100px; opacity: 0.4;">
            <div style="font-size: 5rem;">🚋</div>
            <h3>Waiting for Mission</h3>
            <p>Enter your destination on the left to activate the tracking grid.</p>
        </div>
        """, unsafe_allow_html=True)