import streamlit as st
import streamlit.components.v1 as components
import os
import json
import requests
import pandas as pd
import pydeck as pdk
import time
from datetime import datetime, timedelta
from groq import Groq
from google.transit import gtfs_realtime_pb2
from neo4j import GraphDatabase
from streamlit_js_eval import get_geolocation
from streamlit_searchbox import st_searchbox 

from ai_engine import VectorSearchEngine
from etl_neo4j import run_neo4j_import
from etl_enrich import run_enrichment
from etl_static import load_static_lookups 

st.set_page_config(layout="wide", page_title="Tau AI Navigator")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    .stApp { background: radial-gradient(circle at 10% 20%, rgb(15, 20, 35) 0%, rgb(18, 28, 45) 90%); font-family: 'Inter', sans-serif; color: #E0E0E0; }
    div[data-testid="stVerticalBlock"] > div:has(div.stSearchbox) { z-index: 1000; }
    
    .hero-title { 
        font-size: 3.5rem; 
        font-weight: 800; 
        letter-spacing: -1px;
        background: linear-gradient(90deg, #00C6FF, #0072FF); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 0;
    }
    
    .tech-footer {
        font-size: 0.8rem;
        color: #667;
        text-align: center;
        margin-top: 20px;
        padding-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }

    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 20px; margin-bottom: 15px; }
    .stButton > button { background: linear-gradient(90deg, #00C6FF, #0072FF); color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 12px; font-weight: 600; text-transform: uppercase; width: 100%; }
</style>
""", unsafe_allow_html=True)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123"))
GROQ_KEY = os.getenv("GROQ_API_KEY")
HSL_KEY = os.getenv("DIGITRANSIT_API_KEY")

if 'start_loc' not in st.session_state: st.session_state['start_loc'] = None
if 'end_loc' not in st.session_state: st.session_state['end_loc'] = None
if 'semantic_pois' not in st.session_state: st.session_state['semantic_pois'] = None
if 'route_geometry' not in st.session_state: st.session_state['route_geometry'] = None
if 'use_fallback_line' not in st.session_state: st.session_state['use_fallback_line'] = False

@st.cache_resource
def get_driver():
    try: return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    except: return None
driver = get_driver()

@st.cache_resource
def get_static_data():
    return load_static_lookups()
routes_dict, trip_lookup, direction_lookup = get_static_data()

@st.cache_resource
def get_semantic_engine():
    return VectorSearchEngine()
ai_engine = get_semantic_engine()

if ai_engine.cached_embeddings is None and driver:
    with driver.session() as session:
        query = "MATCH (p:PointOfInterest) RETURN p.name as name, p.description as description, p.lat as lat, p.lon as lon"
        result = session.run(query)
        pois = [r.data() for r in result]
        if pois: ai_engine.fit_index(pois, text_key='description')

def search_hsl_places(searchterm: str):
    if not searchterm: return []
    url = "https://api.digitransit.fi/geocoding/v1/search"
    params = {"text": searchterm, "size": 5, "digitransit-subscription-key": HSL_KEY}
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            return [(f['properties']['label'], json.dumps({"name": f['properties']['label'], "lat": f['geometry']['coordinates'][1], "lon": f['geometry']['coordinates'][0]})) for f in resp.json()['features']]
    except: return []
    return []

def decode_polyline(polyline_str):
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}
    while index < len(polyline_str):
        for unit in ['latitude', 'longitude']:
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20: break
            if (result & 1): changes[unit] = ~(result >> 1)
            else: changes[unit] = (result >> 1)
        lat += changes['latitude']
        lng += changes['longitude']
        coordinates.append([lng / 100000.0, lat / 100000.0])
    return coordinates

def get_hsl_route(start, end):
    if not start or not end: return None
    url = "https://api.digitransit.fi/routing/v1/routers/hsl/index/graphql"
    
    query = """
    { plan(from: {lat: %f, lon: %f}, to: {lat: %f, lon: %f}, numItineraries: 1) {
        itineraries { legs { mode legGeometry { points } } }
    } }
    """ % (start['lat'], start['lon'], end['lat'], end['lon'])
    
    headers = {"Content-Type": "application/json", "digitransit-subscription-key": HSL_KEY}
    try:
        resp = requests.post(url, json={"query": query}, headers=headers)
        data = resp.json()
        path_segments = []
        
        if 'data' in data and data['data']['plan']['itineraries']:
            legs = data['data']['plan']['itineraries'][0]['legs']
            for leg in legs:
                points = decode_polyline(leg['legGeometry']['points'])
                
                color = [0, 180, 255]
                if leg['mode'] == 'TRAM': color = [50, 255, 100]
                elif leg['mode'] == 'SUBWAY': color = [255, 140, 0]
                elif leg['mode'] == 'RAIL': color = [255, 50, 50]
                elif leg['mode'] == 'FERRY': color = [0, 200, 255]
                elif leg['mode'] == 'WALK': color = [200, 200, 200]
                
                path_segments.append({"path": points, "color": color})
            return path_segments
        else:
            return None
    except: 
        return None

def get_planned_itinerary(start, end, departure_time=None):
    if not start or not end: return "Error: Missing location data."
    time_mode = f'dateTime: "{departure_time.strftime("%Y-%m-%dT%H:%M:%S")}+02:00"' if departure_time else ""
    query = """
    { plan(from: {lat: %f, lon: %f}, to: {lat: %f, lon: %f}, numItineraries: 2, %s) {
        itineraries { duration legs { mode startTime route { shortName } from { name } to { name } } }
    } }
    """ % (float(start['lat']), float(start['lon']), float(end['lat']), float(end['lon']), time_mode)
    
    try:
        resp = requests.post("https://api.digitransit.fi/routing/v1/routers/hsl/index/graphql", json={"query": query}, headers={"Content-Type": "application/json", "digitransit-subscription-key": HSL_KEY})
        itins = resp.json()['data']['plan']['itineraries']
        context_str = "OFFICIAL HSL SCHEDULE:\n"
        for i, itin in enumerate(itins):
            context_str += f"Option {i+1} ({int(itin['duration']/60)} min):\n"
            for leg in itin['legs']:
                start_t = datetime.fromtimestamp(leg['startTime']/1000).strftime('%H:%M')
                route = leg['route']['shortName'] if leg['route'] else ""
                context_str += f" - {start_t}: {leg['mode']} {route} from {leg['from']['name']}\n"
        return context_str
    except Exception as e: return f"Planner Error: {str(e)}"

def ask_general_llm(query):
    if not GROQ_KEY: return "AI service offline."
    client = Groq(api_key=GROQ_KEY)
    prompt = f"""
    You are a Helsinki Transport Expert.
    User Question: "{query}"
    Info:
    - Single Ticket (Zone AB): 2.95 Euro
    - Day Ticket: 9.00 Euro
    - Fine: 80 Euro
    Instructions: Be brief, professional, and helpful. No emojis.
    """
    try:
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    except: return "Service unavailable."

def ask_llm(query, start, end, semantic_pois=None, planned_time=None):
    if not GROQ_KEY: return "AI Offline."
    client = Groq(api_key=GROQ_KEY)
    target_name = end['name']
    dest_desc = "Point of Interest"
    if semantic_pois is not None and not semantic_pois.empty:
        target_name = semantic_pois.iloc[0]['name']
        dest_desc = semantic_pois.iloc[0].get('description', '')
        end = {'lat': semantic_pois.iloc[0]['lat'], 'lon': semantic_pois.iloc[0]['lon'], 'name': target_name}

    planner_data = get_planned_itinerary(start, end, planned_time)
    prompt = f"""
    Role: Professional Helsinki Transport Guide.
    Task: Create a structured itinerary from {start['name']} to {target_name}.
    Vibe: {query} ({dest_desc}).
    
    Data: 
    {planner_data}
    
    Output strictly:
    - Departure: [Time] [Location]
    - Route: [Mode] [Line]
    - Arrival: [Time]
    - Tip: Enjoy {dest_desc}.
    """
    try:
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    except: return "AI Error."

def get_live_vehicles():
    try:
        resp = requests.get("https://realtime.hsl.fi/realtime/vehicle-positions/v2/hsl", headers={"digitransit-subscription-key": HSL_KEY}, timeout=2)
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)
        vehicles = []
        for e in feed.entity:
            if e.HasField('vehicle') and e.vehicle.position:
                r_id = e.vehicle.trip.route_id.replace("HSL:", "").strip() if e.vehicle.trip.route_id else ""
                t_id = e.vehicle.trip.trip_id.replace("HSL:", "").strip()
                d_id = str(e.vehicle.trip.direction_id)
                route_data = routes_dict.get(r_id, {"short": r_id, "mode": "BUS", "long": ""})
                
                headsign = trip_lookup.get(t_id)
                if not headsign:
                    headsign = direction_lookup.get((r_id, d_id))
                if not headsign:
                    headsign = "City Centre" if d_id == '1' else "Regional Terminus"
                
                mode = route_data['mode']
                short = route_data['short']
                tooltip = f"<b>{mode} {short}</b><br/>To: {headsign}"
                
                if mode == 'TRAM': color, radius = [0, 200, 100, 200], 40
                elif mode == 'METRO': color, radius = [255, 140, 0, 200], 50
                elif mode == 'TRAIN': color, radius = [200, 0, 0, 200], 50
                elif mode == 'FERRY': color, radius = [0, 100, 255, 200], 60
                else: color, radius = [0, 150, 255, 180], 30
                
                vehicles.append({"lat": e.vehicle.position.latitude, "lon": e.vehicle.position.longitude, "color": color, "radius": radius, "html_tooltip": tooltip})
        return pd.DataFrame(vehicles)
    except: return pd.DataFrame()

def get_graph_pois():
    if not driver: return pd.DataFrame()
    with driver.session() as session:
        result = session.run("MATCH (p:PointOfInterest) RETURN p.name as name, p.lat as lat, p.lon as lon, p.description as desc")
        df = pd.DataFrame([r.data() for r in result])
        if not df.empty:
            df['html_tooltip'] = "<b>" + df['name'] + "</b><br/>" + df['desc'].fillna('Point of Interest')
            df['color'] = [[255, 0, 128, 200]] * len(df)
            df['radius'] = 30
        return df

col_logo, col_title = st.columns([1, 5])
with col_logo: st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Helsinki_vaakuna.svg/1200px-Helsinki_vaakuna.svg.png", width=70)
with col_title: st.markdown('<div class="hero-title">TAU AI NAVIGATOR</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 2.5], gap="medium")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Plan Your Journey")
    if st.toggle("Use GPS Location"):
        loc = get_geolocation()
        if loc and 'coords' in loc:
            st.session_state['start_loc'] = {"name": "My Location", "lat": loc['coords']['latitude'], "lon": loc['coords']['longitude']}
            st.success("GPS Locked")

    start_json = st_searchbox(search_hsl_places, key="s1", placeholder="From...", label="Start")
    if start_json: st.session_state['start_loc'] = json.loads(start_json)

    end_json = st_searchbox(search_hsl_places, key="s2", placeholder="To...", label="Destination")
    if end_json: st.session_state['end_loc'] = json.loads(end_json)

    with st.expander("Preferred Time", expanded=False):
        d = st.date_input("Date", datetime.now())
        t = st.time_input("Time", datetime.now())
        planned_dt = datetime.combine(d, t)

    st.markdown('</div><div class="glass-card">', unsafe_allow_html=True)
    interest = st.text_input("Vibe Search", placeholder="e.g. Quiet Library")
    
    if st.button("Find Route"):
        found_pois = None
        if interest:
            results = ai_engine.search(interest, top_k=5)
            if results:
                found_pois = pd.DataFrame(results)
                found_pois['html_tooltip'] = "<b>" + found_pois['name'] + "</b><br/>Match: " + interest
                found_pois['color'] = [[255, 200, 0, 255]] * len(found_pois)
                found_pois['radius'] = 50
                st.session_state['semantic_pois'] = found_pois
                st.toast(f"Found {len(results)} vibes")
        
        if st.session_state['start_loc'] and st.session_state['end_loc']:
            with st.spinner("Calculating Path..."):
                route_geo = get_hsl_route(st.session_state['start_loc'], st.session_state['end_loc'])
                if route_geo:
                    st.session_state['route_geometry'] = route_geo
                    st.session_state['use_fallback_line'] = False
                else:
                    st.session_state['use_fallback_line'] = True
                    st.session_state['route_geometry'] = None

        if st.session_state['start_loc']:
            with st.spinner("Processing..."):
                plan = ask_llm(interest, st.session_state['start_loc'], st.session_state['end_loc'], found_pois, planned_dt)
                st.info(plan)

    st.markdown("---")
    st.markdown("### Ask the TAU AI Navigator")
    general_q = st.text_input("Ask about fares, rules, etc.", placeholder="How much is a ticket?")
    if general_q:
        ans = ask_general_llm(general_q)
        st.success(ans)

    if st.button("Reload Data"):
        if driver:
            with st.spinner("Reloading..."):
                run_neo4j_import(driver, HSL_KEY)
                run_enrichment(driver)
                st.success("Updated")
                
    st.markdown("""
    <div class='tech-footer'>
        POWERED BY NEURO-SYMBOLIC AI & KNOWLEDGE GRAPHS<br>
        Built for the Public Transport Graph Challenge
    </div>
    """, unsafe_allow_html=True)

with col_right:
    map_placeholder = st.empty()
    while True:
        layers = []
        
        v_df = get_live_vehicles()
        if not v_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", v_df,
                get_position='[lon, lat]', get_fill_color='color', get_radius='radius',
                pickable=True, auto_highlight=True, tooltip="html_tooltip"
            ))

        if st.session_state['semantic_pois'] is not None:
            p_df = st.session_state['semantic_pois']
        else:
            p_df = get_graph_pois()
            
        if not p_df.empty and 'html_tooltip' in p_df.columns:
            layers.append(pdk.Layer(
                "ScatterplotLayer", p_df,
                get_position='[lon, lat]', get_fill_color='color', get_radius='radius',
                pickable=True, stroked=True, get_line_color=[255,255,255]
            ))

        if st.session_state['route_geometry']:
            layers.append(pdk.Layer(
                "PathLayer", data=st.session_state['route_geometry'],
                get_path="path", get_color="color", width_scale=20, width_min_pixels=5, pickable=True
            ))
            
        if st.session_state.get('use_fallback_line') and st.session_state['start_loc'] and st.session_state['end_loc']:
            arc_data = [{
                "source": [st.session_state['start_loc']['lon'], st.session_state['start_loc']['lat']],
                "target": [st.session_state['end_loc']['lon'], st.session_state['end_loc']['lat']]
            }]
            layers.append(pdk.Layer(
                "ArcLayer", data=arc_data,
                get_source_position="source", get_target_position="target",
                get_source_color=[0, 255, 0], get_target_color=[255, 0, 0],
                get_width=5
            ))

        if st.session_state['start_loc']:
             layers.append(pdk.Layer("ScatterplotLayer", data=[st.session_state['start_loc']], get_position='[lon, lat]', get_fill_color=[0, 100, 255, 255], get_radius=100, stroked=True, get_line_color=[255,255,255], get_line_width=5))
        if st.session_state['end_loc']:
             layers.append(pdk.Layer("ScatterplotLayer", data=[st.session_state['end_loc']], get_position='[lon, lat]', get_fill_color=[255, 50, 50, 255], get_radius=100, stroked=True, get_line_color=[255,255,255], get_line_width=5))

        deck = pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(latitude=60.17, longitude=24.94, zoom=13),
            layers=layers,
            tooltip={"html": "{html_tooltip}", "style": {"backgroundColor": "#2c3e50", "color": "white"}}
        )
        map_placeholder.pydeck_chart(deck)
        time.sleep(2)