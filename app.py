import streamlit as st
import streamlit.components.v1 as components
import os
import json
import requests
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta
from groq import Groq
from google.transit import gtfs_realtime_pb2
from neo4j import GraphDatabase
from streamlit_js_eval import get_geolocation
from streamlit_searchbox import st_searchbox  # <--- YOUR NEW COMPONENT

# --- CUSTOM MODULES ---
from ai_engine import VectorSearchEngine
from etl_neo4j import run_neo4j_import
from etl_enrich import run_enrichment
from etl_static import load_static_lookups 

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Helsinki AI Navigator", page_icon="🚋")

# --- MERGED CSS (Mate's Style + Your Z-Index Fix) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* GLOBAL THEME */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 20, 35) 0%, rgb(18, 28, 45) 90%);
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }

    /* FIX Z-INDEX FOR SEARCHBOX (Crucial for your feature) */
    div[data-testid="stVerticalBlock"] > div:has(div.stSearchbox) { z-index: 1000; }
    
    /* REMOVE STREAMLIT PADDING/FOOTER */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* HERO TITLE */
    .hero-title { 
        font-size: 3.5rem; 
        font-weight: 800; 
        letter-spacing: -1px;
        background: linear-gradient(90deg, #00C6FF, #0072FF); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 0;
    }

    /* GLASS CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(90deg, #00C6FF, #0072FF);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        text-transform: uppercase;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# API Keys
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123"))
GROQ_KEY = os.getenv("GROQ_API_KEY")
HSL_KEY = os.getenv("DIGITRANSIT_API_KEY")

# --- 2. INITIALIZATION & CACHING ---

if 'start_loc' not in st.session_state: st.session_state['start_loc'] = None
if 'end_loc' not in st.session_state: st.session_state['end_loc'] = None
if 'semantic_pois' not in st.session_state: st.session_state['semantic_pois'] = None
if 'route_geometry' not in st.session_state: st.session_state['route_geometry'] = None # Mate's feature

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

# Initialize Vector Index (Your Logic)
if ai_engine.cached_embeddings is None:
    if driver:
        with driver.session() as session:
            query = """
            MATCH (p:PointOfInterest) 
            RETURN p.name as name, p.description as description, 
                   p.lat as lat, p.lon as lon
            """
            result = session.run(query)
            pois = [r.data() for r in result]
            if pois: ai_engine.fit_index(pois, text_key='description')

# --- 3. HELPER FUNCTIONS ---

def search_hsl_places(searchterm: str):
    """GOOGLE-STYLE AUTOCOMPLETE (Your Logic)"""
    if not searchterm: return []
    url = "https://api.digitransit.fi/geocoding/v1/search"
    params = {"text": searchterm, "size": 5, "digitransit-subscription-key": HSL_KEY}
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            features = resp.json().get('features', [])
            suggestions = []
            for f in features:
                props = f['properties']
                label = props.get('label', 'Unknown')
                coords = f['geometry']['coordinates']
                value = json.dumps({"name": label, "lat": coords[1], "lon": coords[0]})
                suggestions.append((label, value))
            return suggestions
    except: return []
    return []

def decode_polyline(polyline_str):
    """Decodes Google Polyline (Mate's Logic)"""
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
    """Fetches exact geometry (Mate's Logic)"""
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
        legs = data['data']['plan']['itineraries'][0]['legs']
        for leg in legs:
            points = decode_polyline(leg['legGeometry']['points'])
            color = [0, 255, 100] if leg['mode'] == 'TRAM' else [0, 150, 255]
            path_segments.append({"path": points, "color": color})
        return path_segments
    except: return None

def get_planned_itinerary(start, end, departure_time=None):
    """Time-Based Planning (Your Logic)"""
    if not start or not end: return ""
    
    if departure_time:
        time_str = departure_time.strftime("%Y-%m-%dT%H:%M:%S") + "+02:00"
        time_mode = f'dateTime: "{time_str}"'
    else:
        time_mode = "" 

    query = """
    { plan(from: {lat: %s, lon: %s}, to: {lat: %s, lon: %s}, numItineraries: 3, %s) {
        itineraries { duration legs { mode startTime route { shortName } from { name } to { name } } }
    } }
    """ % (start['lat'], start['lon'], end['lat'], end['lon'], time_mode)

    url = "https://api.digitransit.fi/routing/v1/routers/hsl/index/graphql"
    headers = {"Content-Type": "application/json", "digitransit-subscription-key": HSL_KEY}
    try:
        resp = requests.post(url, json={"query": query}, headers=headers)
        data = resp.json()['data']['plan']['itineraries']
        context_str = "OFFICIAL HSL PLANNER RESULTS:\n"
        for i, itin in enumerate(data):
            duration_min = int(itin['duration'] / 60)
            context_str += f"OPTION {i+1} ({duration_min} min):\n"
            for leg in itin['legs']:
                start_t = datetime.fromtimestamp(leg['startTime']/1000).strftime('%H:%M')
                mode = leg['mode']
                route = leg['route']['shortName'] if leg['route'] else ""
                context_str += f"  - At {start_t}, take {mode} {route} from {leg['from']['name']}\n"
        return context_str
    except Exception as e: return f"Planner Error: {str(e)}"

def ask_llm(query, start, end, semantic_pois=None, planned_time=None):
    """Merged LLM Logic: Your Context + Mate's Witty Persona"""
    if not GROQ_KEY: return "AI Offline."
    client = Groq(api_key=GROQ_KEY)
    
    start_name = start['name'] if start else "Helsinki Center"
    target_name = end['name'] if end else "Destination"
    
    if semantic_pois is not None and not semantic_pois.empty:
        target_name = semantic_pois.iloc[0]['name']
        end = {'lat': semantic_pois.iloc[0]['lat'], 'lon': semantic_pois.iloc[0]['lon'], 'name': target_name}

    planner_context = get_planned_itinerary(start, end, planned_time)
    
    prompt = f"""
    Act as a witty Helsinki Local Guide.
    Request: Go from {start_name} to {target_name}.
    Vibe: "{query}"
    Time: {planned_time if planned_time else "NOW"}
    
    DATA:
    {planner_context}
    
    INSTRUCTIONS:
    1. Summarize the best route option.
    2. Be conversational and helpful.
    3. If a vibe was provided, mention why the destination fits.
    """
    try:
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    except: return "AI Error."

def speak_text(text):
    """Text to Speech (Mate's Logic)"""
    safe_text = text.replace('"', '\\"').replace('\n', ' ')
    js = f"""<script>
        var msg = new SpeechSynthesisUtterance("{safe_text}");
        msg.lang = 'en-US';
        window.speechSynthesis.speak(msg);
    </script>"""
    components.html(js, height=0, width=0)

def get_live_vehicles():
    """FIXED VERSION (Your Logic): Shows Descriptions correctly"""
    try:
        url = "https://realtime.hsl.fi/realtime/vehicle-positions/v2/hsl"
        headers = {"digitransit-subscription-key": HSL_KEY} if HSL_KEY else {}
        resp = requests.get(url, headers=headers, timeout=2)
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
                if not headsign: headsign = direction_lookup.get((r_id, d_id), "Unknown")
                
                mode = route_data['mode']
                short = route_data['short']
                desc = f"{mode} {short}: To {headsign}"
                details = f"Route: {route_data['long']}"
                
                if mode == 'TRAM': color, radius = [0, 200, 100, 200], 40
                elif mode == 'METRO': color, radius = [255, 140, 0, 200], 50
                else: color, radius = [0, 150, 255, 180], 30

                vehicles.append({
                    "lat": e.vehicle.position.latitude, "lon": e.vehicle.position.longitude,
                    "color": color, "radius": radius, "desc": desc, "details": details
                })
        return pd.DataFrame(vehicles)
    except: return pd.DataFrame()

def get_graph_pois():
    if not driver: return pd.DataFrame()
    with driver.session() as session:
        result = session.run("MATCH (p:PointOfInterest) RETURN p.name as name, p.lat as lat, p.lon as lon")
        return pd.DataFrame([r.data() for r in result])

# --- 4. UI LAYOUT ---

col_logo, col_title = st.columns([1, 5])
with col_logo: st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Helsinki_vaakuna.svg/1200px-Helsinki_vaakuna.svg.png", width=70)
with col_title: st.markdown('<div class="hero-title">HELSINKI AI NAVIGATOR</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 2.5], gap="medium")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Plan Your Journey")
    
    # GPS Toggle (Mate's position logic)
    if st.toggle("📍 Use GPS Location"):
        loc = get_geolocation()
        if loc and 'coords' in loc:
            st.session_state['start_loc'] = {"name": "My Location", "lat": loc['coords']['latitude'], "lon": loc['coords']['longitude']}
            st.success("GPS Locked")

    # --- YOUR SEARCHBOXES ---
    st.markdown("**From:**")
    selected_start = st_searchbox(search_hsl_places, key="s1", placeholder="Type start...")
    if selected_start: st.session_state['start_loc'] = json.loads(selected_start)

    st.markdown("**To:**")
    selected_end = st_searchbox(search_hsl_places, key="s2", placeholder="Type destination...")
    if selected_end: st.session_state['end_loc'] = json.loads(selected_end)

    # --- YOUR TIME PICKER ---
    with st.expander("🕒 Preferred Time (Optional)", expanded=False):
        d = st.date_input("Date", datetime.now())
        t = st.time_input("Time", datetime.now())
        planned_dt = datetime.combine(d, t)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    interest = st.text_input("✨ Vibe Search", placeholder="e.g. Underground Techno")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Find Route"):
            # 1. Semantic Search (Your Logic)
            found_pois = None
            if interest:
                results = ai_engine.search(interest, top_k=10)
                if results:
                    found_pois = pd.DataFrame(results)
                    st.session_state['semantic_pois'] = found_pois
                    st.toast(f"Found {len(results)} vibes", icon="🎯")
            
            # 2. Path Geometry (Mate's Logic)
            if st.session_state['start_loc'] and st.session_state['end_loc']:
                st.session_state['route_geometry'] = get_hsl_route(st.session_state['start_loc'], st.session_state['end_loc'])

            # 3. LLM Plan (Merged Logic)
            if st.session_state['start_loc']:
                with st.spinner("AI Planning..."):
                    plan = ask_llm(interest, st.session_state['start_loc'], st.session_state['end_loc'], found_pois, planned_dt)
                    st.info(plan)
                    speak_text(f"Route found. {plan[:100]}") # Audio Feedback

    with col_btn2:
        if st.button("🔄 Reload Data"):
            if driver:
                with st.spinner("Refreshing..."):
                    run_neo4j_import(driver, HSL_KEY)
                    run_enrichment(driver)
                    st.success("Updated")
    
    live_mode = st.toggle("📡 Live Tracking Mode", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- RIGHT COLUMN: MAP ---
with col_right:
    
    @st.fragment(run_every=2 if live_mode else None)
    def render_map():
        # Determine POI Data (Heatmap vs Scatter)
        if st.session_state['semantic_pois'] is not None:
            pois_df = st.session_state['semantic_pois']
            is_heatmap = True
        else:
            pois_df = get_graph_pois()
            is_heatmap = False # Default generic dots are not heatmap
        
        vehicles_df = get_live_vehicles()
        
        view_lat, view_lon, zoom = 60.1699, 24.9384, 13
        if st.session_state['start_loc']:
            view_lat = st.session_state['start_loc']['lat']
            view_lon = st.session_state['start_loc']['lon']
            zoom = 14

        layers = []

        # Layer A: Live Vehicles (Your fixed tooltips)
        if not vehicles_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", vehicles_df,
                get_position='[lon, lat]', get_fill_color='color', get_radius='radius',
                pickable=True, auto_highlight=True
            ))

        # Layer B: POIs (Mate's Heatmap Logic)
        if not pois_df.empty:
            if is_heatmap:
                layers.append(pdk.Layer(
                    "HeatmapLayer", pois_df, get_position='[lon, lat]', opacity=0.6,
                    radiusPixels=50, intensity=1, threshold=0.3
                ))
            else:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", pois_df, get_position='[lon, lat]',
                    get_fill_color=[255, 0, 128, 200], get_radius=30, pickable=True
                ))

        # Layer C: Route Path (Mate's Logic)
        if st.session_state['route_geometry']:
            layers.append(pdk.Layer(
                "PathLayer", data=st.session_state['route_geometry'],
                get_path="path", get_color="color", width_scale=20, width_min_pixels=3
            ))

        # Layer D: Markers
        if st.session_state['start_loc']:
             layers.append(pdk.Layer("ScatterplotLayer", data=[st.session_state['start_loc']], get_position='[lon, lat]', get_fill_color=[0, 100, 255, 255], get_radius=150, stroked=True, get_line_color=[255,255,255], get_line_width=5))
        if st.session_state['end_loc']:
             layers.append(pdk.Layer("ScatterplotLayer", data=[st.session_state['end_loc']], get_position='[lon, lat]', get_fill_color=[255, 50, 50, 255], get_radius=150, stroked=True, get_line_color=[255,255,255], get_line_width=5))

        deck = pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=zoom, pitch=0),
            layers=layers,
            tooltip={"html": "<b>{desc}</b><br/>{details}" if not vehicles_df.empty else "<b>{name}</b>", "style": {"backgroundColor": "steelblue", "color": "white"}}
        )
        st.pydeck_chart(deck)

    render_map()