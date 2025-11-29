import streamlit as st
import streamlit.components.v1 as components
import os
import requests
import pandas as pd
import pydeck as pdk
from groq import Groq
from google.transit import gtfs_realtime_pb2
from neo4j import GraphDatabase
from streamlit_js_eval import get_geolocation

# --- NEW: Semantic Engine Import ---
from ai_engine import VectorSearchEngine

# Import ETL scripts
from etl_neo4j import run_neo4j_import
from etl_enrich import run_enrichment

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Helsinki AI Navigator", page_icon="🚋")

# IMPORT GOOGLE FONTS & INJECT CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* GLOBAL THEME */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 20, 35) 0%, rgb(18, 28, 45) 90%);
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }

    /* REMOVE STREAMLIT PADDING/FOOTER */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* HERO TITLE STYLING */
    .hero-container {
        text-align: left;
        padding-bottom: 20px;
    }
    .hero-title { 
        font-size: 3.5rem; 
        font-weight: 800; 
        letter-spacing: -1px;
        background: linear-gradient(90deg, #00C6FF, #0072FF); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #A0A0A0;
        font-weight: 300;
        margin-top: -10px;
    }

    /* GLASSMORPHISM SIDEBAR / CARDS */
    section[data-testid="stSidebar"] {
        background-color: rgba(20, 25, 40, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    /* INPUT FIELDS - MODERN & DARK */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #FFF;
        border-radius: 10px;
        padding: 10px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00C6FF;
        box-shadow: 0 0 10px rgba(0, 198, 255, 0.3);
    }

    /* BUTTONS - NEON GLOW */
    .stButton > button {
        background: linear-gradient(90deg, #00C6FF, #0072FF);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 198, 255, 0.4);
    }
    
    /* STATUS BOXES */
    .success-box { 
        background: rgba(0, 255, 150, 0.15); 
        border-left: 4px solid #00ff96; 
        padding: 12px; 
        border-radius: 8px; 
        color: #ccffeb; 
        margin-top: 10px;
        font-size: 0.9rem;
    }
    .error-box { 
        background: rgba(255, 80, 80, 0.15); 
        border-left: 4px solid #ff5050; 
        padding: 12px; 
        border-radius: 8px; 
        color: #ffcccc; 
        margin-top: 10px;
        font-size: 0.9rem;
    }

    /* TOGGLE SWITCH */
    .stToggle { margin-top: 10px; }
    
</style>
""", unsafe_allow_html=True)

# Database & API Setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123"))
GROQ_KEY = os.getenv("GROQ_API_KEY")
HSL_KEY = os.getenv("DIGITRANSIT_API_KEY")

# Initialize Session State
if 'start_loc' not in st.session_state: st.session_state['start_loc'] = None
if 'end_loc' not in st.session_state: st.session_state['end_loc'] = None
if 'semantic_pois' not in st.session_state: st.session_state['semantic_pois'] = None
if 'route_geometry' not in st.session_state: st.session_state['route_geometry'] = None # Stores the path

@st.cache_resource
def get_driver():
    try: return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    except: return None
driver = get_driver()

# --- 2. DATA ENGINES (AI & GRAPH) ---

@st.cache_resource
def get_semantic_engine():
    return VectorSearchEngine()

ai_engine = get_semantic_engine()

def initialize_search_index():
    """Syncs Neo4j data with the Vector Engine"""
    if driver:
        with driver.session() as session:
            query = """
            MATCH (p:PointOfInterest)
            RETURN p.name as name, p.lat as lat, p.lon as lon, 
                   p.description as description, p.raw_type as type
            """
            result = session.run(query)
            pois = [r.data() for r in result]
            
            if pois:
                ai_engine.fit_index(pois, text_key='description')
                return len(pois)
    return 0

if ai_engine.cached_embeddings is None:
    initialize_search_index()

def get_pois_semantically(query):
    if not query: return pd.DataFrame()
    results = ai_engine.search(query, top_k=20)
    return pd.DataFrame(results) if results else pd.DataFrame()

def geocode(text):
    if not text: return None
    url = "https://api.digitransit.fi/geocoding/v1/search"
    params = {"text": text, "size": 1, "digitransit-subscription-key": HSL_KEY}
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if data['features']:
                props = data['features'][0]['properties']
                coords = data['features'][0]['geometry']['coordinates']
                return {"name": props['label'], "lat": coords[1], "lon": coords[0]}
    except: return None
    return None

# --- ROUTE GEOMETRY FETCHER ---

def decode_polyline(polyline_str):
    """Decodes Google Polyline format into [lon, lat] list"""
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
                if not byte >= 0x20:
                    break
            if (result & 1):
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)
        lat += changes['latitude']
        lng += changes['longitude']
        coordinates.append([lng / 100000.0, lat / 100000.0])
    return coordinates

def get_hsl_route(start, end):
    """Fetches exact geometry"""
    if not start or not end: return None
    
    url = "https://api.digitransit.fi/routing/v1/routers/hsl/index/graphql"
    query = """
    {
      plan(
        from: {lat: %f, lon: %f}
        to: {lat: %f, lon: %f}
        numItineraries: 1
      ) {
        itineraries {
          legs {
            mode
            legGeometry { points }
          }
        }
      }
    }
    """ % (start['lat'], start['lon'], end['lat'], end['lon'])
    
    headers = {"Content-Type": "application/json", "digitransit-subscription-key": HSL_KEY}
    
    try:
        resp = requests.post(url, json={"query": query}, headers=headers)
        data = resp.json()
        
        path_segments = []
        
        legs = data['data']['plan']['itineraries'][0]['legs']
        
        for leg in legs:
            points = decode_polyline(leg['legGeometry']['points'])
            path_segments.append({
                "path": points,
                "color": [0, 255, 100] if leg['mode'] == 'TRAM' else [0, 150, 255] 
            })
            
        return path_segments
        
    except Exception as e:
        print(f"Routing Error: {e}")
        return None

def get_live_vehicles():
    try:
        url = "https://realtime.hsl.fi/realtime/vehicle-positions/v2/hsl"
        headers = {"digitransit-subscription-key": HSL_KEY} if HSL_KEY else {}
        resp = requests.get(url, headers=headers, timeout=2)
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)
        
        vehicles = []
        for e in feed.entity:
            if e.HasField('vehicle') and e.vehicle.position:
                raw_id = e.vehicle.trip.route_id or ""
                short_id = raw_id.replace("HSL:", "").strip()
                color = [0, 150, 255, 180] 
                radius = 30
                if len(short_id) <= 2 or short_id in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "15"]:
                    color = [0, 255, 100, 200]
                    radius = 45
                if "M1" in short_id or "M2" in short_id:
                    color = [255, 140, 0, 200]
                    radius = 60
                vehicles.append({
                    "lat": e.vehicle.position.latitude,
                    "lon": e.vehicle.position.longitude,
                    "color": color,
                    "radius": radius,
                    "id": short_id
                })
        return pd.DataFrame(vehicles)
    except: return pd.DataFrame()


def get_graph_pois():
    if not driver: return pd.DataFrame()
    try:
        with driver.session() as session:
            result = session.run("MATCH (p:PointOfInterest) RETURN p.name as name, p.lat as lat, p.lon as lon")
            return pd.DataFrame([r.data() for r in result])
    except: return pd.DataFrame()

def ask_llm(query, start, end):
    if not GROQ_KEY: return "AI Offline."
    client = Groq(api_key=GROQ_KEY)
    start_name = start['name'] if start else "Helsinki Center"
    end_name = end['name'] if end else "Anywhere"
    prompt = f"""
    Act as a witty Helsinki local. 
    Plan a route from {start_name} to {end_name}.
    User Interest: "{query}"
    Provide a short, structured itinerary (3 bullet points).
    Add a 'Pro Tip' at the end.
    """
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except: return "AI Connection Failed."

def speak_text(text):
    safe_text = text.replace('"', '\\"').replace('\n', ' ')
    js = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{safe_text}");
        msg.lang = 'en-US';
        window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js, height=0, width=0)

# --- 3. UI LAYOUT ---

col_logo, col_title = st.columns([1, 5])
with col_logo: st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Helsinki_vaakuna.svg/1200px-Helsinki_vaakuna.svg.png", width=70)
with col_title: st.markdown('<div class="hero-title">HELSINKI AI NAVIGATOR</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 2.5], gap="medium")

# --- LEFT COLUMN: CONTROLS ---
with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Plan Your Journey")
    
    if st.toggle("📍 Use GPS Location"):
        loc = get_geolocation()
        if loc and 'coords' in loc:
            st.session_state['start_loc'] = {"name": "My Location", "lat": loc['coords']['latitude'], "lon": loc['coords']['longitude']}
            st.success(f"GPS Locked")
        else:
            st.warning("Requesting GPS...")

    start_val = st.session_state['start_loc']['name'] if st.session_state['start_loc'] else ""
    start_query = st.text_input("From (Start)", value=start_val, placeholder="e.g. Hanken")
    if start_query and start_query != "My Location":
        res = geocode(start_query)
        if res: st.session_state['start_loc'] = res

    end_query = st.text_input("To (Destination)", placeholder="e.g. Sibelius Monument")
    if end_query:
        res = geocode(end_query)
        if res: st.session_state['end_loc'] = res

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎨 Vibe Search")
    interest = st.text_input("What are you looking for?", placeholder="e.g. Underground Techno, Vegan Sushi")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Find Route"):
            if interest:
                found_pois = get_pois_semantically(interest)
                st.session_state['semantic_pois'] = found_pois if not found_pois.empty else None

            if st.session_state['start_loc'] and st.session_state['end_loc']:
                path_data = get_hsl_route(st.session_state['start_loc'], st.session_state['end_loc'])
                st.session_state['route_geometry'] = path_data

            with st.spinner("AI Planning..."):
                plan = ask_llm(interest, st.session_state['start_loc'], st.session_state['end_loc'])
                st.info(plan)
                summary = plan.split('.')[0]
                speak_text(f"Here is a plan for {interest}. {summary}")

    with col_btn2:
        if st.button("🔊 Audio Guide"):
            if interest:
                speak_text(f"Exploring {interest} in Helsinki is a great choice. Follow the green line on the map.")
            else:
                speak_text("Welcome to Helsinki. Please select a destination.")

    st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)
    
    live_mode = st.toggle("📡 Live Tracking Mode", value=True)

    if st.button("🔄 Reload Graph"):
        if driver:
            with st.spinner("Rebuilding Knowledge Graph..."):
                run_neo4j_import(driver, HSL_KEY)
                run_enrichment(driver)
                initialize_search_index()
                st.toast("Knowledge Graph Updated!", icon="🧠")

# --- RIGHT COLUMN: MAP ---
with col_right:
    
    @st.fragment(run_every=2 if live_mode else None)
    def render_map():
        if st.session_state['semantic_pois'] is not None and not st.session_state['semantic_pois'].empty:
            pois_df = st.session_state['semantic_pois']
            is_heatmap = True
        else:
            pois_df = get_graph_pois()
            is_heatmap = False
        
        vehicles_df = get_live_vehicles()
        
        # Adjust view for 2D Heatmap (Tilt=0)
        view_lat, view_lon, zoom = 60.1699, 24.9384, 13
        pitch = 0 # Flat view for 2D Heatmap
        
        if st.session_state['start_loc']:
            view_lat = st.session_state['start_loc']['lat']
            view_lon = st.session_state['start_loc']['lon']
            zoom = 14
            
        layers = []

        # LAYER A: Live Vehicles
        if not vehicles_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", vehicles_df,
                get_position='[lon, lat]', get_fill_color='color',
                get_radius='radius', pickable=True, auto_highlight=True
            ))

        # LAYER B: POIs (2D Heatmap)
        if is_heatmap:
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=pois_df,
                get_position='[lon, lat]',
                opacity=0.6,
                radiusPixels=50,
                intensity=1,
                threshold=0.3,
                # Gradient from Blue (low) to Purple -> Red (High)
                colorRange=[
                    [255, 255, 204],
                    [161, 218, 180],
                    [65, 182, 196],
                    [44, 127, 184],
                    [37, 52, 148]
                ]
            ))
        
        # LAYER C: THE ACTUAL ROUTE
        if st.session_state['route_geometry']:
            layers.append(pdk.Layer(
                "PathLayer",
                data=st.session_state['route_geometry'],
                get_path="path",
                get_color="color",
                width_scale=20,
                width_min_pixels=3,
                pickable=True
            ))

        # LAYER E: Start/End Markers
        for loc, color in [(st.session_state['start_loc'], [0, 100, 255]), (st.session_state['end_loc'], [255, 50, 50])]:
            if loc:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=[loc],
                    get_position='[lon, lat]', get_fill_color=color + [200],
                    get_radius=150, stroked=True, get_line_color=[255,255,255], get_line_width=5
                ))

        # Render
        deck = pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=zoom, pitch=pitch),
            layers=layers,
            tooltip={"text": "{name}"}
        )
        st.pydeck_chart(deck)
    
    render_map()