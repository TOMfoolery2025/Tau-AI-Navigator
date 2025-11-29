import streamlit as st
import os
import time
import json
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
from etl_rdf import generate_rdf_file
from etl_enrich import run_enrichment

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Helsinki AI Navigator", page_icon="🚋")

st.markdown("""
<style>
    .hero-title { font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #00C6FF, #0072FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton>button { border-radius: 20px; font-weight: bold; width: 100%; }
    .status-box { padding: 10px; border-radius: 8px; margin-bottom: 10px; font-weight: bold; }
    .success-box { background: rgba(0, 255, 100, 0.1); border: 1px solid #00ff64; color: #00ff64; }
    .error-box { background: rgba(255, 50, 50, 0.1); border: 1px solid #ff3232; color: #ff3232; }
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

@st.cache_resource
def get_driver():
    try: return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    except: return None
driver = get_driver()

# --- 2. DATA ENGINES (AI & GRAPH) ---

# Initialize Semantic Engine (Singleton)
@st.cache_resource
def get_semantic_engine():
    return VectorSearchEngine()

ai_engine = get_semantic_engine()

def initialize_search_index():
    """Syncs Neo4j data with the Vector Engine"""
    if driver:
        with driver.session() as session:
            # Fetch POIs with their descriptions (populated by etl_enrich.py)
            query = """
            MATCH (p:PointOfInterest)
            RETURN p.name as name, p.lat as lat, p.lon as lon, 
                   p.description as description, p.raw_type as type
            """
            result = session.run(query)
            pois = [r.data() for r in result]
            
            if pois:
                # Fit the index on the description field
                ai_engine.fit_index(pois, text_key='description')
                return len(pois)
    return 0

# Auto-initialize index if empty
if ai_engine.cached_embeddings is None:
    initialize_search_index()

def get_pois_semantically(query):
    """Vector Search for POIs"""
    if not query: return pd.DataFrame()
    results = ai_engine.search(query, top_k=15)
    return pd.DataFrame(results) if results else pd.DataFrame()

def geocode(text):
    """Converts 'Kamppi' -> Lat/Lon"""
    if not text: return None
    url = "https://api.digitransit.fi/geocoding/v1/search"
    params = {"text": text, "size": 1, "digitransit-subscription-key": HSL_KEY}
    try:
        resp = requests.get(url, params=params)
        if resp.status_code != 200: return None
        data = resp.json()
        if data['features']:
            props = data['features'][0]['properties']
            coords = data['features'][0]['geometry']['coordinates']
            return {"name": props['label'], "lat": coords[1], "lon": coords[0]}
    except: return None
    return None

def get_live_vehicles():
    """Fetches Trams AND Buses with Color Coding"""
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
                
                # Default: Bus (Blue)
                color = [0, 150, 255, 180] 
                radius = 30
                
                # Heuristic for Trams (Green)
                if len(short_id) <= 2 or short_id in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "15"]:
                    color = [0, 255, 100, 200]
                    radius = 40
                
                # Heuristic for Metro (Orange)
                if "M1" in short_id or "M2" in short_id:
                    color = [255, 140, 0, 200]
                    radius = 50

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
    """Fetches All Pink Dots from Neo4j (Fallback)"""
    if not driver: return pd.DataFrame()
    try:
        with driver.session() as session:
            result = session.run("MATCH (p:PointOfInterest) RETURN p.name as name, p.lat as lat, p.lon as lon")
            return pd.DataFrame([r.data() for r in result])
    except: return pd.DataFrame()

def ask_llm(query, start, end):
    """RAG Query for Tour Planning"""
    if not GROQ_KEY: return "AI Offline."
    client = Groq(api_key=GROQ_KEY)
    
    start_name = start['name'] if start else "Helsinki Center"
    end_name = end['name'] if end else "Anywhere"
    
    prompt = f"""
    Act as a Helsinki Transit Guide. 
    Plan a route from {start_name} to {end_name}.
    User Interest: "{query}"
    
    Provide a short, structured itinerary (3 bullet points).
    Mention specific Tram numbers (4, 7, etc.) if relevant.
    """
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except: return "AI Connection Failed."

def ask_general_llm(query):
    """General Query for 'Ask Navigator'"""
    if not GROQ_KEY: return "AI Offline."
    client = Groq(api_key=GROQ_KEY)
    prompt = f"""
    You are a Helsinki public transport expert. 
    User asks: "{query}"
    Answer briefly about fares, timings, or general info.
    Current context: Single ticket 2.95€, Day ticket 9€. Trams run approx 06:00-01:30.
    """
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except: return "AI Error."

# --- 3. UI LAYOUT ---

col_logo, col_title = st.columns([1, 5])
with col_logo: st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Helsinki_vaakuna.svg/1200px-Helsinki_vaakuna.svg.png", width=70)
with col_title: st.markdown('<div class="hero-title">HELSINKI AI NAVIGATOR</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 2.5], gap="medium")

# --- LEFT COLUMN: CONTROLS ---
with col_left:
    st.markdown("### 🗺️ Plan Your Journey")
    
    # 1. LIVE GPS TOGGLE
    if st.toggle("📍 Use GPS Location"):
        loc = get_geolocation()
        if loc and 'coords' in loc:
            st.session_state['start_loc'] = {"name": "My Location", "lat": loc['coords']['latitude'], "lon": loc['coords']['longitude']}
            st.success(f"GPS Locked: {loc['coords']['latitude']:.3f}, {loc['coords']['longitude']:.3f}")
        else:
            st.warning("Requesting GPS...")

    # 2. START INPUT
    start_val = st.session_state['start_loc']['name'] if st.session_state['start_loc'] else ""
    start_query = st.text_input("From (Start)", value=start_val, placeholder="e.g. Kamppi")
    
    if start_query and start_query != "My Location":
        res = geocode(start_query)
        if res:
            st.session_state['start_loc'] = res
            st.markdown(f"<div class='status-box success-box'>✅ Found: {res['name']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='status-box error-box'>❌ Not found: {start_query}</div>", unsafe_allow_html=True)

    # 3. DESTINATION INPUT
    end_query = st.text_input("To (Destination)", placeholder="e.g. Airport")
    if end_query:
        res = geocode(end_query)
        if res:
            st.session_state['end_loc'] = res
            st.markdown(f"<div class='status-box success-box'>🏁 Found: {res['name']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='status-box error-box'>❌ Not found: {end_query}</div>", unsafe_allow_html=True)

    # 4. EXPLORE
    interest = st.text_input("Vibe / Interest", placeholder="e.g. Architecture, Cheap Food")
    
    if st.button("🚀 Generate Route"):
        # A. Run Semantic Vector Search
        if interest:
            found_pois = get_pois_semantically(interest)
            if not found_pois.empty:
                st.session_state['semantic_pois'] = found_pois
                st.toast(f"Found {len(found_pois)} locations matching '{interest}'", icon="🎯")
            else:
                st.session_state['semantic_pois'] = None
                st.toast("No specific POIs found for that interest.", icon="⚠️")

        # B. Run LLM Planner
        with st.spinner("AI Planning..."):
            plan = ask_llm(interest, st.session_state['start_loc'], st.session_state['end_loc'])
            st.info(plan)

    st.markdown("---")
    st.markdown("### 💬 Ask the Navigator")
    general_q = st.text_input("Questions about Fares/Timings:", placeholder="How much is a ticket?")
    if general_q:
        ans = ask_general_llm(general_q)
        st.success(ans)

    st.divider()
    
    # 5. LIVE TRACKING CONTROL
    live_mode = st.toggle("📡 Live Tracking Mode", value=True)
    if live_mode:
        st.caption("Map creates live animation loop. Uncheck to stop.")

    if st.button("🔄 Reload Data"):
        if driver:
            with st.spinner("Rebuilding Knowledge Graph..."):
                run_neo4j_import(driver, HSL_KEY)
                run_enrichment(driver)
                # Re-index the Vector Engine
                cnt = initialize_search_index()
                st.session_state['semantic_pois'] = None # Reset search filter
                st.toast(f"System Refreshed. {cnt} items indexed.", icon="📡")

# --- RIGHT COLUMN: MAP ---
with col_right:
    # Map Placeholder for Animation
    map_placeholder = st.empty()
    
    # Static Data Logic: Use Semantic Results if active, else All POIs
    if st.session_state['semantic_pois'] is not None:
        pois_df = st.session_state['semantic_pois']
        poi_color = [255, 200, 0, 255] # Gold for Search Results
        poi_radius = 120
    else:
        pois_df = get_graph_pois()
        poi_color = [255, 0, 128, 200] # Pink for Generic
        poi_radius = 80
    
    # View State Logic
    view_lat, view_lon, zoom = 60.1699, 24.9384, 13
    if st.session_state['start_loc']:
        view_lat = st.session_state['start_loc']['lat']
        view_lon = st.session_state['start_loc']['lon']
        zoom = 14

    # ANIMATION LOOP
    while True:
        # 1. Fetch Real-Time Vehicles
        vehicles_df = get_live_vehicles()
        
        layers = []
        
        # Layer A: Live Vehicles
        if not vehicles_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", vehicles_df,
                get_position='[lon, lat]', get_fill_color='color',
                get_radius='radius', pickable=True, auto_highlight=True
            ))

        # Layer B: Graph POIs (Semantic or All)
        if not pois_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", pois_df,
                get_position='[lon, lat]', get_fill_color=poi_color,
                get_radius=poi_radius, pickable=True, stroked=True, get_line_color=[255,255,255],
                # Add similarity score to tooltip if available
                tooltip="name"
            ))

        # Layer C: Start/End Markers
        if st.session_state['start_loc']:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=[st.session_state['start_loc']],
                get_position='[lon, lat]', get_fill_color=[0, 100, 255, 255],
                get_radius=150, stroked=True, get_line_color=[255,255,255], get_line_width=5
            ))
        if st.session_state['end_loc']:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=[st.session_state['end_loc']],
                get_position='[lon, lat]', get_fill_color=[255, 50, 50, 255],
                get_radius=150, stroked=True, get_line_color=[255,255,255], get_line_width=5
            ))

        # Render Map inside Placeholder
        deck = pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=zoom, pitch=45),
            layers=layers,
            tooltip={"text": "{name}\n{description}"} if 'description' in pois_df.columns else {"text": "{name}"}
        )
        map_placeholder.pydeck_chart(deck)

        # Break loop if Live Mode is turned off (allows UI interaction)
        if not live_mode:
            break
            
        time.sleep(2) # 2-second refresh rate