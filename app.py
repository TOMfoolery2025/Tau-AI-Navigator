import streamlit as st
import os
import time
import json
import requests
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta
from groq import Groq
from google.transit import gtfs_realtime_pb2
from neo4j import GraphDatabase
from streamlit_js_eval import get_geolocation
from streamlit_searchbox import st_searchbox  # <--- NEW COMPONENT

# --- CUSTOM MODULES ---
from ai_engine import VectorSearchEngine
from etl_neo4j import run_neo4j_import
from etl_enrich import run_enrichment
from etl_static import load_static_lookups 

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Helsinki AI Navigator", page_icon="🚋")

st.markdown("""
<style>
    .hero-title { font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #00C6FF, #0072FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton>button { border-radius: 20px; font-weight: bold; width: 100%; }
    .status-box { padding: 10px; border-radius: 8px; margin-bottom: 10px; font-weight: bold; }
    /* Fix Z-Index for Searchbox to appear above map */
    div[data-testid="stVerticalBlock"] > div:has(div.stSearchbox) { z-index: 1000; }
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

# Initialize Vector Index
if ai_engine.cached_embeddings is None:
    if driver:
        with driver.session() as session:
            # --- OLD BROKEN QUERY ---
            # query = "MATCH (p:PointOfInterest) RETURN p.name as name, p.description as description"
            
            # --- NEW FIXED QUERY (Include Lat/Lon) ---
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
    """
    GOOGLE-STYLE AUTOCOMPLETE FUNCTION
    Called dynamically by st_searchbox as the user types.
    """
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
                coords = f['geometry']['coordinates'] # [lon, lat]
                
                # We pack the data into a JSON string to pass it back to Streamlit
                value = json.dumps({"name": label, "lat": coords[1], "lon": coords[0]})
                suggestions.append((label, value))
            return suggestions
    except: return []
    return []

def get_planned_itinerary(start, end, departure_time=None):
    """
    Queries HSL GraphQL API for a planned route at a specific time.
    """
    if not start or not end: return ""
    
    # Format time for GraphQL (ISO 8601)
    # If no time provided, use NOW.
    if departure_time:
        # User provided time (datetime object)
        time_str = departure_time.strftime("%Y-%m-%dT%H:%M:%S") + "+02:00" # Adding timezone manually for simplicity
        time_mode = f'dateTime: "{time_str}"'
    else:
        time_mode = "" # Defaults to NOW

    query = """
    {
      plan(
        from: {lat: %s, lon: %s}
        to: {lat: %s, lon: %s}
        numItineraries: 3
        %s
      ) {
        itineraries {
          duration
          legs {
            mode
            startTime
            endTime
            route { shortName }
            from { name }
            to { name }
          }
        }
      }
    }
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
                from_nm = leg['from']['name']
                to_nm = leg['to']['name']
                
                if mode == 'WALK':
                    context_str += f"  - Walk from {from_nm} to {to_nm}\n"
                else:
                    context_str += f"  - At {start_t}, take {mode} {route} from {from_nm} to {to_nm}\n"
        return context_str
    except Exception as e:
        return f"Planner Error: {str(e)}"

def ask_llm(query, start, end, semantic_pois=None, planned_time=None):
    if not GROQ_KEY: return "AI Offline."
    client = Groq(api_key=GROQ_KEY)
    
    start_name = start['name'] if start else "Helsinki Center"
    target_name = end['name'] if end else "Destination"
    
    # 1. Resolve Destination from Vibe if needed
    if semantic_pois is not None and not semantic_pois.empty:
        target_name = semantic_pois.iloc[0]['name']
        # Update session state end_loc properly for the planner to work
        end = {'lat': semantic_pois.iloc[0]['lat'], 'lon': semantic_pois.iloc[0]['lon'], 'name': target_name}

    # 2. Get Route Plan (The Truth)
    planner_context = get_planned_itinerary(start, end, planned_time)
    
    prompt = f"""
    Act as a Helsinki Guide.
    User Request: Go from {start_name} to {target_name}.
    User Vibe: "{query}"
    Requested Time: {planned_time if planned_time else "NOW"}
    
    OFFICIAL ROUTING DATA:
    {planner_context}
    
    INSTRUCTIONS:
    1. Summarize the best route option from the data above.
    2. Be conversational. "The best way is to take the Tram 4..."
    3. If the user asked for a specific Vibe, mention why the destination fits (e.g. "It's great for Modern Art").
    """
    
    try:
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    except: return "AI Error."

def get_live_vehicles():
    """
    FIXED VERSION: Solves the "Missing Destination" bug
    """
    try:
        url = "https://realtime.hsl.fi/realtime/vehicle-positions/v2/hsl"
        headers = {"digitransit-subscription-key": HSL_KEY} if HSL_KEY else {}
        resp = requests.get(url, headers=headers, timeout=2)
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)
        
        vehicles = []
        for e in feed.entity:
            if e.HasField('vehicle') and e.vehicle.position:
                # 1. IDs
                r_id = e.vehicle.trip.route_id.replace("HSL:", "").strip() if e.vehicle.trip.route_id else ""
                t_id = e.vehicle.trip.trip_id.replace("HSL:", "").strip()
                d_id = str(e.vehicle.trip.direction_id)
                
                # 2. Lookup Route Info
                route_data = routes_dict.get(r_id, {"short": r_id, "mode": "BUS", "long": ""})
                
                # 3. CRITICAL FIX: Trip Headsign Lookup with Fallback
                # First try exact trip_id match
                headsign = trip_lookup.get(t_id)
                # If failed, try Route + Direction match (The Fallback)
                if not headsign:
                    headsign = direction_lookup.get((r_id, d_id), "Unknown")
                
                mode = route_data['mode']
                short_name = route_data['short']
                
                # 4. Description Formatting
                desc = f"{mode} {short_name}: To {headsign}"
                details = f"Route: {route_data['long']}"
                
                # 5. Styling
                if mode == 'TRAM': color, radius = [0, 200, 100, 200], 40
                elif mode == 'METRO': color, radius = [255, 140, 0, 200], 50
                elif mode == 'TRAIN': color, radius = [200, 0, 0, 200], 50
                elif mode == 'FERRY': color, radius = [0, 100, 255, 200], 60
                else: color, radius = [0, 150, 255, 180], 30

                vehicles.append({
                    "lat": e.vehicle.position.latitude,
                    "lon": e.vehicle.position.longitude,
                    "color": color,
                    "radius": radius,
                    "desc": desc,
                    "details": details
                })
        return pd.DataFrame(vehicles)
    except: return pd.DataFrame()

# --- 4. UI LAYOUT ---

col_logo, col_title = st.columns([1, 5])
with col_logo: st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Helsinki_vaakuna.svg/1200px-Helsinki_vaakuna.svg.png", width=70)
with col_title: st.markdown('<div class="hero-title">HELSINKI AI NAVIGATOR</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 2.5], gap="medium")

with col_left:
    st.markdown("### 🗺️ Plan Your Journey")
    
    # --- GOOGLE-STYLE SEARCHBOXES ---
    st.markdown("**From:**")
    selected_start = st_searchbox(
        search_hsl_places, 
        key="start_search", 
        placeholder="Type to search (e.g. Kamppi)..."
    )
    if selected_start:
        st.session_state['start_loc'] = json.loads(selected_start)

    st.markdown("**To:**")
    selected_end = st_searchbox(
        search_hsl_places, 
        key="end_search", 
        placeholder="Type destination..."
    )
    if selected_end:
        st.session_state['end_loc'] = json.loads(selected_end)

    # --- TIME PICKER ---
    with st.expander("🕒 Preferred Time (Optional)", expanded=False):
        d = st.date_input("Date", datetime.now())
        t = st.time_input("Time", datetime.now())
        # Combine into one datetime object
        planned_dt = datetime.combine(d, t)

    interest = st.text_input("Or search by Vibe:", placeholder="e.g. Quiet Library, Brutalist Architecture")
    
    if st.button("🚀 Generate Route"):
        # Semantic Search (if Vibe is present)
        found_pois = None
        if interest:
            results = ai_engine.search(interest, top_k=5)
            if results:
                found_pois = pd.DataFrame(results)
                st.session_state['semantic_pois'] = found_pois
                st.toast(f"Found {len(results)} locations for vibe", icon="🎯")
            else:
                st.session_state['semantic_pois'] = None

        # LLM Planning
        if st.session_state['start_loc']:
            with st.spinner("Querying HSL Planner..."):
                plan = ask_llm(interest, st.session_state['start_loc'], st.session_state['end_loc'], found_pois, planned_dt)
                st.info(plan)
        else:
            st.error("Please select a Start Location first.")

    if st.button("🔄 Reload Data"):
        if driver:
            with st.spinner("Refreshing..."):
                run_neo4j_import(driver, HSL_KEY)
                run_enrichment(driver)
                st.success("Refreshed!")

# --- RIGHT COLUMN: MAP ---
with col_right:
    map_placeholder = st.empty()
    
    # Layer B: POIs
    if st.session_state['semantic_pois'] is not None:
        pois_df = st.session_state['semantic_pois']
        poi_color = [255, 200, 0, 255] # Gold
    else:
        pois_df = pd.DataFrame() # Don't show all pink dots by default to keep map clean? Or fetch generic ones.
        # Uncomment below to show all pink dots again
        # pois_df = get_graph_pois() 
        poi_color = [255, 0, 128, 200]

    view_lat, view_lon, zoom = 60.1699, 24.9384, 13
    if st.session_state['start_loc']:
        view_lat = st.session_state['start_loc']['lat']
        view_lon = st.session_state['start_loc']['lon']
        zoom = 14

    while True:
        layers = []
        # Layer A: Live Vehicles
        v_df = get_live_vehicles()
        if not v_df.empty:
            layers.append(pdk.Layer("ScatterplotLayer", v_df, get_position='[lon, lat]', get_fill_color='color', get_radius='radius', pickable=True, auto_highlight=True, tooltip="desc"))
            
        # Layer B: POIs
        if not pois_df.empty:
            layers.append(pdk.Layer("ScatterplotLayer", pois_df, get_position='[lon, lat]', get_fill_color=poi_color, get_radius=30, pickable=True, stroked=True, get_line_color=[255,255,255], tooltip="name"))

        # Layer C: Start/End
        if st.session_state['start_loc']:
             layers.append(pdk.Layer("ScatterplotLayer", data=[st.session_state['start_loc']], get_position='[lon, lat]', get_fill_color=[0, 100, 255, 255], get_radius=100))
        if st.session_state['end_loc']:
             layers.append(pdk.Layer("ScatterplotLayer", data=[st.session_state['end_loc']], get_position='[lon, lat]', get_fill_color=[255, 50, 50, 255], get_radius=100))

        deck = pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=zoom, pitch=45),
            layers=layers,
            tooltip={"html": "<b>{desc}</b><br/>{details}" if not v_df.empty else "<b>{name}</b>", "style": {"backgroundColor": "steelblue", "color": "white"}}
        )
        map_placeholder.pydeck_chart(deck)
        time.sleep(2)