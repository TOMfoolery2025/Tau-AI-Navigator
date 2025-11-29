import requests
import datetime
from neo4j import GraphDatabase

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def fetch_landmarks_extended():
    """Fetches POIs from Overpass API"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json][timeout:25];
    (
      node["tourism"](60.15,24.90,60.20,24.98);
      node["leisure"](60.15,24.90,60.20,24.98);
      node["amenity"="arts_centre"](60.15,24.90,60.20,24.98);
      node["historic"](60.15,24.90,60.20,24.98);
    );
    out body;
    """
    try:
        response = requests.get(overpass_url, params={'data': overpass_query})
        return response.json().get('elements', [])
    except Exception as e:
        log(f"Error fetching landmarks: {e}")
        return []

def get_placeholder_image(tag_type):
    """Assigns a high-quality 'Vibe Image' based on category"""
    # Hackathon Trick: Use specific Unsplash IDs for reliable, beautiful images
    IMAGES = {
        'museum': 'https://images.unsplash.com/photo-1545989253-02cc26577f88?w=400&q=80', # Art
        'gallery': 'https://images.unsplash.com/photo-1518998053901-5348d3969105?w=400&q=80', # Gallery
        'park': 'https://images.unsplash.com/photo-1496347312933-125c92842fa3?w=400&q=80',   # Nature
        'viewpoint': 'https://images.unsplash.com/photo-1502786129293-79981cb61638?w=400&q=80', # View
        'attraction': 'https://images.unsplash.com/photo-1548957175-84f0f9af659e?w=400&q=80', # City
        'sauna': 'https://images.unsplash.com/photo-1574673627192-3c46927d2c3c?w=400&q=80',   # Sauna
        'default': 'https://images.unsplash.com/photo-1538332539566-b5d1e679a636?w=400&q=80'  # Helsinki Cathedral
    }
    
    tag_lower = str(tag_type).lower()
    if 'museum' in tag_lower or 'art' in tag_lower: return IMAGES['museum']
    if 'park' in tag_lower or 'garden' in tag_lower: return IMAGES['park']
    if 'view' in tag_lower: return IMAGES['viewpoint']
    if 'sauna' in tag_lower: return IMAGES['sauna']
    return IMAGES['default']

def run_enrichment(driver):
    log("Starting Semantic Enrichment...")
    
    landmarks = fetch_landmarks_extended()
    log(f"Fetched {len(landmarks)} raw POIs.")

    with driver.session() as session:
        # A. Import POIs with Images
        log("Creating Rich POI Nodes...")
        
        poi_batch = []
        for row in landmarks:
            if 'tags' in row and 'name' in row['tags']:
                tags = row['tags']
                # Determine main type
                raw_type = tags.get('tourism') or tags.get('leisure') or tags.get('amenity') or 'landmark'
                
                poi_batch.append({
                    "id": row['id'],
                    "name": tags['name'],
                    "lat": row['lat'],
                    "lon": row['lon'],
                    "type": raw_type,
                    "desc": f"{tags.get('name')} ({raw_type})",
                    # Generate Image URL
                    "image": get_placeholder_image(raw_type)
                })

        # Batch Write
        query = """
        UNWIND $batch AS row
        MERGE (p:PointOfInterest {id: row.id})
        SET p.name = row.name, 
            p.raw_type = row.type, 
            p.lat = row.lat, 
            p.lon = row.lon,
            p.description = row.desc,
            p.image_url = row.image
        """
        session.run(query, batch=poi_batch)

        # B. Link Stops to POIs (Distance Based)
        log("Linking Stops to POIs...")
        session.run("""
        MATCH (s:Stop), (p:PointOfInterest)
        WHERE point.distance(point({latitude: s.lat, longitude: s.lon}), point({latitude: p.lat, longitude: p.lon})) < 400
        MERGE (s)-[r:IS_NEAR]->(p)
        """)

    log("Enrichment Complete.")