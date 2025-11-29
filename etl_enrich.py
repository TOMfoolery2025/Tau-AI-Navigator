import requests
import pandas as pd
import os
import datetime
from neo4j import GraphDatabase
from ai_engine import TextNormalizer # Import the new Engine

# --- CONFIG ---
IMPORT_PATH = "/var/lib/neo4j/import"

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def fetch_landmarks_extended():
    """Fetches a broader range of POIs for semantic enrichment"""
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

def run_enrichment(driver):
    log("Starting Semantic Enrichment Process (Expert Mode)...")
    
    # 1. Init NLP Engine
    normalizer = TextNormalizer()
    
    landmarks = fetch_landmarks_extended()
    log(f"Fetched {len(landmarks)} raw POIs.")

    with driver.session() as session:
        # A. Import Raw Landmarks
        log("Creating PointOfInterest Nodes...")
        landmark_query = """
        UNWIND $batch AS row
        MERGE (p:PointOfInterest {id: row.id})
        SET p.name = row.tags.name, 
            p.raw_type = row.tags.tourism, 
            p.lat = row.lat, 
            p.lon = row.lon,
            p.description = row.tags.name + ' ' + coalesce(row.tags.tourism, '') + ' ' + coalesce(row.tags.historic, '')
        """
        # Filter POIs that actually have names
        valid_pois = [x for x in landmarks if 'tags' in x and 'name' in x['tags']]
        session.run(landmark_query, batch=valid_pois)

        # B. Spatial Inference (IS_NEAR)
        log("Inferring Spatial Links...")
        session.run("""
        MATCH (s:Stop), (p:PointOfInterest)
        WHERE point.distance(point({latitude: s.lat, longitude: s.lon}), point({latitude: p.lat, longitude: p.lon})) < 400
        MERGE (s)-[r:IS_NEAR]->(p)
        """)

        # C. Label Propagation (The TVA Logic)
        # We bubble up concepts: POI -> Stop -> Route
        log("Executing Label Propagation (POI -> Stop)...")
        
        # 1. Pull data for Python-side processing (Cypher is bad at text NLP)
        result = session.run("""
            MATCH (s:Stop)-[:IS_NEAR]->(p:PointOfInterest)
            WHERE p.name IS NOT NULL
            RETURN s.id as stop_id, collect(p.name + ' ' + coalesce(p.raw_type, '')) as poi_texts
        """)
        
        stop_tags = []
        for record in result:
            # Combine all text from nearby POIs
            combined_text = " ".join(record['poi_texts'])
            # Use TVA Normalizer to extract clean concepts (e.g., "art", "museum", "history")
            concepts = normalizer.clean_and_stem(combined_text)
            
            if concepts:
                stop_tags.append({"id": record['stop_id'], "tags": concepts})
        
        # 2. Write Tags back to Graph
        session.run("""
            UNWIND $batch as row
            MATCH (s:Stop {id: row.id})
            SET s.semantic_tags = row.tags
        """, batch=stop_tags)
        
        log(f"Propagated labels to {len(stop_tags)} stops.")
        
        # D. Route Classification (Vibe Check)
        # If a Route serves many 'art' stops, it becomes an 'Art Route'
        session.run("""
            MATCH (r:Route)-[:OPERATES_ON]->(s:Stop)
            WHERE s.semantic_tags IS NOT NULL
            WITH r, apoc.coll.flatten(collect(s.semantic_tags)) as all_tags
            WITH r, all_tags, size(all_tags) as total_count
            WHERE total_count > 5
            
            // Heuristic for simple classification based on stems
            SET r.vibe_art = size([x in all_tags WHERE x CONTAINS 'art' OR x CONTAINS 'museu']) > 2,
                r.vibe_nature = size([x in all_tags WHERE x CONTAINS 'park' OR x CONTAINS 'water']) > 2,
                r.vibe_historic = size([x in all_tags WHERE x CONTAINS 'hist' OR x CONTAINS 'cath']) > 2
        """)

    log("Enrichment Complete.")