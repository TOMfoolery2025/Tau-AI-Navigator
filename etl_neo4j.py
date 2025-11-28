import requests
import math
from neo4j import GraphDatabase

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000 # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def run_neo4j_import(driver, api_key):
    print("🚀 Building Semantic Knowledge Graph...")

    url = "https://api.digitransit.fi/routing/v2/hsl/gtfs/v1"
    query = """
    {
      stopsByRadius(lat: 60.171, lon: 24.941, radius: 1500, first: 200) {
        edges {
          node {
            stop { 
              gtfsId
              name
              lat
              lon
              routes {
                gtfsId
                shortName
                mode
              }
            }
          }
        }
      }
    }
    """
    headers = {"Content-Type": "application/json", "digitransit-subscription-key": api_key}
    
    try:
        response = requests.post(url, json={"query": query}, headers=headers)
        data = response.json()
        edges = data.get('data', {}).get('stopsByRadius', {}).get('edges', [])
        
        if not edges:
            print("⚠️ No data found.")
            return 0

        stops_list = []
        routes_list = []
        serves_rels = []
        
        stop_cache = []

        for edge in edges:
            s_node = edge['node']['stop']
            s_id = s_node['gtfsId']
            
            stops_list.append({
                "id": s_id,
                "name": s_node['name'],
                "lat": s_node['lat'],
                "lon": s_node['lon'],
                "type": "sem:Stop"
            })
            
            stop_cache.append(s_node)

            for r in s_node['routes']:
                r_id = r['gtfsId']
                routes_list.append({
                    "id": r_id,
                    "name": r['shortName'],
                    "mode": r['mode'], 
                    "type": "sem:Route"
                })
                serves_rels.append({"route_id": r_id, "stop_id": s_id})

        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Stop) REQUIRE s.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Route) REQUIRE r.id IS UNIQUE")

            print("... Creating Semantic Stop Nodes")
            session.run("""
            UNWIND $batch AS row
            MERGE (s:Stop {id: row.id})
            SET s.name = row.name, s.lat = row.lat, s.lon = row.lon, s.ontologyType = row.type
            """, batch=stops_list)

            print("... Creating Semantic Route Nodes")
            unique_routes = {v['id']: v for v in routes_list}.values()
            session.run("""
            UNWIND $batch AS row
            MERGE (r:Route {id: row.id})
            SET r.name = row.name, r.mode = row.mode, r.ontologyType = row.type
            """, batch=list(unique_routes))

            print("... Linking Topology (OPERATES_ON)")
            session.run("""
            UNWIND $batch AS row
            MATCH (r:Route {id: row.route_id})
            MATCH (s:Stop {id: row.stop_id})
            MERGE (r)-[:OPERATES_ON]->(s)
            """, batch=serves_rels)

            print("... Inferring Spatial Relationships")
            
            walk_links = []
            for i in range(len(stop_cache)):
                for j in range(i + 1, len(stop_cache)):
                    s1 = stop_cache[i]
                    s2 = stop_cache[j]
                    dist = calculate_distance(s1['lat'], s1['lon'], s2['lat'], s2['lon'])
                    
                    if dist < 150: # 150 meters
                        walk_links.append({"a": s1['gtfsId'], "b": s2['gtfsId'], "dist": dist})

            if walk_links:
                session.run("""
                UNWIND $batch AS row
                MATCH (a:Stop {id: row.a})
                MATCH (b:Stop {id: row.b})
                MERGE (a)-[rel:WALKABLE_TO]->(b)
                SET rel.distance_meters = row.dist
                """, batch=walk_links)

        print(f"✅ Semantic Graph Built! ({len(stops_list)} Stops, {len(walk_links)} Walk Links)")
        return len(stops_list)

    except Exception as e:
        print(f"❌ ETL Error: {e}")
        return 0