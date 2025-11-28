import requests
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD

def generate_rdf_file(api_key):
    GTFS = Namespace("http://vocab.gtfs.org/terms#")
    EX = Namespace("http://example.org/hackathon/")
    GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
    
    g = Graph()
    g.bind("gtfs", GTFS)
    g.bind("geo", GEO)

    url = "https://api.digitransit.fi/routing/v2/hsl/gtfs/v1"
    
    query = """
    {
      stopsByRadius(lat: 60.171, lon: 24.941, radius: 1000) {
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
                }
            }
          }
        }
      }
    }
    """
    headers = {"Content-Type": "application/json", "digitransit-subscription-key": api_key}
    data = requests.post(url, json={"query": query}, headers=headers).json()
    
    edges = data.get('data', {}).get('stopsByRadius', {}).get('edges', [])

    for edge in edges:
        stop_data = edge['node']['stop']
        
        stop_uri = URIRef(EX[stop_data['gtfsId'].replace(":", "_")])
        
        g.add((stop_uri, RDF.type, GTFS.Stop))
        g.add((stop_uri, GTFS.name, Literal(stop_data['name'])))
        g.add((stop_uri, GEO.lat, Literal(stop_data['lat'], datatype=XSD.float)))
        g.add((stop_uri, GEO.long, Literal(stop_data['lon'], datatype=XSD.float)))
        
        for route in stop_data['routes']:
            route_uri = URIRef(EX[route['gtfsId'].replace(":", "_")])
            g.add((route_uri, RDF.type, GTFS.Route))
            g.add((route_uri, GTFS.shortName, Literal(route['shortName'])))
            g.add((route_uri, GTFS.serves, stop_uri))

    output_path = "/app/import_stage/helsinki_graph.ttl" 
    g.serialize(destination=output_path, format="turtle")
    return output_path