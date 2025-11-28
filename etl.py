import requests
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import FOAF, XSD

# 1. DEFINE NAMESPACES
GTFS = Namespace("http://vocab.gtfs.org/terms#")
TOUR = Namespace("http://example.org/tour-ontology#")
GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
EX = Namespace("http://example.org/resource/")

g = Graph()
g.bind("gtfs", GTFS)
g.bind("tour", TOUR)
g.bind("geo", GEO)

url = "https://api.digitransit.fi/routing/v2/hsl/gtfs/v1"
query = """
{
  stopsByRadius(lat: 60.171, lon: 24.941, radius: 500) {
    edges {
      node {
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
"""
headers = {"Content-Type": "application/json", "digitransit-subscription-key": "YOUR_API_KEY"}
response = requests.post(url, json={"query": query}, headers=headers).json()

stops = response['data']['stopsByRadius']['edges']

print(f"Transforming {len(stops)} stops into Knowledge Graph triples...")

for edge in stops:
    stop_data = edge['node']
    stop_uri = URIRef(EX[stop_data['gtfsId'].replace(":", "_")]) # e.g., ex:HSL_102030
    
    g.add((stop_uri, RDF.type, GTFS.Stop))
    g.add((stop_uri, GTFS.name, Literal(stop_data['name'])))
    g.add((stop_uri, GEO.lat, Literal(stop_data['lat'], datatype=XSD.float)))
    g.add((stop_uri, GEO.long, Literal(stop_data['lon'], datatype=XSD.float)))
    
    for route in stop_data['routes']:
        route_uri = URIRef(EX[route['gtfsId'].replace(":", "_")])
        g.add((route_uri, RDF.type, GTFS.Route))
        g.add((route_uri, GTFS.shortName, Literal(route['shortName'])))
        g.add((route_uri, GTFS.routeType, Literal(route['mode'])))
        
        g.add((route_uri, GTFS.serves, stop_uri))

print(g.serialize(format="turtle"))
g.serialize(destination="helsinki_graph.ttl", format="turtle")