# 🚋 Helsinki AI Navigator

A Next-Gen Semantic Travel Guide powered by Knowledge Graphs, Vector Search, and Real-Time IoT Data.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![Neo4j](https://img.shields.io/badge/neo4j-5.0+-red.svg)

---

## 🌟 The Pitch

Standard maps tell you **where** things are. Helsinki AI Navigator tells you **how it feels to be there**.

By combining a **Neo4j Knowledge Graph** (for spatial logic) with a **Vector Search Engine** (for semantic understanding), we allow users to search for "Vibes" (e.g., "Underground Techno", "Cozy Book Cafe") rather than just keywords. The app then visualizes live transport options and generates a narrative audio tour guide on the fly.

---

## 🚀 Features

- 🎨 **Semantic "Vibe" Search**: Don't search for "Bar." Search for "gloomy aesthetic with cheap beer." Our Vector Engine finds the best matches.
- 🔥 **Vibe Heatmaps**: Visualizes the density of your interest across the city using glowing 2D heat layers.
- 🕸️ **Hybrid Graph-RAG**: Uses Neo4j to understand connectivity between Stops and POIs, and Llama-3 to generate witty, context-aware itineraries.
- 📡 **Live Digital Twin**: Real-time tracking of Trams (Green), Buses (Blue), and Metro (Orange) using the HSL GTFS-RT feed.
- 🗺️ **Cinematic Routing**: Renders exact street-level path geometry (not just straight lines) using the Digitransit GraphQL API.
- 🔊 **AI Audio Guide**: The app "speaks" to you, acting as a sarcastic or helpful local guide based on your location.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Frontend** | Streamlit | Custom "Nordic Night" Glassmorphism UI |
| **Visualization** | PyDeck (Deck.gl) | 3D ArcLayers, PathLayers, and HeatmapLayers |
| **Knowledge Graph** | Neo4j | Stores Stops, Routes, POIs, and `IS_NEAR` relationships |
| **Vector DB** | FAISS / Local | In-memory semantic index using `sentence-transformers` |
| **LLM** | Groq (Llama-3) | Generates route narratives and tour scripts |
| **Data Source** | Digitransit API | GraphQL for routing, GTFS-RT for live vehicles |
| **ETL** | Python | Custom scripts to fetch OSM & GTFS data |

---

## 🏗️ Architecture

```mermaid
graph TD
    User[User Query: 'Hipster Cafe'] --> Vector[Vector Search Engine]
    Vector -->|Embeddings| Semantic[Semantic Match]
    
    User -->|GPS/Input| Geocoder[HSL Geocoder]
    
    Semantic --> Graph[Neo4j Knowledge Graph]
    Geocoder --> Graph
    
    Graph -->|Context + POIs| LLM[Groq Llama-3]
    LLM -->|Itinerary| UI[Streamlit UI]
    
    Live[HSL Realtime API] -->|Vehicle Positions| UI
    Routing[Digitransit GraphQL] -->|Path Geometry| UI
```

---

## ⚡ Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- A [Groq API Key](https://groq.com/) (for LLM features)
- A Digitransit API Key (for Routing/Geocoding)

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
NEO4J_URI=bolt://neo4j_db:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
GROQ_API_KEY=gsk_your_key_here
DIGITRANSIT_API_KEY=your_digitransit_key
```

### 3. Run with Docker

The easiest way to spin up the Graph DB and the App simultaneously.

```bash
docker-compose up --build
```

Access the app at **http://localhost:7860**.

### 4. Initialize Data (Important!)

On the first run, the database will be empty.

1. Open the app sidebar.
2. Click **"🔄 Reload Graph"**.

This triggers the ETL pipeline:
- `etl_neo4j.py`: Fetches Stops from HSL and nodes them in Neo4j.
- `etl_enrich.py`: Fetches POIs from OpenStreetMap, connects them to Stops, and indexes them for Vector Search.

---

## 📸 Screenshots

### The "Nordic Night" UI
![Nordic Night UI](docs/images/ui_screenshot.png)
*Dark mode glassmorphism interface with real-time transit overlay*

### Semantic Heatmaps
![Vibe Heatmap](docs/images/heatmap_screenshot.png)
*Live visualization of "vibe density" across Helsinki*

---

## 🙏 Credits

- Data provided by [Digitransit](https://digitransit.fi/) & [OpenStreetMap](https://www.openstreetmap.org/)
- LLM powered by [Groq](https://groq.com/)

---
