I want to build a comprehensive AI-powered floor plan design system called "VastuVision AI". 

## Project Overview
Create a multi-agent AI system that generates scientifically-validated residential floor plans by analyzing environmental, geological, and cultural factors.

## Core Requirements

### 1. Multi-Agent System
Implement the following specialized AI agents using LangChain/LangGraph:
- **Orchestrator Agent**: Coordinates workflow between all agents
- **Architect Agent**: Creates initial floor plan layouts
- **Meteorologist Agent**: Analyzes weather patterns, wind direction, rainfall data
- **Geologist Agent**: Evaluates soil type, elevation, seismic considerations
- **Structural Engineer Agent**: Validates structural integrity and load calculations
- **Site Engineer Agent**: Reviews site-specific construction constraints
- **Vastu Expert Agent**: Applies traditional Vastu Shastra principles
- **Interior Designer Agent**: Optimizes room functionality and aesthetics
- **Construction Builder Agent**: Generates construction-ready specifications

### 2. Environmental Data Integration
Connect to and process data from:
- Geolocation services (latitude, longitude, timezone)
- Elevation APIs (Google Elevation or alternatives)
- Weather APIs (OpenWeatherMap for historical data)
- Solar path calculation (PVLib for sun position throughout the year)
- Wind pattern analysis (NOAA or local meteorological data)
- Rainfall data (annual precipitation patterns)

### 3. Scientific Validation
Implement validators that ensure:
- Optimal natural lighting based on sun path (south-facing in northern hemisphere)
- Cross-ventilation aligned with prevailing wind directions
- Thermal mass placement for passive heating/cooling
- Structural stability for the geological conditions
- Energy efficiency (minimize heat gain/loss)
- Compliance with local building codes

### 4. Design Generation
Create modules to:
- Parse plot dimensions and constraints
- Generate room layouts with precise dimensions
- Calculate optimal window/door placements and orientations
- Determine wall thicknesses based on structural requirements
- Suggest material specifications based on climate

### 5. Output Formats
Generate the following deliverables:
- **2D Floor Plan**: High-quality images (PNG/SVG) with dimensions
- **AutoCAD DXF File**: Industry-standard CAD format for architects
- **3D Model**: Interactive 3D visualization (GLTF or OBJ format)
- **Technical Documentation**: PDF with design rationale, calculations, and specifications
- **Sun Path Analysis**: Visualization showing sunlight exposure throughout the year
- **Ventilation Analysis**: Airflow diagrams based on wind data
- **Material Specifications**: BOM (Bill of Materials) with quantities

### 6. Technology Stack
Use the following:
- **Backend**: Python 3.10+, FastAPI, Celery for async tasks
- **AI Framework**: LangChain + LangGraph for agent orchestration
- **LLM**: OpenAI GPT-4 or Anthropic Claude 3.5 Sonnet
- **Vector DB**: ChromaDB or Pinecone for RAG (building codes, Vastu rules)
- **Databases**: PostgreSQL + PostGIS for spatial data, Neo4j for knowledge graph
- **Geometry**: Shapely for 2D operations, PyProj for geographic projections
- **Solar Calculations**: PVLib for sun position
- **CAD Generation**: Ezdxf for DXF files, CadQuery for 3D models
- **Visualization**: Matplotlib, Plotly, Three.js
- **Message Queue**: Redis for inter-agent communication

### 7. Knowledge Base
Include data for:
- Vastu Shastra principles (stored in JSON or database)
- International Building Codes (IBC) and local codes
- Material thermal properties
- Structural engineering formulas
- Climate zone classifications

### 8. API Design
Create REST API endpoints:
- `POST /api/design/analyze-plot` - Start design process
- `GET /api/design/{job_id}/status` - Check progress
- `GET /api/design/{job_id}/result` - Retrieve final design
- `POST /api/design/{job_id}/regenerate` - Regenerate with new constraints
- `GET /api/environmental/sun-path` - Get sun path data
- `GET /api/validation/report` - Get scientific validation report

### 9. Project Structure
Organize code as:
smartplot-architect
├── src/
│ ├── agents/ # All AI agents
│ ├── services/ # External API integrations
│ ├── validators/ # Scientific validation logic
│ ├── processors/ # Design generation and output
│ ├── models/ # Data models (Pydantic)
│ ├── utils/ # Helper functions
│ └── config/ # Configuration management
├── data/
│ ├── vastu_rules.json
│ ├── building_codes/
│ └── material_specs.json
├── tests/ # Unit and integration tests
├── api/ # FastAPI application
├── requirements.txt
└── README.md

### 10. Key Features
- **Conflict Resolution**: When agents disagree, use weighted scoring (science > tradition)
- **Iterative Refinement**: Allow agents to review each other's work
- **Explanation Generation**: Each design decision must have documented reasoning
- **Customization**: Support user preferences (budget, style, room count)
- **Accessibility**: Generate designs that meet ADA/accessibility standards

### 11. Example Workflow
1. User inputs: plot location, dimensions, budget, room requirements
2. Environmental analyzer collects all data (weather, sun, wind, elevation, soil)
3. Geologist validates plot suitability
4. Meteorologist provides climate recommendations
5. Architect generates initial layout optimized for sun/wind
6. Vastu expert suggests adjustments (where scientifically sound)
7. Structural engineer validates and adjusts for load-bearing
8. Interior designer optimizes room functionality
9. Construction builder generates specifications
10. System generates all output files (2D, 3D, DXF, PDF)

### 12. Sample Input
json
{
  "location": {
    "address": "123 Main St, Bangalore, India",
    "coordinates": {"lat": 12.9716, "lon": 77.5946}
  },
  "plot": {
    "dimensions": {"length": 30, "width": 50, "unit": "feet"},
    "orientation": "north",
    "road_facing": "east"
  },
  "requirements": {
    "bedrooms": 3,
    "bathrooms": 2,
    "kitchen": 1,
    "living_room": 1,
    "dining_room": 1,
    "budget": "mid-range",
    "style": "modern",
    "apply_vastu": true
  }
}

### 13. Expected Output
{
  "design_id": "uuid",
  "files": {
    "floor_plan_2d": "url_to_png",
    "autocad_file": "url_to_dxf",
    "3d_model": "url_to_gltf",
    "documentation": "url_to_pdf",
    "sun_analysis": "url_to_visualization"
  },
  "summary": {
    "total_area": "1500 sq ft",
    "room_count": 8,
    "optimization_score": 8.7,
    "energy_efficiency": "A+",
    "vastu_compliance": 92
  },
  "design_decisions": [
    {
      "agent": "architect",
      "decision": "Placed living room on south-east corner",
      "reasoning": "Maximum natural light exposure based on sun path analysis"
    }
  ]
}

Implementation Instructions
Set up the basic FastAPI project structure
Implement environmental data service integrations first
Create the agent architecture using LangGraph
Build validators with scientific formulas
Implement design generation and output modules
Add knowledge bases (Vastu, building codes)
Create API endpoints
Add comprehensive error handling and logging
Write unit tests for critical components
Create documentation and usage examples
Please provide a complete, production-ready implementation with proper error handling, logging, and documentation.
