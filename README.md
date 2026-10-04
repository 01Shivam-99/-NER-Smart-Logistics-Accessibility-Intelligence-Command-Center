# NER Smart Logistics Accessibility Intelligence Platform

## Upgraded GIS Command Center

This version turns the prototype into a map-first logistics and accessibility command center for India's North Eastern Region (NER).

### What the upgraded map shows

- Road network context through OpenStreetMap basemap
- High / Medium / Low risk corridors as colored route lines
- Major reference rivers and water systems
- Traffic hotspots and traffic state by corridor
- Vehicle GPS positions and cargo status
- Natural-disaster risk zones using rainfall + terrain + elevation + river-proximity factors
- Geo-tagged field incidents
- Logistics / airport / service nodes
- Optional satellite and light-map basemaps
- Selected route from OSRM shown directly on the map
- Layer controls so a user can switch information on/off

### Live-data behavior

- **Weather:** Open-Meteo current weather is queried automatically and cached for 5 minutes.
- **GPS:** Set `GPS_API_URL` to a REST endpoint to replace demo vehicle positions.
- **Traffic:** Set `TOMTOM_API_KEY` to enable TomTom live traffic flow for corridor points.
- If an external service is not configured or unavailable, the app falls back to clearly labelled demo/estimated data rather than pretending it is live.

### Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Optional environment variables (PowerShell)

```powershell
$env:TOMTOM_API_KEY="YOUR_KEY"
$env:GPS_API_URL="https://your-server.example/api/vehicles"
streamlit run app.py
```

### GPS endpoint shape

The endpoint can return either a JSON array or an object such as:

```json
{
  "vehicles": [
    {
      "id": "MED-001",
      "type": "Medicine Van",
      "commodity": "Essential Medicines",
      "origin": "Guwahati",
      "dest": "Itanagar",
      "lat": 26.80,
      "lon": 93.50,
      "status": "In Transit",
      "eta_hrs": 4.2
    }
  ]
}
```

### Important prototype note

The river lines, corridor geometry, facility points and hazard model inputs in this repository are reference/demo data. For a real operational deployment, replace these with authoritative GIS datasets and official warning feeds.
