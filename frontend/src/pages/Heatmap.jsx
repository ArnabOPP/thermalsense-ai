// ThermalSense AI — Heatmap with live city search
// Search any city → get OSM boundary → clip dots to that shape

import { useState, useEffect, useRef, useCallback } from 'react'
import { MapContainer, TileLayer, CircleMarker, Tooltip, GeoJSON, useMap } from 'react-leaflet'
import * as turf from '@turf/turf'
import { getHeatmap } from '../api'

// Color scale
function lstToColor(value, min, max) {
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)))
  if (t < 0.25) {
    const s = t / 0.25
    return `rgb(${Math.round(59 + s * 6)}, ${Math.round(130 + s * 76)}, ${Math.round(246 - s * 40)})`
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25
    return `rgb(${Math.round(65 + s * 179)}, ${Math.round(206 - s * 21)}, ${Math.round(206 - s * 177)})`
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25
    return `rgb(${Math.round(244)}, ${Math.round(185 - s * 88)}, ${Math.round(29 - s * 29)})`
  } else {
    const s = (t - 0.75) / 0.25
    return `rgb(${Math.round(244 - s * 5)}, ${Math.round(97 - s * 97)}, 0)`
  }
}

// Check if point is inside polygon using ray casting
function pointInPolygon(lat, lon, polygonCoords) {
  // polygonCoords is array of [lon, lat] pairs (GeoJSON format)
  let inside = false
  const x = lon, y = lat
  for (let i = 0, j = polygonCoords.length - 1; i < polygonCoords.length; j = i++) {
    const xi = polygonCoords[i][0], yi = polygonCoords[i][1]
    const xj = polygonCoords[j][0], yj = polygonCoords[j][1]
    const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
    if (intersect) inside = !inside
  }
  return inside
}

// Get flat ring coordinates from GeoJSON geometry (handles Polygon and MultiPolygon)
function getOuterRing(geometry) {
  if (geometry.type === 'Polygon') {
    return geometry.coordinates[0]
  } else if (geometry.type === 'MultiPolygon') {
    // Return the largest polygon ring
    let largest = []
    for (const poly of geometry.coordinates) {
      if (poly[0].length > largest.length) largest = poly[0]
    }
    return largest
  }
  return null
}

const AVAILABLE_CITIES = {
  kolkata: 'kolkata',
  calcutta: 'kolkata',
  delhi: 'delhi',
  'new delhi': 'delhi',
}

const VARIABLES = [
  { id: 'lst',               label: 'LST (°C)',           unit: '°C' },
  { id: 'ndvi',              label: 'NDVI',               unit: '' },
  { id: 'ndbi',              label: 'NDBI (built-up)',    unit: '' },
  { id: 'shap_era5_humidity',label: 'SHAP: Humidity',    unit: '°C' },
  { id: 'shap_ndvi',         label: 'SHAP: NDVI cooling', unit: '°C' },
]

function MapFlyTo({ center, zoom }) {
  const map = useMap()
  useEffect(() => {
    if (center) map.flyTo(center, zoom, { duration: 1.2 })
  }, [center, zoom])
  return null
}

export default function Heatmap() {
  const [searchQuery, setSearchQuery]     = useState('')
  const [suggestions, setSuggestions]     = useState([])
  const [searching, setSearching]         = useState(false)
  const [selectedCity, setSelectedCity]   = useState(null)   // { name, center, zoom, boundary, dataCity }
  const [variable, setVariable]           = useState('lst')
  const [rawData, setRawData]             = useState(null)
  const [filteredPixels, setFilteredPixels] = useState([])
  const [loading, setLoading]             = useState(false)
  const [noData, setNoData]               = useState(false)
  const searchTimeout                     = useRef(null)

  // Search Nominatim for city suggestions
  const searchCities = useCallback((q) => {
    if (!q || q.length < 2) { setSuggestions([]); return }
    clearTimeout(searchTimeout.current)
    searchTimeout.current = setTimeout(async () => {
      setSearching(true)
      try {
        const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(q)}&format=json&polygon_geojson=1&limit=5&featuretype=city`
        const r = await fetch(url, { headers: { 'Accept-Language': 'en' } })
        const data = await r.json()
        setSuggestions(data.filter(d => d.geojson && ['Polygon','MultiPolygon'].includes(d.geojson.type)))
      } catch(e) {
        console.error(e)
      } finally {
        setSearching(false)
      }
    }, 400)
  }, [])

  useEffect(() => { searchCities(searchQuery) }, [searchQuery])

  // When user picks a suggestion
  function selectSuggestion(item) {
    const bbox = item.boundingbox // [minlat, maxlat, minlon, maxlon]
    const center = [parseFloat(item.lat), parseFloat(item.lon)]
    const latSpan = parseFloat(bbox[1]) - parseFloat(bbox[0])
    const zoom = latSpan > 1 ? 9 : latSpan > 0.3 ? 11 : 12

    // Check if we have thermal data for this city
    const nameLower = item.display_name.toLowerCase()
    let dataCity = null
    for (const [key, val] of Object.entries(AVAILABLE_CITIES)) {
      if (nameLower.includes(key)) { dataCity = val; break }
    }

    setSelectedCity({
      name: item.display_name.split(',')[0],
      center,
      zoom,
      boundary: item.geojson,
      dataCity,
    })
    setSuggestions([])
    setSearchQuery(item.display_name.split(',')[0])
    setNoData(!dataCity)
  }

  // Load thermal data when city or variable changes
  useEffect(() => {
    if (!selectedCity?.dataCity) return
    setLoading(true)
    setRawData(null)
    setFilteredPixels([])
    getHeatmap(variable, selectedCity.dataCity, 15000)
      .then(data => {
        setRawData(data)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [selectedCity, variable])

  // Clip pixels to boundary when data or boundary changes
  useEffect(() => {
    if (!rawData?.pixels || !selectedCity?.boundary) {
      setFilteredPixels(rawData?.pixels?.filter(p => p.value !== null) ?? [])
      return
    }

    const ring = getOuterRing(selectedCity.boundary)
    if (!ring) {
      setFilteredPixels(rawData.pixels.filter(p => p.value !== null))
      return
    }

    const clipped = rawData.pixels.filter(p =>
      p.value !== null && pointInPolygon(p.lat, p.lon, ring)
    )
    setFilteredPixels(clipped)
  }, [rawData, selectedCity])

  const vmin = rawData?.value_min ?? 30
  const vmax = rawData?.value_max ?? 55
  const defaultCenter = [22.55, 88.37]

  return (
    <div>
      <div className="page-header">
        <h2>Spatial Heatmap</h2>
        <p>Search any city · Boundary from OpenStreetMap · Thermal data available for Kolkata & Delhi</p>
      </div>

      {/* Search bar */}
      <div className="section" style={{ padding: '12px 16px', marginBottom: 16 }}>
        <div style={{ position: 'relative', maxWidth: 400, marginBottom: 12 }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{ position: 'relative', flex: 1 }}>
              <input
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder="Search any city... (e.g. Mumbai, Tokyo, Berlin)"
                style={{
                  width: '100%',
                  background: 'var(--bg-deep)',
                  color: 'var(--text-pri)',
                  border: '1px solid var(--accent-isro)',
                  borderRadius: 8,
                  padding: '8px 14px',
                  fontSize: 14,
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
              />
              {searching && (
                <div style={{
                  position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)',
                  color: 'var(--text-sec)', fontSize: 12
                }}>searching...</div>
              )}
            </div>
          </div>

          {/* Suggestions dropdown */}
          {suggestions.length > 0 && (
            <div style={{
              position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 2000,
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              borderRadius: 8, marginTop: 4, overflow: 'hidden', boxShadow: '0 8px 24px rgba(0,0,0,0.4)'
            }}>
              {suggestions.map((s, i) => {
                const nameLower = s.display_name.toLowerCase()
                const hasData = Object.keys(AVAILABLE_CITIES).some(k => nameLower.includes(k))
                return (
                  <div
                    key={i}
                    onClick={() => selectSuggestion(s)}
                    style={{
                      padding: '10px 14px', cursor: 'pointer', borderBottom: '1px solid var(--border)',
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-deep)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  >
                    <div>
                      <div style={{ fontSize: 13, color: 'var(--text-pri)', fontWeight: 500 }}>
                        {s.display_name.split(',').slice(0,2).join(',')}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-sec)', marginTop: 2 }}>
                        {s.display_name.split(',').slice(2).join(',').trim()}
                      </div>
                    </div>
                    <div style={{
                      fontSize: 10, padding: '2px 8px', borderRadius: 4, fontWeight: 600,
                      background: hasData ? 'rgba(34,197,94,0.15)' : 'rgba(100,116,139,0.15)',
                      color: hasData ? '#22c55e' : 'var(--text-sec)',
                    }}>
                      {hasData ? '🌡 Data available' : 'Boundary only'}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Variable buttons */}
        {selectedCity?.dataCity && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            {VARIABLES.map(v => (
              <button
                key={v.id}
                className="btn"
                onClick={() => setVariable(v.id)}
                style={{
                  background: variable === v.id ? 'var(--accent-isro)' : 'var(--border)',
                  color: variable === v.id ? 'white' : 'var(--text-sec)',
                  padding: '6px 12px', fontSize: 12,
                }}
              >
                {v.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Stats */}
      {rawData && selectedCity?.dataCity && (
        <div className="cards-row" style={{ marginBottom: 16 }}>
          <div className="card">
            <div className="card-label">City</div>
            <div className="card-value isro" style={{ fontSize: 16 }}>{selectedCity.name}</div>
          </div>
          <div className="card">
            <div className="card-label">Min</div>
            <div className="card-value cool" style={{ fontSize: 20 }}>{rawData.value_min.toFixed(1)}{rawData.unit}</div>
          </div>
          <div className="card">
            <div className="card-label">Mean</div>
            <div className="card-value warm" style={{ fontSize: 20 }}>{rawData.value_mean.toFixed(1)}{rawData.unit}</div>
          </div>
          <div className="card">
            <div className="card-label">Max</div>
            <div className="card-value hot" style={{ fontSize: 20 }}>{rawData.value_max.toFixed(1)}{rawData.unit}</div>
          </div>
          <div className="card">
            <div className="card-label">Pixels shown</div>
            <div className="card-value" style={{ fontSize: 20 }}>{filteredPixels.length.toLocaleString()}</div>
          </div>
        </div>
      )}

      {/* No data message */}
      {noData && selectedCity && (
        <div className="section" style={{ padding: '16px', marginBottom: 16, borderColor: 'var(--accent-isro)' }}>
          <div style={{ color: 'var(--text-sec)', fontSize: 13 }}>
            📍 Showing boundary for <strong style={{ color: 'var(--text-pri)' }}>{selectedCity.name}</strong>.
            Thermal data is currently available for <strong style={{ color: '#f97316' }}>Kolkata</strong> and <strong style={{ color: '#f97316' }}>Delhi</strong>.
            Search one of those cities to see LST data.
          </div>
        </div>
      )}

      {/* Map */}
      <div className="map-container">
        <MapContainer
          center={defaultCenter}
          zoom={12}
          style={{ height: '100%', width: '100%', background: '#0a0e1a' }}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://carto.com">CARTO</a>'
          />

          {selectedCity && <MapFlyTo center={selectedCity.center} zoom={selectedCity.zoom} />}

          {/* City boundary outline */}
          {selectedCity?.boundary && (
            <GeoJSON
              key={selectedCity.name}
              data={selectedCity.boundary}
              style={{
                color: '#6366f1',
                weight: 2,
                fillColor: 'transparent',
                fillOpacity: 0,
              }}
            />
          )}

          {/* Loading indicator */}
          {loading && (
            <div style={{
              position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)',
              background: 'rgba(10,14,26,0.9)', padding: '8px 16px',
              borderRadius: 8, zIndex: 1000, color: '#7b8fa8',
              fontFamily: 'var(--font-mono)', fontSize: 12,
            }}>
              Loading thermal data...
            </div>
          )}

          {/* Thermal dots */}
          {filteredPixels.map((p, i) => (
            <CircleMarker
              key={i}
              center={[p.lat, p.lon]}
              radius={5}
              pathOptions={{
                fillColor: lstToColor(p.value, vmin, vmax),
                fillOpacity: 0.85,
                color: 'transparent',
                weight: 0,
              }}
            >
              <Tooltip>
                <div style={{ fontFamily: 'monospace', fontSize: 12 }}>
                  <div>{p.lat.toFixed(4)}°N, {p.lon.toFixed(4)}°E</div>
                  <div><strong>{variable.toUpperCase()}:</strong> {p.value.toFixed(2)}{rawData?.unit}</div>
                </div>
              </Tooltip>
            </CircleMarker>
          ))}

          {/* Default prompt */}
          {!selectedCity && (
            <div style={{
              position: 'absolute', top: '50%', left: '50%',
              transform: 'translate(-50%, -50%)',
              background: 'rgba(10,14,26,0.85)', padding: '20px 32px',
              borderRadius: 12, zIndex: 1000, textAlign: 'center',
              border: '1px solid var(--border)',
            }}>
              <div style={{ fontSize: 28, marginBottom: 8 }}>🔍</div>
              <div style={{ color: 'var(--text-pri)', fontSize: 14, fontWeight: 600 }}>Search any city above</div>
              <div style={{ color: 'var(--text-sec)', fontSize: 12, marginTop: 4 }}>
                Thermal data available for Kolkata & Delhi
              </div>
            </div>
          )}
        </MapContainer>
      </div>

      {/* Legend */}
      {rawData && (
        <div className="section" style={{ marginTop: 16, padding: '12px 16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 11, color: 'var(--text-sec)', fontFamily: 'var(--font-mono)' }}>
              {vmin.toFixed(1)}
            </span>
            <div style={{
              flex: 1, height: 12, borderRadius: 6,
              background: 'linear-gradient(to right, #3b82f6, #06b6d4, #eab308, #f97316, #ef4444)',
            }} />
            <span style={{ fontSize: 11, color: 'var(--text-sec)', fontFamily: 'var(--font-mono)' }}>
              {vmax.toFixed(1)}{rawData?.unit}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}