// ThermalSense AI — Global Heatmap with multi-source selector
// MODIS (1km) · Landsat 8 (100m) · Sentinel-2 (10m)

import { useState, useEffect, useRef, useCallback } from 'react'
import { MapContainer, TileLayer, CircleMarker, Tooltip, GeoJSON, useMap } from 'react-leaflet'
import { getGlobalHeatmap } from '../api'

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

function pointInPolygon(lat, lon, ring) {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1]
    const xj = ring[j][0], yj = ring[j][1]
    if (((yi > lat) !== (yj > lat)) && (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi))
      inside = !inside
  }
  return inside
}

function getOuterRing(geometry) {
  if (!geometry) return null
  if (geometry.type === 'Polygon') return geometry.coordinates[0]
  if (geometry.type === 'MultiPolygon') {
    let largest = []
    for (const poly of geometry.coordinates)
      if (poly[0].length > largest.length) largest = poly[0]
    return largest
  }
  return null
}

function MapFlyTo({ center, zoom }) {
  const map = useMap()
  useEffect(() => { if (center) map.flyTo(center, zoom, { duration: 1.2 }) }, [center, zoom])
  return null
}

const SOURCES = [
  {
    id: 'modis',
    label: 'MODIS',
    sublabel: '1km · Global · Fast',
    icon: '🛰',
    color: '#6366f1',
    description: 'Terra satellite · Daily global coverage · Best for large areas & states',
  },
  {
    id: 'landsat',
    label: 'Landsat 8',
    sublabel: '100m · Cities · ~15s',
    icon: '🌍',
    color: '#f97316',
    description: 'USGS Landsat 8 Band 10 · Thermal infrared · Best for cities',
  },
  {
    id: 'sentinel',
    label: 'Sentinel-2',
    sublabel: '10m · Cities · ~30s',
    icon: '🔬',
    color: '#22c55e',
    description: 'ESA Sentinel-2 · Highest resolution · Best for urban detail',
  },
]

export default function Heatmap() {
  const [searchQuery, setSearchQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [searching, setSearching]     = useState(false)
  const [selectedCity, setSelectedCity] = useState(null)
  const [source, setSource]           = useState('modis')
  const [rawData, setRawData]         = useState(null)
  const [filteredPixels, setFilteredPixels] = useState([])
  const [loading, setLoading]         = useState(false)
  const searchTimeout                 = useRef(null)

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
      } catch(e) { console.error(e) }
      finally { setSearching(false) }
    }, 400)
  }, [])

  useEffect(() => { searchCities(searchQuery) }, [searchQuery])

  function selectSuggestion(item) {
    const bbox = item.boundingbox
    const latSpan = parseFloat(bbox[1]) - parseFloat(bbox[0])
    const lonSpan = parseFloat(bbox[3]) - parseFloat(bbox[2])
    const center = [parseFloat(item.lat), parseFloat(item.lon)]
    const zoom = latSpan > 1 ? 9 : latSpan > 0.3 ? 11 : 12
    const radius = Math.max(Math.max(latSpan, lonSpan) / 2 * 1.2, 0.3)
    setSelectedCity({
      name: item.display_name.split(',')[0],
      center, zoom,
      boundary: item.geojson,
      radius,
    })
    setSuggestions([])
    setSearchQuery(item.display_name.split(',')[0])
  }

  // Fetch when city or source changes
  useEffect(() => {
    if (!selectedCity) return
    setLoading(true)
    setRawData(null)
    setFilteredPixels([])
    // Pass source to API
    const API_BASE = import.meta.env.VITE_API_URL || 'https://thermalsense-ai-production.up.railway.app'
    fetch(`${API_BASE}/heatmap/global?lat=${selectedCity.center[0]}&lon=${selectedCity.center[1]}&name=${encodeURIComponent(selectedCity.name)}&radius=${selectedCity.radius}&source=${source}`)
      .then(r => r.json())
      .then(data => setRawData(data))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [selectedCity, source])

  // Clip to boundary
  useEffect(() => {
    if (!rawData?.pixels) return
    const ring = selectedCity?.boundary ? getOuterRing(selectedCity.boundary) : null
    if (!ring) {
      setFilteredPixels(rawData.pixels.filter(p => p.value !== null))
      return
    }
    setFilteredPixels(
      rawData.pixels.filter(p => p.value !== null && pointInPolygon(p.lat, p.lon, ring))
    )
  }, [rawData, selectedCity])

  const vmin = rawData?.value_min ?? 30
  const vmax = rawData?.value_max ?? 55
  const activeSource = SOURCES.find(s => s.id === source)

  return (
    <div>
      <div className="page-header">
        <h2>Spatial Heatmap</h2>
        <p>Search any city worldwide · Real satellite LST · Boundary from OpenStreetMap</p>
      </div>

      {/* Search + Source selector */}
      <div className="section" style={{ padding: '12px 16px', marginBottom: 16 }}>

        {/* Search bar */}
        <div style={{ position: 'relative', maxWidth: 500, marginBottom: 12 }}>
          <input
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="🔍  Search any city — Kolkata, Tokyo, London, Mumbai..."
            style={{
              width: '100%', boxSizing: 'border-box',
              background: 'var(--bg-deep)', color: 'var(--text-pri)',
              border: '1px solid var(--accent-isro)', borderRadius: 8,
              padding: '10px 16px', fontSize: 14, outline: 'none',
            }}
          />
          {searching && (
            <div style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)',
              color: 'var(--text-sec)', fontSize: 11 }}>searching...</div>
          )}
          {suggestions.length > 0 && (
            <div style={{
              position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 2000,
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              borderRadius: 8, marginTop: 4, boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
              overflow: 'hidden',
            }}>
              {suggestions.map((s, i) => (
                <div key={i} onClick={() => selectSuggestion(s)}
                  style={{ padding: '10px 14px', cursor: 'pointer',
                    borderBottom: i < suggestions.length-1 ? '1px solid var(--border)' : 'none' }}
                  onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-deep)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                >
                  <div style={{ fontSize: 13, color: 'var(--text-pri)', fontWeight: 500 }}>
                    {s.display_name.split(',').slice(0,2).join(',')}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-sec)', marginTop: 2 }}>
                    {s.display_name.split(',').slice(2).join(',').trim()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Source selector */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {SOURCES.map(src => (
            <button
              key={src.id}
              onClick={() => setSource(src.id)}
              title={src.description}
              style={{
                display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
                padding: '8px 14px', borderRadius: 8, cursor: 'pointer',
                border: `1px solid ${source === src.id ? src.color : 'var(--border)'}`,
                background: source === src.id ? `${src.color}20` : 'var(--bg-deep)',
                transition: 'all 0.15s',
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 600,
                color: source === src.id ? src.color : 'var(--text-sec)' }}>
                {src.icon} {src.label}
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-sec)', marginTop: 2 }}>
                {src.sublabel}
              </div>
            </button>
          ))}
        </div>

        {/* Source description */}
        {activeSource && (
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-sec)' }}>
            {activeSource.description}
          </div>
        )}
      </div>

      {/* Stats */}
      {rawData && (
        <div className="cards-row" style={{ marginBottom: 16 }}>
          <div className="card">
            <div className="card-label">City</div>
            <div className="card-value isro" style={{ fontSize: 14 }}>{selectedCity?.name}</div>
          </div>
          <div className="card">
            <div className="card-label">Source</div>
            <div className="card-value" style={{ fontSize: 10, color: 'var(--text-sec)' }}>{rawData.source}</div>
          </div>
          <div className="card">
            <div className="card-label">Min</div>
            <div className="card-value cool" style={{ fontSize: 20 }}>{rawData.value_min.toFixed(1)}°C</div>
          </div>
          <div className="card">
            <div className="card-label">Mean</div>
            <div className="card-value warm" style={{ fontSize: 20 }}>{rawData.value_mean.toFixed(1)}°C</div>
          </div>
          <div className="card">
            <div className="card-label">Max</div>
            <div className="card-value hot" style={{ fontSize: 20 }}>{rawData.value_max.toFixed(1)}°C</div>
          </div>
          <div className="card">
            <div className="card-label">Pixels</div>
            <div className="card-value" style={{ fontSize: 20 }}>{filteredPixels.length.toLocaleString()}</div>
          </div>
        </div>
      )}

      {/* Map */}
      <div className="map-container">
        <MapContainer center={[22.55, 88.37]} zoom={4}
          style={{ height: '100%', width: '100%', background: '#0a0e1a' }}>
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://carto.com">CARTO</a>'
          />
          {selectedCity && <MapFlyTo center={selectedCity.center} zoom={selectedCity.zoom} />}
          {selectedCity?.boundary && (
            <GeoJSON key={selectedCity.name} data={selectedCity.boundary}
              style={{ color: activeSource?.color || '#6366f1', weight: 2, fillOpacity: 0 }} />
          )}
          {loading && (
            <div style={{
              position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)',
              background: 'rgba(10,14,26,0.9)', padding: '8px 20px',
              borderRadius: 8, zIndex: 1000, color: '#7b8fa8',
              fontFamily: 'var(--font-mono)', fontSize: 12,
            }}>
              Fetching {activeSource?.label} data for {selectedCity?.name}...
            </div>
          )}
          {filteredPixels.map((p, i) => (
            <CircleMarker key={i} center={[p.lat, p.lon]} radius={4}
              pathOptions={{ fillColor: lstToColor(p.value, vmin, vmax),
                fillOpacity: 0.85, color: 'transparent', weight: 0 }}>
              <Tooltip>
                <div style={{ fontFamily: 'monospace', fontSize: 12 }}>
                  <div>{p.lat.toFixed(4)}°N, {p.lon.toFixed(4)}°E</div>
                  <div><strong>LST:</strong> {p.value.toFixed(1)}°C</div>
                  <div style={{ color: '#888', fontSize: 10 }}>{rawData?.source}</div>
                </div>
              </Tooltip>
            </CircleMarker>
          ))}
          {!selectedCity && (
            <div style={{
              position: 'absolute', top: '50%', left: '50%',
              transform: 'translate(-50%, -50%)',
              background: 'rgba(10,14,26,0.85)', padding: '24px 40px',
              borderRadius: 12, zIndex: 1000, textAlign: 'center',
              border: '1px solid var(--border)',
            }}>
              <div style={{ fontSize: 36, marginBottom: 8 }}>🌍</div>
              <div style={{ color: 'var(--text-pri)', fontSize: 15, fontWeight: 600 }}>Search any city above</div>
              <div style={{ color: 'var(--text-sec)', fontSize: 12, marginTop: 6 }}>
                MODIS · Landsat 8 · Sentinel-2 · Any city on Earth
              </div>
            </div>
          )}
        </MapContainer>
      </div>

      {rawData && (
        <div className="section" style={{ marginTop: 16, padding: '12px 16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 11, color: 'var(--text-sec)', fontFamily: 'var(--font-mono)' }}>
              {vmin.toFixed(1)}°C
            </span>
            <div style={{ flex: 1, height: 12, borderRadius: 6,
              background: 'linear-gradient(to right, #3b82f6, #06b6d4, #eab308, #f97316, #ef4444)' }} />
            <span style={{ fontSize: 11, color: 'var(--text-sec)', fontFamily: 'var(--font-mono)' }}>
              {vmax.toFixed(1)}°C
            </span>
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-sec)' }}>
            {rawData.source} · 2024 pre-monsoon composite · Boundary: OpenStreetMap
          </div>
        </div>
      )}
    </div>
  )
}