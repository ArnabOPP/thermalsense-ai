// ThermalSense AI — Heatmap Page
// Interactive map showing LST or other variables for Kolkata or Delhi.

import { useState, useEffect, useRef } from 'react'
import { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } from 'react-leaflet'
import { getHeatmap } from '../api'

// Color scale: blue (cool) → yellow → red (hot)
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

// City configs
const CITIES = {
  kolkata: { label: 'Kolkata', center: [22.55, 88.37], zoom: 12 },
  delhi:   { label: 'Delhi',   center: [28.65, 77.20], zoom: 11 },
}

const VARIABLES = [
  { id: 'lst',               label: 'LST (°C)',           unit: '°C' },
  { id: 'ndvi',              label: 'NDVI',               unit: '' },
  { id: 'ndbi',              label: 'NDBI (built-up)',    unit: '' },
  { id: 'shap_era5_humidity',label: 'SHAP: Humidity',    unit: '°C' },
  { id: 'shap_ndvi',         label: 'SHAP: NDVI cooling', unit: '°C' },
]

// Helper to re-center map when city changes
function MapController({ center, zoom }) {
  const map = useMap()
  useEffect(() => { map.setView(center, zoom) }, [center, zoom])
  return null
}

export default function Heatmap() {
  const [city, setCity]         = useState('kolkata')
  const [variable, setVariable] = useState('lst')
  const [data, setData]         = useState(null)
  const [loading, setLoading]   = useState(false)

  function loadData(v, c) {
    setLoading(true)
    setData(null)
    getHeatmap(v, c, 5000)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false))
  }

  // Reload whenever city or variable changes
  useEffect(() => { loadData(variable, city) }, [variable, city])

  const pixels = data?.pixels?.filter(p => p.value !== null) ?? []
  const vmin = data?.value_min ?? 30
  const vmax = data?.value_max ?? 55
  const cityConfig = CITIES[city]

  return (
    <div>
      <div className="page-header">
        <h2>Spatial Heatmap</h2>
        <p>100m resolution · 2024 pre-monsoon · {cityConfig.label}</p>
      </div>

      {/* City + Variable selectors */}
      <div className="section" style={{ padding: '12px 16px', marginBottom: 16 }}>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>

          {/* City dropdown */}
          <select
            value={city}
            onChange={e => setCity(e.target.value)}
            style={{
              background: 'var(--bg-deep)',
              color: 'var(--text-pri)',
              border: '1px solid var(--accent-isro)',
              borderRadius: 6,
              padding: '7px 14px',
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            <option value="kolkata">🏙 Kolkata</option>
            <option value="delhi">🏛 Delhi</option>
          </select>

          <div style={{ width: 1, height: 28, background: 'var(--border)' }} />

          {/* Variable buttons */}
          {VARIABLES.map(v => (
            <button
              key={v.id}
              className="btn"
              onClick={() => setVariable(v.id)}
              style={{
                background: variable === v.id ? 'var(--accent-isro)' : 'var(--border)',
                color: variable === v.id ? 'white' : 'var(--text-sec)',
                padding: '6px 12px',
                fontSize: 12,
              }}
            >
              {v.label}
            </button>
          ))}
        </div>
      </div>

      {/* Stats row */}
      {data && (
        <div className="cards-row" style={{ marginBottom: 16 }}>
          <div className="card">
            <div className="card-label">City</div>
            <div className="card-value isro" style={{ fontSize: 18 }}>{cityConfig.label}</div>
          </div>
          <div className="card">
            <div className="card-label">Min</div>
            <div className="card-value cool" style={{ fontSize: 20 }}>
              {data.value_min.toFixed(1)}{data.unit}
            </div>
          </div>
          <div className="card">
            <div className="card-label">Mean</div>
            <div className="card-value warm" style={{ fontSize: 20 }}>
              {data.value_mean.toFixed(1)}{data.unit}
            </div>
          </div>
          <div className="card">
            <div className="card-label">Max</div>
            <div className="card-value hot" style={{ fontSize: 20 }}>
              {data.value_max.toFixed(1)}{data.unit}
            </div>
          </div>
          <div className="card">
            <div className="card-label">Pixels</div>
            <div className="card-value" style={{ fontSize: 20 }}>
              {data.n_pixels.toLocaleString()}
            </div>
          </div>
        </div>
      )}

      {/* Map */}
      <div className="map-container">
        <MapContainer
          center={cityConfig.center}
          zoom={cityConfig.zoom}
          style={{ height: '100%', width: '100%', background: '#0a0e1a' }}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://carto.com">CARTO</a>'
          />

          {/* Re-center when city changes */}
          <MapController center={cityConfig.center} zoom={cityConfig.zoom} />

          {loading && (
            <div style={{
              position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)',
              background: 'rgba(10,14,26,0.9)', padding: '8px 16px',
              borderRadius: 8, zIndex: 1000, color: '#7b8fa8',
              fontFamily: 'var(--font-mono)', fontSize: 12,
            }}>
              Loading {cityConfig.label} {variable}...
            </div>
          )}

          {pixels.map((p, i) => (
            <CircleMarker
              key={i}
              center={[p.lat, p.lon]}
              radius={4}
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
                  <div><strong>{variable.toUpperCase()}:</strong> {p.value.toFixed(2)}{data?.unit}</div>
                </div>
              </Tooltip>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>

      {/* Legend */}
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
            {vmax.toFixed(1)}{data?.unit}
          </span>
        </div>
      </div>
    </div>
  )
}