// ThermalSense AI — Scenarios Page
// Click an intervention, see per-pixel cooling impact on the map.

import { useState, useEffect } from 'react'
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet'
import { getScenarioSummary, runScenario } from '../api'

const SCENARIOS = [
  {
    id: 'urban_greening',
    name: 'Urban Greening',
    desc: '+15% tree cover in low-vegetation zones',
    icon: '🌳',
    cost: '₹2.5Cr/km²',
  },
  {
    id: 'cool_roofs',
    name: 'Cool Roofs',
    desc: 'Albedo 0.12 → 0.65 in dense residential',
    icon: '🏠',
    cost: '₹1.8Cr/km²',
  },
  {
    id: 'ekw_restoration',
    name: 'EKW Restoration',
    desc: 'East Kolkata Wetlands — expand water cover',
    icon: '💧',
    cost: '₹0.8Cr/km²',
  },
  {
    id: 'green_corridors',
    name: 'Green Corridors',
    desc: 'Street trees along AJC Bose Rd + EM Bypass',
    icon: '🛣️',
    cost: '₹1.2Cr/km²',
  },
]

function deltaColor(delta) {
  if (delta < -3) return '#22c55e'
  if (delta < -1) return '#86efac'
  if (delta < -0.5) return '#bef264'
  if (delta < 0.5) return '#fde68a'
  return '#fca5a5'
}

export default function Scenarios() {
  const [selected, setSelected] = useState(null)
  const [summary, setSummary] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getScenarioSummary().then(setSummary).catch(console.error)
  }, [])

  function handleRun(scenarioId) {
    setSelected(scenarioId)
    setResult(null)
    setLoading(true)
    runScenario(scenarioId)
      .then(setResult)
      .catch(console.error)
      .finally(() => setLoading(false))
  }

  const summaryScenarios = summary?.scenarios ?? {}

  return (
    <div>
      <div className="page-header">
        <h2>Cooling Intervention Scenarios</h2>
        <p>Click a scenario to simulate city-wide cooling impact · 2024 baseline LST = 43.8°C</p>
      </div>

      {/* Scenario cards */}
      <div className="scenario-grid">
        {SCENARIOS.map(s => {
          const precomputed = summaryScenarios[s.id]
          return (
            <div
              key={s.id}
              className={`scenario-card ${selected === s.id ? 'selected' : ''}`}
              onClick={() => handleRun(s.id)}
            >
              <div style={{ fontSize: 24, marginBottom: 6 }}>{s.icon}</div>
              <div className="scenario-name">{s.name}</div>
              <div className="scenario-meta">{s.desc}</div>
              <div className="scenario-meta" style={{ marginTop: 4 }}>
                Cost: {s.cost}
              </div>
              {precomputed && (
                <div
                  className="scenario-delta"
                  style={{ color: precomputed.mean_delta_t_c < 0 ? 'var(--accent-veg)' : 'var(--accent-hot)' }}
                >
                  {precomputed.mean_delta_t_c > 0 ? '+' : ''}{precomputed.mean_delta_t_c.toFixed(2)}°C
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Results */}
      {loading && (
        <div className="loading"><div className="spinner" /> Running scenario...</div>
      )}

      {result && !loading && (
        <>
          {/* Summary stats */}
          <div className="cards-row">
            <div className="card">
              <div className="card-label">Mean ΔT</div>
              <div className="card-value" style={{ color: result.mean_delta_t_c < 0 ? 'var(--accent-veg)' : 'var(--accent-hot)', fontSize: 28 }}>
                {result.mean_delta_t_c > 0 ? '+' : ''}{result.mean_delta_t_c.toFixed(2)}°C
              </div>
              <div className="card-sub">City-wide average cooling</div>
            </div>
            <div className="card">
              <div className="card-label">Max Cooling</div>
              <div className="card-value veg" style={{ fontSize: 24 }}>
                {result.max_cooling_c.toFixed(2)}°C
              </div>
              <div className="card-sub">Hotspot reduction</div>
            </div>
            <div className="card">
              <div className="card-label">Pixels Cooled</div>
              <div className="card-value cool" style={{ fontSize: 24 }}>
                {result.pct_pixels_cooled.toFixed(1)}%
              </div>
              <div className="card-sub">Cooled by &gt;0.5°C</div>
            </div>
            <div className="card">
              <div className="card-label">Cost</div>
              <div className="card-value mid" style={{ fontSize: 24 }}>
                ₹{result.cost_cr_per_km2}Cr
              </div>
              <div className="card-sub">Per km²</div>
            </div>
          </div>

          {/* Map */}
          <div className="section">
            <div className="section-title">
              Spatial cooling map — {result.scenario_name}
              <span>green = cooling · red = warming</span>
            </div>
            <div className="map-container" style={{ height: 380 }}>
              <MapContainer
                center={[22.55, 88.37]}
                zoom={12}
                style={{ height: '100%', width: '100%' }}
              >
                <TileLayer
                  url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                  attribution='&copy; CARTO'
                />
                {result.pixels.map((p, i) => (
                  <CircleMarker
                    key={i}
                    center={[p.lat, p.lon]}
                    radius={4}
                    pathOptions={{
                      fillColor: deltaColor(p.delta_t),
                      fillOpacity: 0.85,
                      color: 'transparent',
                    }}
                  >
                    <Tooltip>
                      <div style={{ fontFamily: 'monospace', fontSize: 11 }}>
                        <div>Baseline: {p.baseline_lst.toFixed(1)}°C</div>
                        <div>Modified: {p.modified_lst.toFixed(1)}°C</div>
                        <div>ΔT: {p.delta_t > 0 ? '+' : ''}{p.delta_t.toFixed(2)}°C</div>
                      </div>
                    </Tooltip>
                  </CircleMarker>
                ))}
              </MapContainer>
            </div>

            {/* Legend */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 12 }}>
              <span style={{ fontSize: 11, color: 'var(--text-sec)' }}>Warming</span>
              <div style={{
                flex: 1, height: 10, borderRadius: 5,
                background: 'linear-gradient(to right, #fca5a5, #fde68a, #bef264, #86efac, #22c55e)',
              }} />
              <span style={{ fontSize: 11, color: 'var(--text-sec)' }}>Cooling</span>
            </div>
          </div>
        </>
      )}

      {/* EKW context box */}
      {!result && !loading && (
        <div className="section" style={{ borderColor: 'var(--accent-veg)' }}>
          <div className="section-title" style={{ color: 'var(--accent-veg)' }}>
            💡 Recommended: EKW Wetland Restoration
          </div>
          <p style={{ color: 'var(--text-sec)', fontSize: 13, lineHeight: 1.8 }}>
            The East Kolkata Wetlands (Ramsar site, 12,500 ha) deliver the highest cooling
            return of all four interventions — city-wide mean cooling of <strong style={{ color: 'var(--accent-veg)' }}>−1.46°C</strong> at
            the lowest cost of ₹0.8Cr/km². Expanding water cover in the eastern zone (lon &gt; 88.37°)
            reduces both ISA% and NDBI while increasing NDWI-driven evaporative cooling.
            Select <strong>EKW Restoration</strong> above to simulate the spatial impact.
          </p>
        </div>
      )}
    </div>
  )
}
