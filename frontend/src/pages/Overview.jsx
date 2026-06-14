// ThermalSense AI — Overview Page
// Shows city stats, model performance, and key findings at a glance.

import { useState, useEffect } from 'react'
import { getStats } from '../api'

export default function Overview() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getStats()
      .then(setStats)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="loading"><div className="spinner" /> Loading stats...</div>
  )

  return (
    <div>
      <div className="page-header">
        <h2>Kolkata Urban Heat Intelligence</h2>
        <p>Landsat 8 + Sentinel-2 + ERA5 · 100m resolution · 2019–2024</p>
      </div>

      {/* Key metrics */}
      <div className="cards-row">
        <div className="card">
          <div className="card-label">Mean LST 2024</div>
          <div className="card-value hot">43.8°C</div>
          <div className="card-sub">Pre-monsoon peak</div>
        </div>
        <div className="card">
          <div className="card-label">UHI Intensity</div>
          <div className="card-value warm">+8.2°C</div>
          <div className="card-sub">Urban vs rural delta</div>
        </div>
        <div className="card">
          <div className="card-label">ISA Coverage</div>
          <div className="card-value mid">57.8%</div>
          <div className="card-sub">Impervious surface area</div>
        </div>
        <div className="card">
          <div className="card-label">Pixels Analysed</div>
          <div className="card-value isro">
            {stats ? stats.data.n_pixels_cached.toLocaleString() : '—'}
          </div>
          <div className="card-sub">At 100m resolution</div>
        </div>
        <div className="card">
          <div className="card-label">OSM Buildings</div>
          <div className="card-value cool">282K</div>
          <div className="card-sub">Morphology features</div>
        </div>
        <div className="card">
          <div className="card-label">EKW Cooling</div>
          <div className="card-value veg">−1.46°C</div>
          <div className="card-sub">Wetland restoration</div>
        </div>
      </div>

      {/* Model performance */}
      <div className="section">
        <div className="section-title">
          Model Performance <span>XGBoost + Physics-Informed Neural Network</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Value</th>
              <th>Target</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {stats && [
              { metric: 'R² (test set)',      value: stats.performance.r2_test,         target: '> 0.75',  pass: stats.performance.r2_test > 0.75 },
              { metric: 'MAE (test set)',     value: `${stats.performance.mae_test_c}°C`, target: '< 2.0°C', pass: stats.performance.mae_test_c < 2.0 },
              { metric: 'RMSE (test set)',    value: `${stats.performance.rmse_test_c}°C`, target: '< 3.0°C', pass: stats.performance.rmse_test_c < 3.0 },
              { metric: 'CPCB validation MAE', value: `${stats.performance.cpcb_mae_c}°C`, target: '< 3.0°C', pass: stats.performance.cpcb_mae_c < 3.0 },
              { metric: 'Physics violations', value: `${stats.performance.physics_violation_pct}%`, target: '< 15%', pass: stats.performance.physics_violation_pct < 15 },
            ].map(row => (
              <tr key={row.metric}>
                <td>{row.metric}</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{row.value}</td>
                <td style={{ color: 'var(--text-sec)', fontFamily: 'var(--font-mono)' }}>{row.target}</td>
                <td>
                  <span className={`badge ${row.pass ? 'badge-veg' : 'badge-hot'}`}>
                    {row.pass ? 'PASS' : 'FAIL'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Data sources */}
      <div className="section">
        <div className="section-title">Data Sources</div>
        <table>
          <thead>
            <tr><th>Source</th><th>What we use</th><th>Resolution</th><th>Years</th></tr>
          </thead>
          <tbody>
            {[
              { src: 'Landsat 8 (USGS)', use: 'Land Surface Temperature (Band 10)', res: '100m', years: '2019–2024' },
              { src: 'Sentinel-2 (ESA)', use: 'NDVI, NDWI, NDBI, Albedo', res: '10m → 100m', years: '2019–2024' },
              { src: 'ERA5 (ECMWF via GEE)', use: 'Atmospheric temp, humidity, wind', res: '11km → 100m', years: '2019–2024' },
              { src: 'OSM / Geofabrik PBF', use: 'Buildings, roads, water bodies', res: 'Vector → 100m', years: '2024' },
              { src: 'JRC GHSL', use: 'Building height (default 5m)', res: '100m', years: '2018' },
              { src: 'CPCB', use: 'Ground truth air temperature', res: 'Station', years: '2024' },
            ].map(row => (
              <tr key={row.src}>
                <td style={{ fontWeight: 600 }}>{row.src}</td>
                <td style={{ color: 'var(--text-sec)' }}>{row.use}</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-isro)' }}>{row.res}</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-sec)' }}>{row.years}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Key finding */}
      <div className="section" style={{ borderColor: 'var(--accent-isro)', background: '#0d111f' }}>
        <div className="section-title" style={{ color: 'var(--accent-isro)' }}>
          🛰 Key Finding for ISRO
        </div>
        <p style={{ color: 'var(--text-sec)', lineHeight: 1.8, fontSize: 13 }}>
          East Kolkata Wetlands (EKW) restoration delivers the highest urban cooling return —
          <strong style={{ color: 'var(--accent-veg)' }}> −1.46°C mean city-wide</strong> at the
          lowest cost (₹0.8Cr/km²). ERA5 humidity is the single largest driver of LST variation
          (SHAP = 3.44°C), confirming that Kolkata's humid tropical climate requires
          wetland-based solutions rather than purely albedo-based interventions used in arid UHI studies.
          The PINN enforces energy-balance physics constraints, with only
          <strong style={{ color: 'var(--accent-isro)' }}> 8.7% physics violations</strong> on the test set.
        </p>
      </div>
    </div>
  )
}
