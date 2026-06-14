// ThermalSense AI — API calls
// All backend communication goes through here.
// Person D: never call fetch() directly in components — use these functions.

const BASE = 'https://thermalsense-ai-production.up.railway.app'  // proxied to http://localhost:8000 via vite.config.js

// Generic fetch helper
async function get(path) {
  const res = await fetch(BASE + path)
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`)
  return res.json()
}

async function post(path, body) {
  const res = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`)
  return res.json()
}

// ─── Exported API functions ───────────────────────────────────────────────────

/** Health check + model status */
export const getHealth  = () => get('/')

/** Full model + data stats */
export const getStats   = () => get('/stats')

/**
 * Get spatial heatmap pixels.
 * variable: "lst" | "ndvi" | "ndbi" | "isa_pct" | "shap_era5_humidity" etc.
 */
export const getHeatmap = (variable = 'lst', city = 'kolkata', maxPixels = 15000) =>
  get(`/heatmap?variable=${variable}&city=${city}&max_pixels=${maxPixels}`)

/** List available scenarios */
export const getScenarios = () => get('/scenarios')

/** Run a cooling intervention scenario */
export const runScenario = (scenario, year = 2024, season = 'pre_monsoon') =>
  post('/scenarios', { scenario, year, season })

/** Pre-computed scenario summary */
export const getScenarioSummary = () => get('/scenarios/summary')

/** SHAP explainability values */
export const getShap = (maxPixels = 3000) =>
  get(`/shap?max_pixels=${maxPixels}`)

/** Predict LST for a single pixel (for interactive tool) */
export const predictLST = (features) => post('/predict', features)
export const getCities = () => get('/cities')
export const getGlobalHeatmap = (lat, lon, name, radius = 0.15) =>
  get(`/heatmap/global?lat=${lat}&lon=${lon}&name=${encodeURIComponent(name)}&radius=${radius}`)