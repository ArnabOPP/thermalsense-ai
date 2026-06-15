// ThermalSense AI — Spatial Heatmap
// Sidebar layout + single-date + compare mode (side-by-side, fade, swipe) + fixed swipe

import { useState, useEffect, useRef, useCallback } from 'react'
import { MapContainer, TileLayer, CircleMarker, Tooltip, GeoJSON, useMap } from 'react-leaflet'

// ─── helpers ─────────────────────────────────────────────────────────────────

function lstToColor(value, min, max) {
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)))
  if (t < 0.25) { const s = t/0.25; return `rgb(${Math.round(59+s*6)},${Math.round(130+s*76)},${Math.round(246-s*40)})` }
  if (t < 0.50) { const s = (t-0.25)/0.25; return `rgb(${Math.round(65+s*179)},${Math.round(206-s*21)},${Math.round(206-s*177)})` }
  if (t < 0.75) { const s = (t-0.5)/0.25; return `rgb(244,${Math.round(185-s*88)},${Math.round(29-s*29)})` }
  const s = (t-0.75)/0.25; return `rgb(${Math.round(244-s*5)},${Math.round(97-s*97)},0)`
}

function pointInPolygon(lat, lon, ring) {
  let inside = false
  for (let i=0, j=ring.length-1; i<ring.length; j=i++) {
    const xi=ring[i][0],yi=ring[i][1],xj=ring[j][0],yj=ring[j][1]
    if (((yi>lat)!==(yj>lat))&&(lon<(xj-xi)*(lat-yi)/(yj-yi)+xi)) inside=!inside
  }
  return inside
}

function getOuterRing(geometry) {
  if (!geometry) return null
  if (geometry.type==='Polygon') return geometry.coordinates[0]
  if (geometry.type==='MultiPolygon') {
    let largest=[]
    for (const poly of geometry.coordinates) if (poly[0].length>largest.length) largest=poly[0]
    return largest
  }
  return null
}

function maxDate() {
  const d = new Date()
  d.setDate(d.getDate() - 3)
  return d.toISOString().split('T')[0]
}

function MapFlyTo({ center, zoom }) {
  const map = useMap()
  useEffect(() => { if (center) map.flyTo(center, zoom, { duration: 1.2 }) }, [center, zoom])
  return null
}

const SOURCES = [
  { id:'modis',    label:'MODIS',      sublabel:'1km · Global · Fast',   icon:'🛰', color:'#6366f1', minDate:'2000-02-24' },
  { id:'landsat',  label:'Landsat 8',  sublabel:'100m · Cities · ~15s',  icon:'🌍', color:'#f97316', minDate:'2013-04-11' },
  { id:'sentinel', label:'Sentinel-2', sublabel:'10m · Cities · ~30s',   icon:'🔬', color:'#22c55e', minDate:'2015-06-23' },
]

const API_BASE = import.meta.env.VITE_API_URL || 'https://thermalsense-ai-production.up.railway.app'

// ─── CitySearchSlot ───────────────────────────────────────────────────────────

function CitySearchSlot({ color, onCitySelected }) {
  const [query, setQuery]       = useState('')
  const [suggestions, setSugg]  = useState([])
  const [searching, setSearch]  = useState(false)
  const timeout = useRef(null)

  const search = useCallback((q) => {
    if (!q || q.length < 2) { setSugg([]); return }
    clearTimeout(timeout.current)
    timeout.current = setTimeout(async () => {
      setSearch(true)
      try {
        const r = await fetch(
          `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(q)}&format=json&polygon_geojson=1&limit=5&featuretype=city`,
          { headers:{ 'Accept-Language':'en' } }
        )
        const data = await r.json()
        setSugg(data.filter(d => d.geojson && ['Polygon','MultiPolygon'].includes(d.geojson.type)))
      } catch {}
      finally { setSearch(false) }
    }, 400)
  }, [])

  useEffect(() => { search(query) }, [query])

  function select(item) {
    const bbox = item.boundingbox
    const latSpan = parseFloat(bbox[1])-parseFloat(bbox[0])
    const lonSpan = parseFloat(bbox[3])-parseFloat(bbox[2])
    const center  = [parseFloat(item.lat), parseFloat(item.lon)]
    const zoom    = latSpan>1?9:latSpan>0.3?11:12
    const radius  = Math.max(Math.max(latSpan,lonSpan)/2*1.2, 0.5)
    const name    = item.display_name.split(',')[0]
    setQuery(name); setSugg([])
    onCitySelected({ name, center, zoom, boundary:item.geojson, radius, color })
  }

  return (
    <div style={{ position:'relative' }}>
      <div style={{ position:'relative', display:'flex', alignItems:'center', gap:6 }}>
        <div style={{ width:8, height:8, borderRadius:'50%', background:color, flexShrink:0 }} />
        <div style={{ position:'relative', flex:1 }}>
          <input value={query} onChange={e=>setQuery(e.target.value)}
            placeholder="Search city…"
            style={{
              width:'100%', boxSizing:'border-box',
              background:'var(--bg-deep)', color:'var(--text-pri)',
              border:`1px solid ${color}55`, borderRadius:8,
              padding:'7px 10px', fontSize:12, outline:'none',
            }} />
          {searching && <div style={{ position:'absolute',right:8,top:'50%',transform:'translateY(-50%)',color:'var(--text-sec)',fontSize:10 }}>…</div>}
          {suggestions.length>0 && (
            <div style={{
              position:'absolute',top:'100%',left:0,right:0,zIndex:3000,
              background:'var(--bg-card)',border:'1px solid var(--border)',
              borderRadius:8,marginTop:4,boxShadow:'0 8px 24px rgba(0,0,0,0.5)',overflow:'hidden',
            }}>
              {suggestions.map((s,i) => (
                <div key={i} onClick={()=>select(s)}
                  style={{ padding:'8px 12px',cursor:'pointer',fontSize:12,
                    borderBottom:i<suggestions.length-1?'1px solid var(--border)':'none',
                    color:'var(--text-pri)' }}
                  onMouseEnter={e=>e.currentTarget.style.background='var(--bg-deep)'}
                  onMouseLeave={e=>e.currentTarget.style.background='transparent'}>
                  <div style={{ fontWeight:500 }}>{s.display_name.split(',').slice(0,2).join(',')}</div>
                  <div style={{ fontSize:10,color:'var(--text-sec)',marginTop:2 }}>{s.display_name.split(',').slice(2,4).join(',').trim()}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── SwipeMap (fixed) ─────────────────────────────────────────────────────────

function SwipeMap({ cityA, pixelsA, colorA, cityB, pixelsB, colorB, vmin, vmax, flyTarget }) {
  const [dividerPct, setDividerPct] = useState(50)
  const containerRef = useRef(null)
  const isDragging   = useRef(false)
  const mapARef      = useRef(null)
  const mapBRef      = useRef(null)

  // sync map views
  function syncMaps(sourceMap, targetMap) {
    if (!sourceMap || !targetMap) return
    sourceMap.on('move', () => {
      targetMap.setView(sourceMap.getCenter(), sourceMap.getZoom(), { animate:false })
    })
  }

  function onMouseDown(e) { e.preventDefault(); isDragging.current = true }

  useEffect(() => {
    function onMove(e) {
      if (!isDragging.current || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const clientX = e.touches ? e.touches[0].clientX : e.clientX
      const pct = ((clientX - rect.left) / rect.width) * 100
      setDividerPct(Math.max(5, Math.min(95, pct)))
    }
    function onUp() { isDragging.current = false }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    window.addEventListener('touchmove', onMove)
    window.addEventListener('touchend', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      window.removeEventListener('touchmove', onMove)
      window.removeEventListener('touchend', onUp)
    }
  }, [])

  const center = cityA?.center || [22.55, 88.37]
  const zoom   = cityA?.zoom   || 9

  return (
    <div ref={containerRef} style={{ position:'relative', height:'100%', overflow:'hidden', userSelect:'none' }}>
      {/* Map A — full width, clipped to left of divider */}
      <div style={{ position:'absolute', inset:0, clipPath:`inset(0 ${100-dividerPct}% 0 0)` }}>
        <MapContainer center={center} zoom={zoom} style={{ height:'100%', width:'100%', background:'#0a0e1a' }}
          whenCreated={map => { mapARef.current = map }}>
          <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution='&copy; CARTO' />
          {flyTarget && <MapFlyTo center={flyTarget.center} zoom={flyTarget.zoom} />}
          {cityA?.boundary && <GeoJSON data={cityA.boundary} style={{ color:colorA, weight:2, fillOpacity:0 }} />}
          {(pixelsA||[]).map((p,i) => (
            <CircleMarker key={i} center={[p.lat,p.lon]} radius={6}
              pathOptions={{ fillColor:lstToColor(p.value,vmin,vmax), fillOpacity:0.88, color:'transparent', weight:0 }}>
              <Tooltip><div style={{ fontFamily:'monospace',fontSize:11 }}><b style={{color:colorA}}>{cityA?.name}</b><br/>{p.lat.toFixed(4)}°N {p.lon.toFixed(4)}°E<br/><b>LST:</b> {p.value.toFixed(1)}°C</div></Tooltip>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>

      {/* Map B — full width, clipped to right of divider */}
      <div style={{ position:'absolute', inset:0, clipPath:`inset(0 0 0 ${dividerPct}%)` }}>
        <MapContainer center={center} zoom={zoom} style={{ height:'100%', width:'100%', background:'#0a0e1a' }}
          whenCreated={map => { mapBRef.current = map }}>
          <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution='&copy; CARTO' />
          {flyTarget && <MapFlyTo center={flyTarget.center} zoom={flyTarget.zoom} />}
          {cityB?.boundary && <GeoJSON data={cityB.boundary} style={{ color:colorB, weight:2, fillOpacity:0 }} />}
          {(pixelsB||[]).map((p,i) => (
            <CircleMarker key={i} center={[p.lat,p.lon]} radius={6}
              pathOptions={{ fillColor:lstToColor(p.value,vmin,vmax), fillOpacity:0.88, color:'transparent', weight:0 }}>
              <Tooltip><div style={{ fontFamily:'monospace',fontSize:11 }}><b style={{color:colorB}}>{cityB?.name}</b><br/>{p.lat.toFixed(4)}°N {p.lon.toFixed(4)}°E<br/><b>LST:</b> {p.value.toFixed(1)}°C</div></Tooltip>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>

      {/* Divider line */}
      <div
        onMouseDown={onMouseDown} onTouchStart={onMouseDown}
        style={{
          position:'absolute', top:0, bottom:0,
          left:`calc(${dividerPct}% - 1px)`,
          width:3, background:'white', cursor:'ew-resize', zIndex:2000,
          display:'flex', alignItems:'center', justifyContent:'center',
        }}>
        <div style={{
          width:32, height:32, borderRadius:'50%', background:'white',
          display:'flex', alignItems:'center', justifyContent:'center',
          fontSize:14, color:'#0a0e1a', fontWeight:800,
          boxShadow:'0 2px 12px rgba(0,0,0,0.6)',
        }}>↔</div>
      </div>

      {/* Labels */}
      <div style={{ position:'absolute',top:10,left:10,zIndex:1500,
        background:'rgba(10,14,26,0.85)',border:`1px solid ${colorA}`,
        borderRadius:8,padding:'5px 12px',fontSize:11,fontWeight:700,color:colorA }}>
        A
      </div>
      <div style={{ position:'absolute',top:10,right:10,zIndex:1500,
        background:'rgba(10,14,26,0.85)',border:`1px solid ${colorB}`,
        borderRadius:8,padding:'5px 12px',fontSize:11,fontWeight:700,color:colorB }}>
        B
      </div>
    </div>
  )
}

// ─── Main Heatmap ─────────────────────────────────────────────────────────────

export default function Heatmap() {
  // Primary
  const [city, setCity]             = useState(null)
  const [source, setSource]         = useState('modis')
  const [date, setDate]             = useState(maxDate)
  const [rawData, setRawData]       = useState(null)
  const [pixels, setPixels]         = useState([])
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState('')
  const [clipBdy, setClipBdy]       = useState(true)
  const [flyTarget, setFlyTarget]   = useState(null)

  // Compare
  const [compare, setCompare]       = useState(false)
  const [cmpMode, setCmpMode]       = useState('sidebyside')  // sidebyside | fade | swipe
  const [cmpDate, setCmpDate]       = useState('')
  const [cmpSrc, setCmpSrc]         = useState('modis')
  const [cmpRaw, setCmpRaw]         = useState(null)
  const [cmpPixels, setCmpPixels]   = useState([])
  const [cmpLoading, setCmpLoading] = useState(false)
  const [cmpError, setCmpError]     = useState('')
  const [fadeOpacity, setFadeOp]    = useState(50)

  const srcObj    = SOURCES.find(s => s.id === source)
  const cmpSrcObj = SOURCES.find(s => s.id === cmpSrc)
  const MAX_DATE  = maxDate()

  // ── data fetching ───────────────────────────────────────────────────────────

  function clip(data, c, doClip, setter) {
    if (!data) return
    const ring = c?.boundary ? getOuterRing(c.boundary) : null
    setter(
      (doClip && ring
        ? data.pixels.filter(p => p.value !== null && pointInPolygon(p.lat, p.lon, ring))
        : data.pixels.filter(p => p.value !== null))
    )
  }

  async function fetchPrimary(c, s, d) {
    if (!c) return
    setLoading(true); setError(''); setRawData(null); setPixels([])
    try {
      const r = await fetch(`${API_BASE}/heatmap/global?lat=${c.center[0]}&lon=${c.center[1]}&name=${encodeURIComponent(c.name)}&radius=${c.radius}&source=${s}&date_start=${d}&date_end=${d}`)
      const data = await r.json()
      if (data.detail) { setError(data.detail); return }
      setRawData(data); clip(data, c, clipBdy, setPixels)
    } catch { setError('Network error') }
    finally { setLoading(false) }
  }

  async function fetchCompare(c, s, d) {
    if (!c || !d) return
    setCmpLoading(true); setCmpError(''); setCmpRaw(null); setCmpPixels([])
    try {
      const r = await fetch(`${API_BASE}/heatmap/global?lat=${c.center[0]}&lon=${c.center[1]}&name=${encodeURIComponent(c.name)}&radius=${c.radius}&source=${s}&date_start=${d}&date_end=${d}`)
      const data = await r.json()
      if (data.detail) { setCmpError(data.detail); return }
      setCmpRaw(data); clip(data, c, clipBdy, setCmpPixels)
    } catch { setCmpError('Network error') }
    finally { setCmpLoading(false) }
  }

  useEffect(() => { clip(rawData, city, clipBdy, setPixels); clip(cmpRaw, city, clipBdy, setCmpPixels) }, [clipBdy])
  useEffect(() => { if (city) fetchPrimary(city, source, date) }, [source, date])
  useEffect(() => { if (city && compare && cmpDate) fetchCompare(city, cmpSrc, cmpDate) }, [cmpSrc, cmpDate])

  useEffect(() => {
    if (compare && !cmpDate) {
      const d = new Date(date); d.setDate(d.getDate() - 30)
      const s = d.toISOString().split('T')[0]
      setCmpDate(s < (cmpSrcObj?.minDate||'2000-02-24') ? cmpSrcObj?.minDate : s)
    }
  }, [compare])

  function onCitySelected(c) {
    setCity(c); setFlyTarget({ center:c.center, zoom:c.zoom })
    fetchPrimary(c, source, date)
    if (compare && cmpDate) fetchCompare(c, cmpSrc, cmpDate)
  }

  const allValues = [...(rawData?.pixels?.map(p=>p.value)||[]), ...(cmpRaw?.pixels?.map(p=>p.value)||[])]
  const vmin = allValues.length ? Math.min(...allValues) : 20
  const vmax = allValues.length ? Math.max(...allValues) : 50
  const anyLoading = loading || cmpLoading

  // ── render ──────────────────────────────────────────────────────────────────

  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100%' }}>

      {/* page header */}
      <div className="page-header">
        <h2>Spatial Heatmap</h2>
        <p>Real satellite LST · Search any city · Compare across dates or satellites</p>
      </div>

      <div style={{ display:'flex', flex:1, gap:16, minHeight:0 }}>

        {/* ── SIDEBAR ───────────────────────────────────────────────────────── */}
        <div style={{
          width:240, flexShrink:0,
          background:'var(--bg-card)', border:'1px solid var(--border)',
          borderRadius:12, padding:16,
          display:'flex', flexDirection:'column', gap:20,
          overflowY:'auto',
        }}>

          {/* Location */}
          <div>
            <div style={{ fontSize:11, fontWeight:700, color:'var(--text-sec)',
              textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:8 }}>
              Location
            </div>
            <CitySearchSlot color="#6366f1" onCitySelected={onCitySelected} />
            {city && (
              <div style={{ marginTop:8, fontSize:11, color:'var(--text-sec)' }}>
                <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                  <div style={{ width:6, height:6, borderRadius:'50%', background:'#6366f1' }} />
                  <span style={{ color:'var(--text-pri)', fontWeight:600 }}>{city.name}</span>
                </div>
                {rawData && !loading && (
                  <div style={{ marginTop:4, paddingLeft:12 }}>
                    <span style={{ color:'#fb923c' }}>{rawData.value_mean.toFixed(1)}°C avg</span>
                    {' · '}
                    <span>{pixels.length.toLocaleString()} px</span>
                  </div>
                )}
                {loading && <div style={{ paddingLeft:12, color:'#6366f1', marginTop:4 }}>Loading…</div>}
                {error && <div style={{ paddingLeft:12, color:'#f87171', marginTop:4, fontSize:10 }}>{error}</div>}
              </div>
            )}
          </div>

          {/* Satellite */}
          <div>
            <div style={{ fontSize:11, fontWeight:700, color:'var(--text-sec)',
              textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:8 }}>
              Select Satellite
            </div>
            <div style={{ display:'flex', flexDirection:'column', gap:6 }}>
              {SOURCES.map(src => (
                <button key={src.id} onClick={() => setSource(src.id)}
                  style={{
                    display:'flex', alignItems:'center', gap:8,
                    padding:'8px 10px', borderRadius:8, cursor:'pointer', textAlign:'left',
                    border:`1px solid ${source===src.id ? src.color : 'var(--border)'}`,
                    background:source===src.id ? `${src.color}20` : 'var(--bg-deep)',
                    transition:'all 0.15s', width:'100%',
                  }}>
                  <span style={{ fontSize:14 }}>{src.icon}</span>
                  <div>
                    <div style={{ fontSize:12, fontWeight:600, color:source===src.id?src.color:'var(--text-pri)' }}>{src.label}</div>
                    <div style={{ fontSize:10, color:'var(--text-sec)' }}>{src.sublabel}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Date (single, capped at today-3) */}
          <div>
            <div style={{ fontSize:11, fontWeight:700, color:'var(--text-sec)',
              textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:8 }}>
              Date
            </div>
            <input type="date" value={date}
              min={srcObj?.minDate} max={MAX_DATE}
              onChange={e => setDate(e.target.value)}
              style={{
                width:'100%', boxSizing:'border-box', background:'var(--bg-deep)',
                color:'var(--text-pri)', border:'1px solid var(--border)', borderRadius:8,
                padding:'6px 10px', fontSize:11, outline:'none', colorScheme:'dark',
              }} />
            <div style={{ fontSize:9, color:'var(--text-sec)', marginTop:4 }}>
              📡 Latest available: {MAX_DATE} (3-day lag)
            </div>

            {/* Quick presets */}
            <div style={{ display:'flex', flexWrap:'wrap', gap:4, marginTop:8 }}>
              {[
                { label:'Yesterday', offset:4 },
                { label:'-7 days',   offset:10 },
                { label:'-30 days',  offset:33 },
                { label:'-1 year',   offset:368 },
              ].map(p => {
                const d = new Date(); d.setDate(d.getDate()-p.offset)
                const val = d.toISOString().split('T')[0]
                const active = date === val
                return (
                  <button key={p.label} onClick={() => setDate(val)}
                    style={{
                      fontSize:9, padding:'3px 7px', borderRadius:6, cursor:'pointer',
                      border:`1px solid ${active?'#6366f1':'var(--border)'}`,
                      background:active?'#6366f120':'var(--bg-deep)',
                      color:active?'#6366f1':'var(--text-sec)',
                    }}>{p.label}</button>
                )
              })}
            </div>
          </div>

          {/* Clip boundary */}
          <div>
            <label style={{ display:'flex', alignItems:'flex-start', gap:10, cursor:'pointer' }}>
              <input type="checkbox" checked={clipBdy} onChange={e=>setClipBdy(e.target.checked)}
                style={{ marginTop:2, accentColor:'#6366f1', width:14, height:14, cursor:'pointer' }} />
              <span style={{ fontSize:12, color:'var(--text-sec)', lineHeight:1.4 }}>
                Clip to boundary
              </span>
            </label>
          </div>

          {/* Compare toggle */}
          <div>
            <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:compare?12:0 }}>
              <div style={{ fontSize:11, fontWeight:700, color:'var(--text-sec)',
                textTransform:'uppercase', letterSpacing:'0.08em' }}>
                Compare Mode
              </div>
              {/* Toggle switch */}
              <div onClick={() => setCompare(c => !c)} style={{ cursor:'pointer' }}>
                <div style={{
                  width:36, height:20, borderRadius:10, position:'relative',
                  background:compare?'#6366f1':'var(--border)', transition:'background 0.2s',
                }}>
                  <div style={{
                    position:'absolute', top:3, left:compare?18:3,
                    width:14, height:14, borderRadius:'50%', background:'white',
                    transition:'left 0.2s',
                  }} />
                </div>
              </div>
            </div>

            {compare && (
              <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
                {/* Compare satellite */}
                <div style={{ fontSize:10, color:'var(--text-sec)', marginBottom:2 }}>Satellite B</div>
                <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
                  {SOURCES.map(s => (
                    <button key={s.id} onClick={() => setCmpSrc(s.id)}
                      style={{
                        display:'flex', alignItems:'center', gap:6,
                        padding:'6px 8px', borderRadius:8, cursor:'pointer', textAlign:'left',
                        border:`1px solid ${cmpSrc===s.id ? s.color : 'var(--border)'}`,
                        background:cmpSrc===s.id?`${s.color}20`:'var(--bg-deep)',
                        width:'100%',
                      }}>
                      <span style={{ fontSize:12 }}>{s.icon}</span>
                      <div style={{ fontSize:11, fontWeight:600, color:cmpSrc===s.id?s.color:'var(--text-pri)' }}>{s.label}</div>
                    </button>
                  ))}
                </div>

                {/* Compare date */}
                <div style={{ fontSize:10, color:'var(--text-sec)' }}>Date B</div>
                <input type="date" value={cmpDate}
                  min={cmpSrcObj?.minDate} max={MAX_DATE}
                  onChange={e => setCmpDate(e.target.value)}
                  style={{
                    width:'100%', boxSizing:'border-box', background:'var(--bg-deep)',
                    color:'var(--text-pri)', border:'1px solid #f97316', borderRadius:8,
                    padding:'6px 10px', fontSize:11, outline:'none', colorScheme:'dark',
                  }} />

                {/* View mode */}
                <div style={{ fontSize:10, color:'var(--text-sec)' }}>View mode</div>
                <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
                  {[
                    { id:'sidebyside', label:'Side by side', icon:'⬛⬜' },
                    { id:'fade',       label:'Fade / Opacity', icon:'◑' },
                    { id:'swipe',      label:'Swipe compare', icon:'↔' },
                  ].map(m => (
                    <button key={m.id} onClick={() => setCmpMode(m.id)}
                      style={{
                        padding:'6px 10px', borderRadius:8, cursor:'pointer', textAlign:'left',
                        border:`1px solid ${cmpMode===m.id?'#f97316':'var(--border)'}`,
                        background:cmpMode===m.id?'#f9731620':'var(--bg-deep)',
                        color:cmpMode===m.id?'#f97316':'var(--text-sec)',
                        fontSize:11, fontWeight:cmpMode===m.id?700:400,
                        display:'flex', alignItems:'center', gap:8,
                      }}>
                      <span>{m.icon}</span> {m.label}
                    </button>
                  ))}
                </div>

                {/* Fade slider */}
                {cmpMode==='fade' && (
                  <div>
                    <div style={{ display:'flex', justifyContent:'space-between', fontSize:9, color:'var(--text-sec)', marginBottom:4 }}>
                      <span>A ({date})</span><span>B ({cmpDate})</span>
                    </div>
                    <input type="range" min={0} max={100} value={fadeOpacity}
                      onChange={e => setFadeOp(+e.target.value)}
                      style={{ width:'100%', accentColor:'#f97316' }} />
                    <div style={{ fontSize:9, color:'var(--text-sec)', textAlign:'center', marginTop:2 }}>
                      A: {100-fadeOpacity}% · B: {fadeOpacity}%
                    </div>
                  </div>
                )}

                {cmpError && <div style={{ fontSize:10, color:'#f87171' }}>{cmpError}</div>}
                {cmpLoading && <div style={{ fontSize:10, color:'#f97316' }}>⏳ Fetching B…</div>}

                {/* Compare stats */}
                {cmpRaw && !cmpLoading && (
                  <div style={{ background:'var(--bg-deep)', borderRadius:8, padding:10 }}>
                    <div style={{ fontSize:10, color:'#f97316', fontWeight:700, marginBottom:6 }}>B: {cmpDate}</div>
                    {[['Min', cmpRaw.value_min.toFixed(1)+'°C','#38bdf8'],
                      ['Mean',cmpRaw.value_mean.toFixed(1)+'°C','#fb923c'],
                      ['Max', cmpRaw.value_max.toFixed(1)+'°C','#f87171'],
                    ].map(([l,v,c])=>(
                      <div key={l} style={{ display:'flex', justifyContent:'space-between', fontSize:11, marginBottom:3 }}>
                        <span style={{ color:'var(--text-sec)' }}>{l}</span>
                        <span style={{ color:c, fontFamily:'var(--font-mono)', fontWeight:700 }}>{v}</span>
                      </div>
                    ))}
                    {rawData && (
                      <div style={{ display:'flex', justifyContent:'space-between', fontSize:11, marginTop:6,
                        paddingTop:6, borderTop:'1px solid var(--border)' }}>
                        <span style={{ color:'var(--text-sec)' }}>Δ Mean</span>
                        <span style={{
                          color: rawData.value_mean > cmpRaw.value_mean ? '#f87171' : '#4ade80',
                          fontFamily:'var(--font-mono)', fontWeight:700,
                        }}>
                          {rawData.value_mean > cmpRaw.value_mean ? '+' : ''}{(rawData.value_mean - cmpRaw.value_mean).toFixed(1)}°C
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Loading indicator */}
          {anyLoading && (
            <div style={{ fontSize:11, color:'#6366f1', fontFamily:'var(--font-mono)' }}>
              ⏳ Fetching {srcObj?.label}…
            </div>
          )}
        </div>

        {/* ── MAP AREA ──────────────────────────────────────────────────────── */}
        <div style={{ flex:1, display:'flex', flexDirection:'column', gap:12, minWidth:0 }}>

          {/* Map container */}
          <div style={{ flex:1, borderRadius:12, overflow:'hidden',
            border:'1px solid var(--border)', minHeight:400, position:'relative' }}>

            {/* SWIPE mode */}
            {compare && cmpMode==='swipe' ? (
              <SwipeMap
                cityA={city} pixelsA={pixels}   colorA={srcObj?.color||'#6366f1'}
                cityB={city} pixelsB={cmpPixels} colorB={cmpSrcObj?.color||'#f97316'}
                vmin={vmin} vmax={vmax} flyTarget={flyTarget}
              />
            ) : compare && cmpMode==='sidebyside' ? (
              /* SIDE BY SIDE mode */
              <div style={{ display:'flex', height:'100%' }}>
                <div style={{ flex:1, position:'relative', borderRight:'2px solid var(--border)' }}>
                  <div style={{ position:'absolute',top:8,left:8,zIndex:1000,
                    background:'rgba(10,14,26,0.85)',border:`1px solid ${srcObj?.color}`,
                    borderRadius:8,padding:'4px 10px',fontSize:10,fontWeight:700,color:srcObj?.color }}>
                    A · {date} · {srcObj?.label}
                  </div>
                  <MapContainer center={city?.center||[22.55,88.37]} zoom={city?.zoom||5}
                    style={{ height:'100%', width:'100%', background:'#0a0e1a' }}>
                    <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution='&copy; CARTO' />
                    {flyTarget && <MapFlyTo center={flyTarget.center} zoom={flyTarget.zoom} />}
                    {city?.boundary && <GeoJSON data={city.boundary} style={{ color:srcObj?.color, weight:2, fillOpacity:0 }} />}
                    {pixels.map((p,i) => (
                      <CircleMarker key={i} center={[p.lat,p.lon]} radius={6}
                        pathOptions={{ fillColor:lstToColor(p.value,vmin,vmax), fillOpacity:0.88, color:'transparent', weight:0 }} />
                    ))}
                  </MapContainer>
                </div>
                <div style={{ flex:1, position:'relative' }}>
                  <div style={{ position:'absolute',top:8,left:8,zIndex:1000,
                    background:'rgba(10,14,26,0.85)',border:`1px solid ${cmpSrcObj?.color}`,
                    borderRadius:8,padding:'4px 10px',fontSize:10,fontWeight:700,color:cmpSrcObj?.color }}>
                    B · {cmpDate||'—'} · {cmpSrcObj?.label}
                  </div>
                  <MapContainer center={city?.center||[22.55,88.37]} zoom={city?.zoom||5}
                    style={{ height:'100%', width:'100%', background:'#0a0e1a' }}>
                    <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution='&copy; CARTO' />
                    {flyTarget && <MapFlyTo center={flyTarget.center} zoom={flyTarget.zoom} />}
                    {city?.boundary && <GeoJSON data={city.boundary} style={{ color:cmpSrcObj?.color, weight:2, fillOpacity:0 }} />}
                    {cmpPixels.map((p,i) => (
                      <CircleMarker key={i} center={[p.lat,p.lon]} radius={6}
                        pathOptions={{ fillColor:lstToColor(p.value,vmin,vmax), fillOpacity:0.88, color:'transparent', weight:0 }} />
                    ))}
                  </MapContainer>
                </div>
              </div>
            ) : (
              /* SINGLE MAP (with optional fade overlay) */
              <MapContainer center={city?.center||[22.55,88.37]} zoom={city?.zoom||5}
                style={{ height:'100%', width:'100%', background:'#0a0e1a' }}>
                <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution='&copy; CARTO' />
                {flyTarget && <MapFlyTo center={flyTarget.center} zoom={flyTarget.zoom} />}

                {/* Primary boundary + pixels */}
                {city?.boundary && <GeoJSON data={city.boundary} style={{ color:srcObj?.color, weight:2, fillOpacity:0 }} />}
                {pixels.map((p,i) => (
                  <CircleMarker key={`a${i}`} center={[p.lat,p.lon]} radius={6}
                    pathOptions={{ fillColor:lstToColor(p.value,vmin,vmax),
                      fillOpacity: compare && cmpMode==='fade' ? (100-fadeOpacity)/100*0.88 : 0.88,
                      color:'transparent', weight:0 }}>
                    <Tooltip><div style={{ fontFamily:'monospace',fontSize:11 }}>
                      <div style={{ fontWeight:600, color:srcObj?.color }}>{city?.name} (A)</div>
                      <div>{p.lat.toFixed(4)}°N, {p.lon.toFixed(4)}°E</div>
                      <div><b>LST:</b> {p.value.toFixed(1)}°C</div>
                    </div></Tooltip>
                  </CircleMarker>
                ))}

                {/* Fade compare overlay */}
                {compare && cmpMode==='fade' && cmpPixels.map((p,i) => (
                  <CircleMarker key={`b${i}`} center={[p.lat,p.lon]} radius={6}
                    pathOptions={{ fillColor:lstToColor(p.value,vmin,vmax),
                      fillOpacity:fadeOpacity/100*0.88, color:'transparent', weight:0 }}>
                    <Tooltip><div style={{ fontFamily:'monospace',fontSize:11 }}>
                      <div style={{ fontWeight:600, color:cmpSrcObj?.color }}>{city?.name} (B)</div>
                      <div>{p.lat.toFixed(4)}°N, {p.lon.toFixed(4)}°E</div>
                      <div><b>LST:</b> {p.value.toFixed(1)}°C</div>
                    </div></Tooltip>
                  </CircleMarker>
                ))}

                {/* Empty state */}
                {!city && (
                  <div style={{
                    position:'absolute',top:'50%',left:'50%',transform:'translate(-50%,-50%)',
                    background:'rgba(10,14,26,0.85)',padding:'24px 40px',borderRadius:12,
                    zIndex:1000,textAlign:'center',border:'1px solid var(--border)',
                  }}>
                    <div style={{ fontSize:36, marginBottom:8 }}>🌍</div>
                    <div style={{ color:'var(--text-pri)',fontSize:15,fontWeight:600 }}>Search any city</div>
                    <div style={{ color:'var(--text-sec)',fontSize:12,marginTop:6 }}>
                      MODIS · Landsat 8 · Sentinel-2 real satellite data
                    </div>
                  </div>
                )}
              </MapContainer>
            )}
          </div>

          {/* Color scale footer */}
          {rawData && (
            <div style={{ display:'flex', alignItems:'center', gap:10,
              background:'var(--bg-card)', border:'1px solid var(--border)',
              borderRadius:8, padding:'8px 14px' }}>
              <span style={{ fontSize:11,color:'var(--text-sec)',fontFamily:'var(--font-mono)',whiteSpace:'nowrap' }}>
                {vmin.toFixed(1)}°C
              </span>
              <div style={{ flex:1, height:10, borderRadius:5,
                background:'linear-gradient(to right,#3b82f6,#06b6d4,#eab308,#f97316,#ef4444)' }} />
              <span style={{ fontSize:11,color:'var(--text-sec)',fontFamily:'var(--font-mono)',whiteSpace:'nowrap' }}>
                {vmax.toFixed(1)}°C
              </span>
              <span style={{ fontSize:10,color:'var(--text-sec)',marginLeft:8,whiteSpace:'nowrap' }}>
                {srcObj?.label} · {date}
                {compare && cmpRaw && ` vs ${cmpDate}`}
              </span>
            </div>
          )}

          {/* Stats cards */}
          {rawData && (
            <div style={{ display:'flex', gap:10 }}>
              {[
                { label:'City',   value:rawData.city,                     style:{ color:'#6366f1',fontSize:13 } },
                { label:'Source', value:rawData.source,                   style:{ fontSize:10,color:'var(--text-sec)' } },
                { label:'Date',   value:date,                             style:{ fontSize:11,color:'var(--text-sec)' } },
                { label:'Min',    value:`${rawData.value_min.toFixed(1)}°C`, style:{ color:'#38bdf8',fontSize:16 } },
                { label:'Mean',   value:`${rawData.value_mean.toFixed(1)}°C`,style:{ color:'#fb923c',fontSize:16 } },
                { label:'Max',    value:`${rawData.value_max.toFixed(1)}°C`, style:{ color:'#f87171',fontSize:16 } },
                { label:'Pixels', value:pixels.length.toLocaleString(),   style:{ fontSize:16 } },
              ].map((c,i) => (
                <div key={i} className="card" style={{ flex:1 }}>
                  <div className="card-label">{c.label}</div>
                  <div className="card-value" style={c.style}>{c.value}</div>
                </div>
              ))}
            </div>
          )}

        </div>
      </div>
    </div>
  )
}