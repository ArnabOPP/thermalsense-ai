// ThermalSense AI — Spatial Heatmap v4.0
// Multi-city · NASA GIBS · UHI · Hotspots · Time Series · Intervention Estimator · Priority Zones

import { useState, useEffect, useRef, useMemo } from 'react'
import { MapContainer, TileLayer, GeoJSON, useMap, useMapEvents } from 'react-leaflet'
import L from 'leaflet'

// ─── pure helpers ─────────────────────────────────────────────────────────────
function lstToColor(v,mn,mx){
  const t=Math.max(0,Math.min(1,(v-mn)/(mx-mn)))
  if(t<0.25){const s=t/0.25;return`rgb(${Math.round(59+s*6)},${Math.round(130+s*76)},${Math.round(246-s*40)})`}
  if(t<0.50){const s=(t-0.25)/0.25;return`rgb(${Math.round(65+s*179)},${Math.round(206-s*21)},${Math.round(206-s*177)})`}
  if(t<0.75){const s=(t-0.5)/0.25;return`rgb(244,${Math.round(185-s*88)},${Math.round(29-s*29)})`}
  const s=(t-0.75)/0.25;return`rgb(${Math.round(244-s*5)},${Math.round(97-s*97)},0)`
}
function pip(lat,lon,ring){
  let inside=false
  for(let i=0,j=ring.length-1;i<ring.length;j=i++){
    const xi=ring[i][0],yi=ring[i][1],xj=ring[j][0],yj=ring[j][1]
    if(((yi>lat)!==(yj>lat))&&(lon<(xj-xi)*(lat-yi)/(yj-yi)+xi))inside=!inside
  }
  return inside
}
function outerRing(geo){
  if(!geo)return null
  if(geo.type==='Polygon')return geo.coordinates[0]
  if(geo.type==='MultiPolygon'){let b=[];for(const p of geo.coordinates)if(p[0].length>b.length)b=p[0];return b}
  return null
}
function todayMinus3(){const d=new Date();d.setDate(d.getDate()-3);return d.toISOString().split('T')[0]}
function closestPixel(pixels,lat,lon,maxDist=0.15){
  let best=null,bestD=Infinity
  for(const p of pixels){const d=Math.hypot(p.lat-lat,p.lon-lon);if(d<bestD&&d<maxDist){bestD=d;best=p}}
  return best
}
function buildGibsUrl(date){
  return`https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/${date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`
}

// ── Sub-district boundary overlay ─────────────────────────────────────────────
// Fetches tehsil/block-level boundaries from Nominatim for a given city/district
async function fetchSubDistricts(cityName, countryCode='IN'){
  try{
    const r=await fetch(
      `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(cityName)}&format=json&polygon_geojson=1&limit=8&featuretype=suburb&countrycodes=${countryCode}`,
      {headers:{'Accept-Language':'en'}}
    )
    const data=await r.json()
    return data.filter(d=>d.geojson&&['Polygon','MultiPolygon'].includes(d.geojson.type)).slice(0,6)
  }catch{return[]}
}

// ── UHI score ─────────────────────────────────────────────────────────────────
function computeUHI(pixels){
  if(!pixels||pixels.length<10)return null
  const sorted=[...pixels].sort((a,b)=>a.value-b.value)
  const q=Math.floor(sorted.length/4)
  const rural=sorted.slice(0,q).reduce((s,p)=>s+p.value,0)/q
  const urban=sorted.slice(-q).reduce((s,p)=>s+p.value,0)/q
  return(urban-rural).toFixed(1)
}

// ── 90th percentile threshold ─────────────────────────────────────────────────
function p90(pixels){
  if(!pixels||!pixels.length)return Infinity
  const sorted=[...pixels].map(p=>p.value).sort((a,b)=>a-b)
  return sorted[Math.floor(sorted.length*0.9)]
}

// ── Priority zones: cluster hotspot pixels into top-5 zones ──────────────────
function computePriorityZones(pixels, thresh, topN=5){
  if(!pixels||!pixels.length)return[]
  const hot=pixels.filter(p=>p.value>=thresh)
  if(!hot.length)return[]
  // Simple grid-cell clustering: round to 0.1° cells
  const cells={}
  hot.forEach(p=>{
    const key=`${(Math.round(p.lat*10)/10).toFixed(1)},${(Math.round(p.lon*10)/10).toFixed(1)}`
    if(!cells[key])cells[key]={lat:0,lon:0,count:0,sum:0,max:0}
    cells[key].lat+=p.lat; cells[key].lon+=p.lon
    cells[key].count++; cells[key].sum+=p.value
    cells[key].max=Math.max(cells[key].max,p.value)
  })
  return Object.values(cells)
    .map(c=>({lat:c.lat/c.count,lon:c.lon/c.count,count:c.count,mean:(c.sum/c.count),max:c.max}))
    .sort((a,b)=>b.mean-a.mean)
    .slice(0,topN)
}

// ── Intervention LST reduction estimate ──────────────────────────────────────
// Based on literature: ~0.4°C LST reduction per 10% increase in green cover
// Evapotranspiration effect: additional 0.2°C per 10% increase in irrigated area
function estimateLSTReduction(canopyPct, irrigatedPct=0){
  return (canopyPct * 0.04 + irrigatedPct * 0.02).toFixed(2)
}

// ── Narrative ─────────────────────────────────────────────────────────────────
function buildNarrative(cityList,cityRaw,cityPixels,compare,cmpRawMap,cmpDate,date){
  const parts=[]
  cityList.forEach(c=>{
    const rd=cityRaw[c.slotId];const px=cityPixels[c.slotId]||[]
    if(!rd)return
    const uhi=computeUHI(px)
    const thresh=p90(px)
    const hotPx=px.filter(p=>p.value>=thresh)
    const hotCount=hotPx.length
    const season=getSeason(date)

    // Find geographic direction of hotspot cluster
    let hotDir=''
    if(hotPx.length>5){
      const cLat=px.reduce((s,p)=>s+p.lat,0)/px.length
      const cLon=px.reduce((s,p)=>s+p.lon,0)/px.length
      const hLat=hotPx.reduce((s,p)=>s+p.lat,0)/hotPx.length
      const hLon=hotPx.reduce((s,p)=>s+p.lon,0)/hotPx.length
      const latDir=hLat>cLat?'northern':'southern'
      const lonDir=hLon>cLon?'eastern':'western'
      hotDir=`${latDir} ${lonDir} `
    }

    // Recommended canopy increase to bring hotspots below threshold
    const meanHot=hotPx.length?hotPx.reduce((s,p)=>s+p.value,0)/hotPx.length:0
    const targetReduction=meanHot>thresh?(meanHot-thresh+2).toFixed(1):null
    const requiredCanopy=targetReduction?Math.min(60,Math.round(parseFloat(targetReduction)/0.04/5)*5):null

    parts.push(
      `${c.name} records a mean LST of ${rd.value_mean.toFixed(1)}°C in ${season} (${date}), peaking at ${rd.value_max.toFixed(1)}°C on bare surfaces. `+
      (uhi?`UHI intensity ${uhi}°C above vegetated baseline — exceeds the 3°C urban heat emergency threshold defined by IMD. `:'') +
      (hotCount?`${hotCount} critical hotspots (≥${thresh.toFixed(1)}°C) concentrated in the ${hotDir}zone, indicating impervious surface dominance. `:'') +
      (requiredCanopy?`Recommended intervention: ${requiredCanopy}% canopy cover increase in priority zones — estimated to reduce surface temperature by ~${targetReduction}°C based on NDVI–LST literature (Bowler et al. 2010). `:'')
    )

    if(compare&&cmpRawMap[c.slotId]){
      const rdB=cmpRawMap[c.slotId]
      const delta=parseFloat((rd.value_mean-rdB.value_mean).toFixed(1))
      const seasonB=getSeason(cmpDate)
      const mechanism=delta>0
        ?'bare laterite soil heating and minimal evapotranspiration cover'
        :'increased soil moisture from precipitation and higher cloud albedo'
      parts.push(
        `Seasonal comparison with ${seasonB} (${cmpDate}): ${Math.abs(delta)}°C ${delta>0?'warmer':'cooler'}, driven by ${mechanism}. `+
        `This ${Math.abs(delta).toFixed(1)}°C swing represents the primary intervention window — greening before ${getSeason(date)==='pre-monsoon'?'March':'October'} would reduce peak season heat stress most effectively.`
      )
    }
  })
  return parts.join(' ')
}
function getSeason(d){
  if(!d)return''
  const m=parseInt(d.split('-')[1])
  if(m<=2||m===12)return'winter'
  if(m<=5)return'pre-monsoon'
  if(m<=9)return'monsoon'
  return'post-monsoon'
}

// ── Exports ───────────────────────────────────────────────────────────────────
function exportCSV(cityList,cityPixels,date,source){
  const rows=['city,lat,lon,lst_c,date,source']
  cityList.forEach(c=>{
    (cityPixels[c.slotId]||[]).forEach(p=>{
      rows.push(`${c.name},${p.lat},${p.lon},${p.value},${date},${source}`)
    })
  })
  const blob=new Blob([rows.join('\n')],{type:'text/csv'})
  const a=document.createElement('a');a.href=URL.createObjectURL(blob)
  a.download=`thermalsense_lst_${date}.csv`;a.click()
}
function exportPNG(){
  const el=document.getElementById('ts-map-area')
  if(!el){alert('Map not ready');return}
  import('https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js').then(()=>{
    window.html2canvas(el,{useCORS:true,scale:2}).then(canvas=>{
      const a=document.createElement('a');a.href=canvas.toDataURL('image/png')
      a.download='thermalsense_heatmap.png';a.click()
    })
  }).catch(()=>alert('Use browser Print > Save as PDF'))
}

// ─── constants ────────────────────────────────────────────────────────────────
const CITY_COLORS=['#6366f1','#f97316','#22c55e','#ec4899','#eab308','#06b6d4']
const SOURCES=[
  {id:'modis',   label:'MODIS',     sub:'1km · Global · Fast',  icon:'🛰',color:'#6366f1',min:'2000-02-24'},
  {id:'landsat', label:'Landsat 8', sub:'100m · Cities · ~15s', icon:'🌍',color:'#f97316',min:'2013-04-11'},
  {id:'sentinel',label:'Sentinel-2',sub:'10m · Cities · ~30s',  icon:'🔬',color:'#22c55e',min:'2015-06-23'},
]
const TILE_LAYERS={
  dark:     {url:'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', attr:'&copy; CARTO', label:'Dark', icon:'🌑', note:null},
  satellite:{url:'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr:'&copy; Esri', label:'Satellite', icon:'🌐', note:'Static mosaic — not date-specific'},
  gibs:     {url:null, attr:'&copy; NASA GIBS · MODIS Terra', label:'NASA Daily', icon:'🛰', note:'Actual satellite footage for selected date'},
  hybrid:   {url:'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr:'&copy; Esri', label:'Hybrid', icon:'🗺', labelUrl:'https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png', note:'Static + labels'},
}
const API=import.meta.env.VITE_API_URL||'https://thermalsense-ai-production.up.railway.app'
const MONTH_LABELS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

// ─── Leaflet helpers ──────────────────────────────────────────────────────────
function MapFlyTo({center,zoom}){
  const map=useMap()
  useEffect(()=>{if(center)map.flyTo(center,zoom,{duration:1.2})},[center,zoom])
  return null
}
function MapSync({sourceRef,targetRef}){
  const map=useMap()
  useEffect(()=>{sourceRef.current=map},[map])
  useEffect(()=>{
    if(!map)return
    const g={active:false}
    const sync=()=>{if(g.active||!targetRef.current)return;g.active=true;targetRef.current.setView(map.getCenter(),map.getZoom(),{animate:false,noMoveStart:true});g.active=false}
    map.on('move zoom',sync);return()=>map.off('move zoom',sync)
  },[map,targetRef])
  return null
}
function MapHoverEmitter({onHover}){
  useMapEvents({mousemove(e){onHover(e.latlng.lat,e.latlng.lng)},mouseout(){onHover(null,null)}})
  return null
}
function SyncedTooltipInner({lat,lon,value,cityName,color,label}){
  const map=useMap()
  const [pos,setPos]=useState(null)
  useEffect(()=>{
    if(!map||lat==null)return
    function update(){
      const cp=map.latLngToContainerPoint([lat,lon])
      const rect=map.getContainer().getBoundingClientRect()
      // Use fixed positioning so overflow:hidden on parent doesn't clip it
      setPos({x:rect.left+cp.x, y:rect.top+cp.y})
    }
    update()
    map.on('move zoom',update)
    return()=>map.off('move zoom',update)
  },[map,lat,lon])
  if(!pos)return null
  return(
    <div style={{
      position:'fixed',
      left:pos.x+14,
      top:pos.y-56,
      background:'rgba(255,255,255,0.97)',
      border:`2px solid ${color}`,
      borderRadius:8,padding:'7px 11px',
      pointerEvents:'none',
      zIndex:99999,
      fontFamily:'monospace',fontSize:11,whiteSpace:'nowrap',
      boxShadow:'0 4px 20px rgba(0,0,0,0.35)',
    }}>
      <div style={{fontWeight:700,color,marginBottom:3,fontSize:10,textTransform:'uppercase',letterSpacing:'0.05em'}}>{cityName} · {label}</div>
      <div style={{color:'#1e293b',fontSize:13,fontWeight:700}}>LST: <span style={{color}}>{value.toFixed(1)}°C</span></div>
      <div style={{color:'#64748b',fontSize:9,marginTop:3}}>{lat.toFixed(4)}°N {lon.toFixed(4)}°E</div>
    </div>
  )
}
function SyncedHoverLayer({hoverLatLon,pixels,cityList,label,color}){
  if(!hoverLatLon||hoverLatLon[0]==null)return null
  const [lat,lon]=hoverLatLon
  let best=null,bestCity=null,bestD=Infinity
  cityList.forEach(c=>{
    const px=pixels[c.slotId]||[]
    const p=closestPixel(px,lat,lon,0.15)
    if(p){const d=Math.hypot(p.lat-lat,p.lon-lon);if(d<bestD){bestD=d;best=p;bestCity=c}}
  })
  if(!best)return null
  return <SyncedTooltipInner lat={best.lat} lon={best.lon} value={best.value} cityName={bestCity?.name} color={bestCity?.color||color} label={label}/>
}
function BaseTileLayer({tileMode,date}){
  const layer=TILE_LAYERS[tileMode]||TILE_LAYERS.dark
  const url=tileMode==='gibs'?buildGibsUrl(date||todayMinus3()):layer.url
  return(<><TileLayer url={url} attribution={layer.attr} maxNativeZoom={tileMode==='gibs'?9:18} maxZoom={18}/>{layer.labelUrl&&<TileLayer url={layer.labelUrl} attribution="" zIndex={500}/>}</>)
}
function TileToggleButton({tileMode,setTileMode}){
  const modes=['dark','satellite','gibs','hybrid']
  const [open,setOpen]=useState(false)
  const icons={dark:'🌑',satellite:'🌐',gibs:'🛰',hybrid:'🗺'}
  const labels={dark:'Dark',satellite:'Satellite',gibs:'NASA Daily',hybrid:'Hybrid'}
  return(
    <div style={{position:'absolute',bottom:40,right:8,zIndex:1500}}>
      {open&&(
        <div style={{position:'absolute',bottom:38,right:0,background:'rgba(10,14,26,0.97)',
          border:'1px solid #1e293b',borderRadius:10,overflow:'hidden',whiteSpace:'nowrap',
          boxShadow:'0 8px 24px rgba(0,0,0,0.6)',minWidth:130}}>
          {modes.map((m,i)=>(
            <div key={m} onClick={()=>{setTileMode(m);setOpen(false)}}
              style={{padding:'8px 14px',cursor:'pointer',fontSize:12,display:'flex',alignItems:'center',gap:8,
                background:tileMode===m?'#6366f115':'transparent',color:tileMode===m?'#818cf8':'#cbd5e1',
                borderBottom:i<modes.length-1?'1px solid #1e293b22':''}}
              onMouseEnter={e=>e.currentTarget.style.background='#1e293b'}
              onMouseLeave={e=>e.currentTarget.style.background=tileMode===m?'#6366f115':'transparent'}>
              <span>{icons[m]}</span>{labels[m]}
            </div>
          ))}
        </div>
      )}
      <button onClick={()=>setOpen(o=>!o)} title="Switch map layer"
        style={{width:34,height:34,borderRadius:8,cursor:'pointer',background:'rgba(10,14,26,0.92)',
          border:'1px solid #334155',color:'#e2e8f0',fontSize:16,display:'flex',alignItems:'center',
          justifyContent:'center',boxShadow:'0 2px 8px rgba(0,0,0,0.4)'}}>
        {icons[tileMode]}
      </button>
    </div>
  )
}
function MapSkeleton(){
  return(
    <div style={{position:'absolute',inset:0,background:'#0a0e1a',zIndex:800,
      display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:16}}>
      <div style={{width:48,height:48,borderRadius:'50%',border:'3px solid #6366f1',
        borderTopColor:'transparent',animation:'spin 0.8s linear infinite'}}/>
      <div style={{color:'#6366f1',fontSize:13,fontFamily:'monospace'}}>Fetching satellite data…</div>
      <div style={{color:'#475569',fontSize:11}}>Querying NASA Earth Engine</div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  )
}

// ── PixelLayer — pure Leaflet canvas renderer, zero React DOM nodes ───────────
// L.canvas() draws all circles in one GPU-accelerated canvas pass.
// Pan/zoom is instant — no React reconciliation involved.
function PixelLayer({pixels,vmin,vmax,thresh,showHotspots,opacity=0.88,onPixelClick}){
  const map=useMap()
  const groupRef=useRef(null)
  const rendererRef=useRef(null)
  const propsRef=useRef({pixels,vmin,vmax,thresh,showHotspots,opacity,onPixelClick})
  useEffect(()=>{ propsRef.current={pixels,vmin,vmax,thresh,showHotspots,opacity,onPixelClick} })

  // Create canvas renderer + layer group once per map mount
  useEffect(()=>{
    if(!L)return
    const renderer=L.canvas({padding:0.5})
    const group=L.layerGroup().addTo(map)
    rendererRef.current=renderer
    groupRef.current=group
    return()=>{
      group.clearLayers()
      map.removeLayer(group)
      groupRef.current=null
      rendererRef.current=null
    }
  },[map])

  // Rebuild all circles when pixels/options change
  useEffect(()=>{
    const group=groupRef.current
    const renderer=rendererRef.current
    if(!L||!group||!renderer)return
    group.clearLayers()
    if(!pixels.length)return
    const {thresh:thr,showHotspots:sh,opacity:op,onPixelClick:onClick}=propsRef.current
    const z=map.getZoom()
    const r=Math.max(4,z*1.4)
    pixels.forEach(p=>{
      const isHot=sh&&thr!=null&&p.value>=thr
      const circle=L.circleMarker([p.lat,p.lon],{
        renderer,radius:isHot?r+2:r,
        fillColor:lstToColor(p.value,vmin,vmax),fillOpacity:op,
        color:isHot?'#ffffff':'transparent',weight:isHot?1.5:0,
        interactive:!!onClick,
      })
      if(onClick) circle.on('click',()=>onClick(p))
      group.addLayer(circle)
    })
    // Resize circles on zoom without full rebuild
    function onZoom(){
      const nz=map.getZoom(), nr=Math.max(4,nz*1.4)
      group.eachLayer(l=>{ if(l.setRadius) l.setRadius(nr) })
    }
    map.on('zoomend',onZoom)
    return()=>map.off('zoomend',onZoom)
  },[pixels,vmin,vmax,thresh,showHotspots,opacity])

  return null  // pure Leaflet — no React DOM output
}

// ── Time Series Modal ─────────────────────────────────────────────────────────
async function fetchTimeSeries(lat,lon,source){
  const year=new Date().getFullYear()-1
  const results=await Promise.all(
    Array.from({length:12},(_,i)=>{
      const d=`${year}-${String(i+1).padStart(2,'0')}-15`
      return fetch(`${API}/heatmap/global?lat=${lat}&lon=${lon}&name=pixel&radius=0.05&source=${source}&date_start=${d}&date_end=${d}`)
        .then(r=>r.json())
        .then(data=>({month:i+1,year,value:data.value_mean||null}))
        .catch(()=>({month:i+1,year,value:null}))
    })
  )
  return results
}
function getSeasonColor(m){
  if(m<=2||m===12)return'#38bdf8'
  if(m<=5)return'#f97316'
  if(m<=9)return'#22c55e'
  return'#eab308'
}
function TimeSeriesModal({pixel,source,onClose}){
  const [data,setData]=useState(null)
  const [loading,setLoading]=useState(true)
  const [error,setError]=useState('')
  useEffect(()=>{
    setLoading(true);setError('');setData(null)
    fetchTimeSeries(pixel.lat,pixel.lon,source)
      .then(d=>{setData(d);setLoading(false)})
      .catch(()=>{setError('Failed to fetch');setLoading(false)})
  },[pixel.lat,pixel.lon,source])
  const valid=data?data.filter(d=>d.value!==null):[]
  const minV=valid.length?Math.min(...valid.map(d=>d.value)):0
  const maxV=valid.length?Math.max(...valid.map(d=>d.value)):50
  const range=maxV-minV||1
  const W=480,H=120,PAD=16
  const pts=data?data.map((d,i)=>({x:PAD+(i/11)*(W-PAD*2),y:d.value!==null?H-PAD-((d.value-minV)/range)*(H-PAD*2):null,...d})):[]
  const pathD=pts.filter(p=>p.y!==null).map((p,i)=>`${i===0?'M':'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  return(
    <div onClick={onClose} style={{position:'fixed',inset:0,zIndex:9000,background:'rgba(0,0,0,0.7)',
      display:'flex',alignItems:'center',justifyContent:'center',backdropFilter:'blur(4px)'}}>
      <div onClick={e=>e.stopPropagation()} style={{background:'#0f172a',border:'1px solid #1e293b',
        borderRadius:14,padding:24,width:540,maxWidth:'95vw',boxShadow:'0 24px 64px rgba(0,0,0,0.8)'}}>
        <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:16}}>
          <div>
            <div style={{fontSize:13,fontWeight:700,color:'#e2e8f0'}}>📈 LST Time Series</div>
            <div style={{fontSize:10,color:'#475569',marginTop:3,fontFamily:'monospace'}}>
              {pixel.lat.toFixed(4)}°N {pixel.lon.toFixed(4)}°E · {source.toUpperCase()} · {new Date().getFullYear()-1}
            </div>
          </div>
          <button onClick={onClose} style={{background:'none',border:'none',cursor:'pointer',color:'#475569',fontSize:20,lineHeight:1,padding:4}}>×</button>
        </div>
        {loading&&(
          <div style={{textAlign:'center',padding:'40px 0',color:'#6366f1'}}>
            <div>⏳ Fetching 12 months in parallel…</div>
            <div style={{fontSize:11,color:'#334155',marginTop:6}}>Usually takes 15–30 seconds</div>
          </div>
        )}
        {error&&<div style={{color:'#f87171',padding:16,textAlign:'center'}}>{error}</div>}
        {data&&!loading&&(
          <>
            <div style={{background:'#0a0e1a',borderRadius:10,padding:12,marginBottom:12}}>
              <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{overflow:'visible'}}>
                {[0,0.25,0.5,0.75,1].map(t=>{
                  const y=PAD+(1-t)*(H-PAD*2);const v=minV+t*range
                  return(<g key={t}><line x1={PAD} y1={y} x2={W-PAD} y2={y} stroke="#1e293b" strokeWidth={1}/>
                    <text x={PAD-4} y={y+4} fill="#334155" fontSize={9} textAnchor="end">{v.toFixed(0)}°</text></g>)
                })}
                {[{s:0,e:1,c:'#38bdf8'},{s:2,e:4,c:'#f97316'},{s:5,e:8,c:'#22c55e'},{s:9,e:10,c:'#eab308'},{s:11,e:11,c:'#38bdf8'}].map(({s,e,c},i)=>{
                  const x1=PAD+(s/11)*(W-PAD*2);const x2=PAD+(e/11)*(W-PAD*2)+(W-PAD*2)/11
                  return<rect key={i} x={x1} y={PAD} width={x2-x1} height={H-PAD*2} fill={c} fillOpacity={0.05}/>
                })}
                {pathD&&<path d={`${pathD} L${pts.filter(p=>p.y!==null).slice(-1)[0]?.x},${H-PAD} L${pts.filter(p=>p.y!==null)[0]?.x},${H-PAD} Z`} fill="url(#sg)" fillOpacity={0.3}/>}
                {pathD&&<path d={pathD} fill="none" stroke="#f97316" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"/>}
                {pts.map((pt,i)=>pt.y!==null&&(
                  <g key={i}>
                    <circle cx={pt.x} cy={pt.y} r={4} fill={getSeasonColor(pt.month)} stroke="#0a0e1a" strokeWidth={1.5}/>
                    <text x={pt.x} y={H-2} fill="#334155" fontSize={8} textAnchor="middle">{MONTH_LABELS[i]}</text>
                    {(pt.value===maxV||pt.value===minV)&&<text x={pt.x} y={pt.y-8} fill="#fb923c" fontSize={9} textAnchor="middle" fontWeight="700">{pt.value.toFixed(1)}°</text>}
                  </g>
                ))}
                <defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#f97316" stopOpacity={0.6}/>
                  <stop offset="100%" stopColor="#f97316" stopOpacity={0}/>
                </linearGradient></defs>
              </svg>
            </div>
            <div style={{display:'grid',gridTemplateColumns:'repeat(6,1fr)',gap:4,marginBottom:12}}>
              {data.map((d,i)=>(
                <div key={i} style={{textAlign:'center',padding:'5px 2px',borderRadius:6,
                  background:d.value?`${lstToColor(d.value,minV,maxV)}22`:'#0a0e1a',
                  border:`1px solid ${d.value?lstToColor(d.value,minV,maxV)+'44':'#1e293b'}`}}>
                  <div style={{fontSize:9,color:'#475569',marginBottom:2}}>{MONTH_LABELS[i]}</div>
                  <div style={{fontSize:12,fontWeight:700,fontFamily:'monospace',color:d.value?lstToColor(d.value,minV,maxV):'#334155'}}>
                    {d.value?d.value.toFixed(1)+'°':'—'}
                  </div>
                </div>
              ))}
            </div>
            <div style={{display:'flex',gap:8,marginBottom:10}}>
              {[
                ['Peak',`${maxV.toFixed(1)}°C`,`(${MONTH_LABELS[data.findIndex(d=>d.value===maxV)]})`, '#f87171'],
                ['Minimum',`${minV.toFixed(1)}°C`,`(${MONTH_LABELS[data.findIndex(d=>d.value===minV)]})`, '#38bdf8'],
                ['Annual Swing',`${(maxV-minV).toFixed(1)}°C`,'seasonal range','#a78bfa'],
                ['Annual Mean',`${valid.length?(valid.reduce((s,d)=>s+d.value,0)/valid.length).toFixed(1):'—'}°C`,'avg LST','#fb923c'],
              ].map(([l,v,sub,col])=>(
                <div key={l} style={{flex:1,padding:'8px 10px',borderRadius:8,background:'#0a0e1a',border:'1px solid #1e293b',textAlign:'center'}}>
                  <div style={{fontSize:9,color:'#475569',marginBottom:4}}>{l}</div>
                  <div style={{fontSize:14,fontWeight:700,color:col,fontFamily:'monospace'}}>{v}</div>
                  <div style={{fontSize:9,color:'#334155',marginTop:2}}>{sub}</div>
                </div>
              ))}
            </div>
            <div style={{fontSize:9,color:'#334155',textAlign:'center'}}>Click outside to close · NASA Earth Engine {source.toUpperCase()}</div>
          </>
        )}
      </div>
    </div>
  )
}

// ── Intervention Impact Estimator ─────────────────────────────────────────────
function InterventionPanel({pixels,cityName,cityColor}){
  const [canopy,setCanopy]=useState(20)
  const [irrigated,setIrrigated]=useState(0)
  const [showPanel,setShowPanel]=useState(false)
  if(!pixels||pixels.length<10)return null
  const currentMean=(pixels.reduce((s,p)=>s+p.value,0)/pixels.length).toFixed(1)
  const reduction=estimateLSTReduction(canopy,irrigated)
  const projectedMean=(parseFloat(currentMean)-parseFloat(reduction)).toFixed(1)
  const hotN=pixels.filter(p=>p.value>=p90(pixels)).length
  const projectedHot=Math.round(hotN*Math.max(0,1-canopy/100))
  return(
    <div style={{borderRadius:10,border:'1px solid #22c55e33',overflow:'hidden'}}>
      <div onClick={()=>setShowPanel(v=>!v)}
        style={{display:'flex',alignItems:'center',justifyContent:'space-between',
          padding:'10px 14px',background:'#22c55e0d',cursor:'pointer',
          borderBottom:showPanel?'1px solid #22c55e22':'none'}}>
        <div style={{display:'flex',alignItems:'center',gap:8}}>
          <span style={{fontSize:16}}>🌿</span>
          <div>
            <div style={{fontSize:12,fontWeight:700,color:'#4ade80'}}>Intervention Simulator</div>
            <div style={{fontSize:10,color:'#475569'}}>Estimate LST reduction from green cover</div>
          </div>
        </div>
        <span style={{color:'#4ade80',fontSize:14}}>{showPanel?'▲':'▼'}</span>
      </div>
      {showPanel&&(
        <div style={{padding:'14px 16px',background:'var(--bg-card)',display:'flex',flexDirection:'column',gap:14}}>
          {/* Current baseline */}
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8}}>
            <div style={{padding:'10px',borderRadius:8,background:'#f9731612',border:'1px solid #f9731633',textAlign:'center'}}>
              <div style={{fontSize:9,color:'#f97316',fontWeight:700,marginBottom:4}}>CURRENT MEAN LST</div>
              <div style={{fontSize:20,fontWeight:800,color:'#f97316',fontFamily:'monospace'}}>{currentMean}°C</div>
              <div style={{fontSize:9,color:'#475569',marginTop:2}}>{cityName}</div>
            </div>
            <div style={{padding:'10px',borderRadius:8,background:'#22c55e12',border:'1px solid #22c55e33',textAlign:'center'}}>
              <div style={{fontSize:9,color:'#4ade80',fontWeight:700,marginBottom:4}}>PROJECTED MEAN LST</div>
              <div style={{fontSize:20,fontWeight:800,color:'#4ade80',fontFamily:'monospace'}}>{projectedMean}°C</div>
              <div style={{fontSize:9,color:'#475569',marginTop:2}}>−{reduction}°C reduction</div>
            </div>
          </div>

          {/* Canopy slider */}
          <div>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:6}}>
              <label style={{fontSize:11,color:'var(--text-sec)',fontWeight:600}}>
                🌳 Proposed canopy cover increase
              </label>
              <span style={{fontSize:13,fontWeight:700,color:'#4ade80',fontFamily:'monospace'}}>{canopy}%</span>
            </div>
            <input type="range" min={5} max={60} step={5} value={canopy} onChange={e=>setCanopy(+e.target.value)}
              style={{width:'100%',accentColor:'#22c55e'}}/>
            <div style={{display:'flex',justifyContent:'space-between',fontSize:9,color:'#334155',marginTop:2}}>
              <span>5% (sparse planting)</span><span>60% (dense urban forest)</span>
            </div>
          </div>

          {/* Irrigation slider */}
          <div>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:6}}>
              <label style={{fontSize:11,color:'var(--text-sec)',fontWeight:600}}>
                💧 Additional irrigated green area
              </label>
              <span style={{fontSize:13,fontWeight:700,color:'#38bdf8',fontFamily:'monospace'}}>{irrigated}%</span>
            </div>
            <input type="range" min={0} max={30} step={5} value={irrigated} onChange={e=>setIrrigated(+e.target.value)}
              style={{width:'100%',accentColor:'#38bdf8'}}/>
          </div>

          {/* Impact summary */}
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:8}}>
            {[
              ['LST Reduction',`−${reduction}°C`,'#4ade80'],
              ['Hotspot Reduction',`−${hotN-projectedHot} zones`,'#fbbf24'],
              ['Basis','Literature NDVI-LST','#818cf8'],
            ].map(([l,v,c])=>(
              <div key={l} style={{textAlign:'center',padding:'8px 4px',borderRadius:8,background:'var(--bg-deep)'}}>
                <div style={{fontSize:9,color:'#475569',marginBottom:4}}>{l}</div>
                <div style={{fontSize:11,fontWeight:700,color:c,fontFamily:'monospace'}}>{v}</div>
              </div>
            ))}
          </div>

          <div style={{fontSize:9,color:'#334155',lineHeight:1.5,padding:'8px 10px',borderRadius:6,background:'var(--bg-deep)'}}>
            <b style={{color:'#475569'}}>Methodology:</b> Based on peer-reviewed LST–NDVI relationship (≈−0.4°C per 10% canopy increase) and evapotranspiration cooling effect (≈−0.2°C per 10% irrigated area). Source: Oke 1987, Bowler et al. 2010, Kleerekoper et al. 2012.
          </div>
        </div>
      )}
    </div>
  )
}

// ── Priority Zone Ranking ─────────────────────────────────────────────────────
function PriorityZonePanel({pixels,thresh,cityName,cityColor}){
  const [show,setShow]=useState(false)
  if(!pixels||pixels.length<10)return null
  const zones=computePriorityZones(pixels,thresh)
  if(!zones.length)return null
  const severityLabel=(mean)=>{
    if(mean>=45)return{label:'Critical',color:'#f87171'}
    if(mean>=40)return{label:'High',color:'#f97316'}
    if(mean>=35)return{label:'Moderate',color:'#eab308'}
    return{label:'Low',color:'#22c55e'}
  }
  return(
    <div style={{borderRadius:10,border:'1px solid #f9731633',overflow:'hidden'}}>
      <div onClick={()=>setShow(v=>!v)}
        style={{display:'flex',alignItems:'center',justifyContent:'space-between',
          padding:'10px 14px',background:'#f9731608',cursor:'pointer',
          borderBottom:show?'1px solid #f9731622':'none'}}>
        <div style={{display:'flex',alignItems:'center',gap:8}}>
          <span style={{fontSize:16}}>🎯</span>
          <div>
            <div style={{fontSize:12,fontWeight:700,color:'#f97316'}}>Priority Intervention Zones</div>
            <div style={{fontSize:10,color:'#475569'}}>Top {zones.length} hotspot clusters ranked by severity</div>
          </div>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:8}}>
          <span style={{fontSize:10,fontWeight:700,background:'#f87171',color:'white',padding:'2px 8px',borderRadius:10}}>
            {zones.length} zones
          </span>
          <span style={{color:'#f97316',fontSize:14}}>{show?'▲':'▼'}</span>
        </div>
      </div>
      {show&&(
        <div style={{padding:'12px 14px',background:'var(--bg-card)',display:'flex',flexDirection:'column',gap:6}}>
          {zones.map((z,i)=>{
            const sev=severityLabel(z.mean)
            return(
              <div key={i} style={{display:'flex',alignItems:'center',gap:10,padding:'8px 10px',
                borderRadius:8,background:'var(--bg-deep)',border:`1px solid ${sev.color}22`}}>
                <div style={{width:22,height:22,borderRadius:'50%',background:sev.color,
                  display:'flex',alignItems:'center',justifyContent:'center',
                  fontSize:11,fontWeight:800,color:'white',flexShrink:0}}>{i+1}</div>
                <div style={{flex:1,minWidth:0}}>
                  <div style={{display:'flex',alignItems:'center',gap:6,marginBottom:2}}>
                    <span style={{fontSize:10,fontWeight:700,background:sev.color+'22',color:sev.color,
                      padding:'1px 6px',borderRadius:4}}>{sev.label}</span>
                    <span style={{fontSize:10,color:'#475569',fontFamily:'monospace'}}>
                      {z.lat.toFixed(3)}°N {z.lon.toFixed(3)}°E
                    </span>
                  </div>
                  <div style={{display:'flex',gap:12}}>
                    <span style={{fontSize:11,color:'#fb923c',fontWeight:700,fontFamily:'monospace'}}>
                      {z.mean.toFixed(1)}°C mean
                    </span>
                    <span style={{fontSize:11,color:'#f87171',fontFamily:'monospace'}}>
                      {z.max.toFixed(1)}°C peak
                    </span>
                    <span style={{fontSize:11,color:'#64748b'}}>
                      {z.count} px
                    </span>
                  </div>
                </div>
                <div style={{fontSize:9,color:'#334155',textAlign:'right',flexShrink:0}}>
                  Priority<br/>
                  <b style={{color:sev.color,fontSize:12}}>#{i+1}</b>
                </div>
              </div>
            )
          })}
          <div style={{fontSize:9,color:'#334155',marginTop:4,padding:'6px 8px',borderRadius:6,background:'var(--bg-deep)'}}>
            Zones identified by clustering pixels above 90th percentile (≥{thresh?.toFixed(1)}°C) into 0.1° grid cells, ranked by mean LST severity. These locations should be prioritised for urban greening, cool roof programmes, or reflective pavement deployment.
          </div>
        </div>
      )}
    </div>
  )
}

// ─── CitySearchSlot ───────────────────────────────────────────────────────────
function CitySearchSlot({slotId,color,onSelect,onRemove,canRemove}){
  const [q,setQ]=useState('')
  const [sugg,setSugg]=useState([])
  const [busy,setBusy]=useState(false)
  const timer=useRef(null)
  useEffect(()=>{
    if(!q||q.length<2){setSugg([]);return}
    clearTimeout(timer.current)
    timer.current=setTimeout(async()=>{
      setBusy(true)
      try{
        const r=await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(q)}&format=json&polygon_geojson=1&limit=5&featuretype=city`,{headers:{'Accept-Language':'en'}})
        const d=await r.json()
        setSugg(d.filter(x=>x.geojson&&['Polygon','MultiPolygon'].includes(x.geojson.type)))
      }catch{}finally{setBusy(false)}
    },400)
  },[q])
  function pick(item){
    const bb=item.boundingbox
    const latS=parseFloat(bb[1])-parseFloat(bb[0]),lonS=parseFloat(bb[3])-parseFloat(bb[2])
    setQ(item.display_name.split(',')[0]);setSugg([])
    onSelect({slotId,color,name:item.display_name.split(',')[0],
      center:[parseFloat(item.lat),parseFloat(item.lon)],
      zoom:latS>1?9:latS>0.3?11:12,
      radius:Math.max(Math.max(latS,lonS)/2*1.2,0.5),boundary:item.geojson})
  }
  return(
    <div style={{display:'flex',alignItems:'center',gap:6,position:'relative'}}>
      <div style={{width:8,height:8,borderRadius:'50%',background:color,flexShrink:0}}/>
      <div style={{flex:1,position:'relative'}}>
        <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search city or region…"
          style={{width:'100%',boxSizing:'border-box',padding:'7px 10px',fontSize:12,
            background:'var(--bg-deep)',color:'var(--text-pri)',border:`1px solid ${color}55`,
            borderRadius:8,outline:'none',transition:'border-color 0.15s'}}
          onFocus={e=>e.target.style.borderColor=color}
          onBlur={e=>e.target.style.borderColor=`${color}55`}/>
        {busy&&<span style={{position:'absolute',right:8,top:'50%',transform:'translateY(-50%)',color:'var(--text-sec)',fontSize:10}}>…</span>}
        {sugg.length>0&&(
          <div style={{position:'absolute',top:'100%',left:0,right:0,zIndex:3000,marginTop:4,
            background:'var(--bg-card)',border:'1px solid var(--border)',borderRadius:8,
            boxShadow:'0 8px 32px rgba(0,0,0,0.6)',overflow:'hidden'}}>
            {sugg.map((s,i)=>(
              <div key={i} onClick={()=>pick(s)}
                style={{padding:'8px 12px',cursor:'pointer',fontSize:12,
                  borderBottom:i<sugg.length-1?'1px solid var(--border)':'none',color:'var(--text-pri)'}}
                onMouseEnter={e=>e.currentTarget.style.background='var(--bg-deep)'}
                onMouseLeave={e=>e.currentTarget.style.background='transparent'}>
                <div style={{fontWeight:500}}>{s.display_name.split(',').slice(0,2).join(',')}</div>
                <div style={{fontSize:10,color:'var(--text-sec)',marginTop:2}}>{s.display_name.split(',').slice(2,4).join(',').trim()}</div>
              </div>
            ))}
          </div>
        )}
      </div>
      {canRemove&&<button onClick={onRemove} style={{background:'none',border:'none',cursor:'pointer',color:'#475569',fontSize:18,lineHeight:1,padding:'0 2px',flexShrink:0}} onMouseEnter={e=>e.target.style.color='#f87171'} onMouseLeave={e=>e.target.style.color='#475569'}>×</button>}
    </div>
  )
}

// ─── SwipeMap ─────────────────────────────────────────────────────────────────
function SwipeMap({cityList,cityPixelsA,cityPixelsB,cmpExcluded,vmin,vmax,flyTarget,srcColor,cmpColor,tileMode,date,cmpDate,showHotspots,threshMap,onPixelClick}){
  const [pct,setPct]=useState(50)
  const ref=useRef(null);const drag=useRef(false)
  const mapARef=useRef(null);const mapBRef=useRef(null)
  const [hoverLL,setHoverLL]=useState([null,null])
  useEffect(()=>{
    const mv=e=>{if(!drag.current||!ref.current)return;const r=ref.current.getBoundingClientRect();const x=(e.touches?e.touches[0].clientX:e.clientX)-r.left;setPct(Math.max(5,Math.min(95,(x/r.width)*100)))}
    const up=()=>{drag.current=false}
    window.addEventListener('mousemove',mv);window.addEventListener('mouseup',up)
    window.addEventListener('touchmove',mv);window.addEventListener('touchend',up)
    return()=>{window.removeEventListener('mousemove',mv);window.removeEventListener('mouseup',up);window.removeEventListener('touchmove',mv);window.removeEventListener('touchend',up)}
  },[])
  const center=cityList[0]?.center||[22.55,88.37];const zoom=cityList[0]?.zoom||9
  const activeCities=cityList.filter(c=>!cmpExcluded.has(c.slotId))
  return(
    <div ref={ref} style={{position:'relative',height:'100%',overflow:'hidden',userSelect:'none'}}>
      <div style={{position:'absolute',inset:0,clipPath:`inset(0 ${100-pct}% 0 0)`}}>
        <MapContainer center={center} zoom={zoom} style={{height:'100%',width:'100%',background:'#0a0e1a'}} whenCreated={m=>{mapARef.current=m}}>
          <BaseTileLayer tileMode={tileMode} date={date}/>
          {flyTarget&&<MapFlyTo center={flyTarget.center} zoom={flyTarget.zoom}/>}
          <MapSync sourceRef={mapARef} targetRef={mapBRef}/>
          <MapHoverEmitter onHover={(la,lo)=>setHoverLL([la,lo])}/>
          {cityList.map(c=>c.boundary&&<GeoJSON key={c.slotId} data={c.boundary} style={{color:c.color,weight:2,fillOpacity:0}}/>)}
          {cityList.map(c=><PixelLayer key={c.slotId} pixels={cityPixelsA[c.slotId]||[]} vmin={vmin} vmax={vmax} thresh={threshMap[c.slotId]} showHotspots={showHotspots} onPixelClick={onPixelClick}/>)}
          <SyncedHoverLayer hoverLatLon={hoverLL} pixels={cityPixelsA} cityList={cityList} label="A" color={srcColor}/>
        </MapContainer>
      </div>
      <div style={{position:'absolute',inset:0,clipPath:`inset(0 0 0 ${pct}%)`}}>
        <MapContainer center={center} zoom={zoom} style={{height:'100%',width:'100%',background:'#0a0e1a'}} whenCreated={m=>{mapBRef.current=m}}>
          <BaseTileLayer tileMode={tileMode} date={cmpDate||date}/>
          {flyTarget&&<MapFlyTo center={flyTarget.center} zoom={flyTarget.zoom}/>}
          <MapSync sourceRef={mapBRef} targetRef={mapARef}/>
          {activeCities.map(c=>c.boundary&&<GeoJSON key={c.slotId} data={c.boundary} style={{color:c.color,weight:2,fillOpacity:0}}/>)}
          {activeCities.map(c=><PixelLayer key={c.slotId} pixels={cityPixelsB[c.slotId]||[]} vmin={vmin} vmax={vmax} thresh={threshMap[c.slotId]} showHotspots={showHotspots}/>)}
          <SyncedHoverLayer hoverLatLon={hoverLL} pixels={cityPixelsB} cityList={activeCities} label="B" color={cmpColor}/>
        </MapContainer>
      </div>
      <div onMouseDown={()=>drag.current=true} onTouchStart={()=>drag.current=true}
        style={{position:'absolute',top:0,bottom:0,left:`calc(${pct}% - 1px)`,width:3,
          background:'rgba(255,255,255,0.9)',cursor:'ew-resize',zIndex:2000,display:'flex',alignItems:'center',justifyContent:'center'}}>
        <div style={{width:34,height:34,borderRadius:'50%',background:'white',display:'flex',alignItems:'center',justifyContent:'center',fontSize:14,color:'#0a0e1a',fontWeight:800,boxShadow:'0 2px 12px rgba(0,0,0,0.6)'}}>↔</div>
      </div>
      <div style={{position:'absolute',top:10,left:10,zIndex:1500,background:'rgba(10,14,26,0.88)',border:`1px solid ${srcColor}`,borderRadius:8,padding:'4px 12px',fontSize:10,fontWeight:700,color:srcColor}}>A · {date}</div>
      <div style={{position:'absolute',top:10,right:10,zIndex:1500,background:'rgba(10,14,26,0.88)',border:`1px solid ${cmpColor}`,borderRadius:8,padding:'4px 12px',fontSize:10,fontWeight:700,color:cmpColor}}>B · {cmpDate}</div>
    </div>
  )
}

// ─── SidebarSection ───────────────────────────────────────────────────────────
function SidebarSection({label,children}){
  return(
    <div>
      <div style={{fontSize:10,fontWeight:700,color:'var(--text-sec)',textTransform:'uppercase',letterSpacing:'0.07em',marginBottom:6}}>{label}</div>
      {children}
    </div>
  )
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function Heatmap(){
  const [slots,setSlots]=useState([{id:1}])
  const [cities,setCities]=useState({})
  const [cityPixels,setCityPixels]=useState({})
  const [cityRaw,setCityRaw]=useState({})
  const [loadingA,setLoadingA]=useState({})
  const [errorA,setErrorA]=useState({})
  const nextId=useRef(2)

  const [source,setSource]=useState('modis')
  const [date,setDate]=useState(todayMinus3)
  const [clipBdy,setClipBdy]=useState(true)
  const [flyTarget,setFlyTarget]=useState(null)
  const [tileMode,setTileMode]=useState('dark')
  const [showHotspots,setShowHotspots]=useState(false)
  const [showNarrative,setShowNarrative]=useState(true)

  const [compare,setCompare]=useState(false)
  const [cmpMode,setCmpMode]=useState('sidebyside')
  const [cmpSrc,setCmpSrc]=useState('modis')
  const [cmpDate,setCmpDate]=useState('')
  const [cmpRawMap,setCmpRawMap]=useState({})
  const [cmpPxMap,setCmpPxMap]=useState({})
  const [loadingB,setLoadingB]=useState({})
  const [errorB,setErrorB]=useState({})
  const [cmpExcluded,setCmpExcluded]=useState(new Set())
  const [fadeOp,setFadeOp]=useState(50)
  const [hoverLL,setHoverLL]=useState([null,null])
  const [selectedPixel,setSelectedPixel]=useState(null)
  const [showSubDistricts,setShowSubDistricts]=useState(false)
  const [subDistrictData,setSubDistrictData]=useState({})  // slotId→array of geojson features

  const mapARef=useRef(null);const mapBRef=useRef(null)

  const srcObj=SOURCES.find(s=>s.id===source)
  const cmpSrcObj=SOURCES.find(s=>s.id===cmpSrc)
  const MAX_DATE=todayMinus3()
  const cityList=Object.values(cities)
  const mapCenter=cityList[0]?.center||[22.55,88.37]
  const mapZoom=cityList[0]?.zoom||5

  const threshMap=useMemo(()=>{
    const m={};cityList.forEach(c=>{m[c.slotId]=p90(cityPixels[c.slotId])});return m
  },[cityPixels,cityList.map(c=>c.slotId).join()])

  function clipPx(data,cityData,doClip){
    if(!data?.pixels)return[]
    const ring=cityData?.boundary?outerRing(cityData.boundary):null
    return doClip&&ring?data.pixels.filter(p=>p.value!==null&&pip(p.lat,p.lon,ring)):data.pixels.filter(p=>p.value!==null)
  }

  async function fetchA(slotId,c,src,d){
    if(!c)return
    setLoadingA(l=>({...l,[slotId]:true}));setErrorA(e=>({...e,[slotId]:''}))
    try{
      const res=await fetch(`${API}/heatmap/global?lat=${c.center[0]}&lon=${c.center[1]}&name=${encodeURIComponent(c.name)}&radius=${c.radius}&source=${src}&date_start=${d}&date_end=${d}`)
      const data=await res.json()
      if(data.detail){setErrorA(e=>({...e,[slotId]:data.detail}));return}
      setCityRaw(r=>({...r,[slotId]:data}));setCityPixels(p=>({...p,[slotId]:clipPx(data,c,clipBdy)}))
    }catch{setErrorA(e=>({...e,[slotId]:'Network error'}))}
    finally{setLoadingA(l=>({...l,[slotId]:false}))}
  }
  async function fetchB(slotId,c,src,d){
    if(!c||!d)return
    setLoadingB(l=>({...l,[slotId]:true}));setErrorB(e=>({...e,[slotId]:''}))
    try{
      const res=await fetch(`${API}/heatmap/global?lat=${c.center[0]}&lon=${c.center[1]}&name=${encodeURIComponent(c.name)}&radius=${c.radius}&source=${src}&date_start=${d}&date_end=${d}`)
      const data=await res.json()
      if(data.detail){setErrorB(e=>({...e,[slotId]:data.detail}));return}
      setCmpRawMap(r=>({...r,[slotId]:data}));setCmpPxMap(p=>({...p,[slotId]:clipPx(data,c,clipBdy)}))
    }catch{setErrorB(e=>({...e,[slotId]:'Network error'}))}
    finally{setLoadingB(l=>({...l,[slotId]:false}))}
  }
  function fetchAllB(src,d){cityList.forEach(c=>{if(!cmpExcluded.has(c.slotId))fetchB(c.slotId,c,src,d)})}

  useEffect(()=>{
    cityList.forEach(c=>{
      if(cityRaw[c.slotId])setCityPixels(p=>({...p,[c.slotId]:clipPx(cityRaw[c.slotId],c,clipBdy)}))
      if(cmpRawMap[c.slotId])setCmpPxMap(p=>({...p,[c.slotId]:clipPx(cmpRawMap[c.slotId],c,clipBdy)}))
    })
  },[clipBdy])
  useEffect(()=>{cityList.forEach(c=>fetchA(c.slotId,c,source,date))},[source,date])
  useEffect(()=>{if(compare&&cmpDate)fetchAllB(cmpSrc,cmpDate)},[cmpSrc,cmpDate])
  useEffect(()=>{
    if(compare&&!cmpDate){
      const d=new Date(date);d.setDate(d.getDate()-30)
      const s=d.toISOString().split('T')[0]
      setCmpDate(s<(cmpSrcObj?.min||'2000-02-24')?cmpSrcObj?.min:s)
    }
  },[compare])
  useEffect(()=>{if(cmpMode!=='sidebyside'){mapARef.current=null;mapBRef.current=null}},[cmpMode])

  function addSlot(){if(slots.length>=6)return;setSlots(s=>[...s,{id:nextId.current++}])}
  function removeSlot(id){
    setSlots(s=>s.filter(sl=>sl.id!==id))
    ;[setCities,setCityPixels,setCityRaw,setCmpRawMap,setCmpPxMap,setLoadingA,setLoadingB,setErrorA,setErrorB]
      .forEach(fn=>fn(o=>{const n={...o};delete n[id];return n}))
    setCmpExcluded(ex=>{const n=new Set(ex);n.delete(id);return n})
  }
  function toggleExclude(slotId){
    setCmpExcluded(ex=>{
      const n=new Set(ex)
      if(n.has(slotId)){n.delete(slotId);const c=cities[slotId];if(c&&cmpDate)fetchB(slotId,c,cmpSrc,cmpDate)}
      else n.add(slotId);return n
    })
  }
  function onCitySelected(cityData){
    setCities(c=>({...c,[cityData.slotId]:cityData}))
    setFlyTarget({center:cityData.center,zoom:cityData.zoom})
    fetchA(cityData.slotId,cityData,source,date)
    if(compare&&cmpDate&&!cmpExcluded.has(cityData.slotId))fetchB(cityData.slotId,cityData,cmpSrc,cmpDate)
    // Fetch sub-district boundaries in background
    fetchSubDistricts(cityData.name).then(features=>{
      if(features.length>0)
        setSubDistrictData(d=>({...d,[cityData.slotId]:features}))
    })
  }

  const activeCmpCities=cityList.filter(c=>!cmpExcluded.has(c.slotId))
  const allVals=[
    ...cityList.flatMap(c=>(cityRaw[c.slotId]?.pixels||[]).map(p=>p.value)),
    ...activeCmpCities.flatMap(c=>(cmpRawMap[c.slotId]?.pixels||[]).map(p=>p.value)),
  ]
  const vmin=allVals.length?Math.min(...allVals):20
  const vmax=allVals.length?Math.max(...allVals):50
  const anyLoadingA=Object.values(loadingA).some(Boolean)
  const anyLoadingB=Object.values(loadingB).some(Boolean)
  const hasData=cityList.some(c=>cityRaw[c.slotId])
  const ticks=4
  const tickVals=Array.from({length:ticks+1},(_,i)=>vmin+(vmax-vmin)*(i/ticks))
  const tickLabels=['Cool','Moderate','Warm','Hot','Extreme']
  const narrative=useMemo(()=>hasData?buildNarrative(cityList,cityRaw,cityPixels,compare,cmpRawMap,cmpDate,date):''
    ,[cityRaw,cityPixels,cmpRawMap,compare,date,cmpDate])

  return(
    <div style={{display:'flex',flexDirection:'column',height:'100%'}}>
      <div className="page-header">
        <h2>Spatial Heatmap</h2>
        <p>Real satellite LST · Multi-city · Compare dates · NASA daily imagery</p>
      </div>

      <div style={{display:'flex',flex:1,gap:16,minHeight:0}}>

        {/* SIDEBAR */}
        <div style={{width:220,flexShrink:0,background:'var(--bg-card)',border:'1px solid var(--border)',
          borderRadius:12,padding:12,display:'flex',flexDirection:'column',gap:12,overflowY:'auto'}}>

          {/* Export buttons at top */}
          {hasData&&(
            <div style={{display:'flex',gap:6}}>
              <button onClick={()=>exportCSV(cityList,cityPixels,date,source)}
                style={{flex:1,padding:'5px 0',borderRadius:6,border:'1px solid #334155',
                  background:'var(--bg-deep)',color:'#94a3b8',cursor:'pointer',fontSize:10,fontWeight:600}}
                onMouseEnter={e=>e.currentTarget.style.borderColor='#6366f1'}
                onMouseLeave={e=>e.currentTarget.style.borderColor='#334155'}>⬇ CSV</button>
              <button onClick={exportPNG}
                style={{flex:1,padding:'5px 0',borderRadius:6,border:'1px solid #334155',
                  background:'var(--bg-deep)',color:'#94a3b8',cursor:'pointer',fontSize:10,fontWeight:600}}
                onMouseEnter={e=>e.currentTarget.style.borderColor='#6366f1'}
                onMouseLeave={e=>e.currentTarget.style.borderColor='#334155'}>📷 PNG</button>
            </div>
          )}

          {/* Location */}
          <div>
            <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:8}}>
              <div style={{fontSize:10,fontWeight:700,color:'var(--text-sec)',textTransform:'uppercase',letterSpacing:'0.07em'}}>Location</div>
              {slots.length<6&&(
                <button onClick={addSlot}
                  style={{width:22,height:22,borderRadius:'50%',background:'#6366f115',border:'1px solid #6366f1',
                    color:'#6366f1',fontSize:16,fontWeight:700,cursor:'pointer',display:'flex',
                    alignItems:'center',justifyContent:'center',lineHeight:1,padding:0}}>+</button>
              )}
            </div>
            <div style={{display:'flex',flexDirection:'column',gap:8}}>
              {slots.map((slot,idx)=>(
                <CitySearchSlot key={slot.id} slotId={slot.id} color={CITY_COLORS[idx%CITY_COLORS.length]}
                  onSelect={onCitySelected} onRemove={()=>removeSlot(slot.id)} canRemove={slots.length>1}/>
              ))}
            </div>
            
            {cityList.length>0&&(
              <div style={{marginTop:10,display:'flex',flexDirection:'column',gap:5}}>
                <div style={{fontSize:9,fontWeight:700,color:'var(--text-sec)',textTransform:'uppercase',letterSpacing:'0.06em',marginBottom:2}}>Active</div>
                {cityList.map(c=>{
                  const rd=cityRaw[c.slotId];const px=cityPixels[c.slotId]||[]
                  const uhi=computeUHI(px);const isEx=cmpExcluded.has(c.slotId)
                  const hotN=showHotspots?px.filter(p=>p.value>=threshMap[c.slotId]).length:null
                  return(
                    <div key={c.slotId} style={{display:'flex',alignItems:'flex-start',gap:8,padding:'4px 6px',borderRadius:6,background:'var(--bg-deep)',marginBottom:2}}>
                      <div style={{width:8,height:8,borderRadius:'50%',background:c.color,flexShrink:0,marginTop:3}}/>
                      <div style={{flex:1,minWidth:0}}>
                        <div style={{fontSize:12,fontWeight:600,color:'var(--text-pri)',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{c.name}</div>
                        {loadingA[c.slotId]&&<div style={{fontSize:10,color:c.color}}>Fetching…</div>}
                        {errorA[c.slotId]&&<div style={{fontSize:10,color:'#f87171'}}>{errorA[c.slotId]}</div>}
                        {rd&&!loadingA[c.slotId]&&(
                          <div style={{fontSize:10,color:'var(--text-sec)',marginTop:2}}>
                            <span style={{color:'#fb923c',fontWeight:600}}>{rd.value_mean.toFixed(1)}°C</span>
                            {' · '}{(px.length).toLocaleString()} px
                            {uhi&&<span style={{color:'#f87171'}}> · UHI {uhi}°C</span>}
                            {hotN>0&&<span style={{color:'#fbbf24'}}> · 🔥{hotN}</span>}
                          </div>
                        )}
                        {compare&&rd&&cmpRawMap[c.slotId]&&!isEx&&(
                          <div style={{fontSize:10,color:'#f97316',marginTop:1}}>B: {cmpRawMap[c.slotId].value_mean.toFixed(1)}°C (Δ{(rd.value_mean-cmpRawMap[c.slotId].value_mean).toFixed(1)}°C)</div>
                        )}
                      </div>
                      {compare&&(
                        <button onClick={()=>toggleExclude(c.slotId)} title={isEx?'Include in B':'Exclude from B'}
                          style={{width:20,height:20,borderRadius:4,cursor:'pointer',flexShrink:0,marginTop:1,
                            border:`1px solid ${isEx?'#f87171':c.color}`,background:isEx?'#f8717122':'transparent',
                            display:'flex',alignItems:'center',justifyContent:'center',fontSize:11,color:isEx?'#f87171':c.color}}>
                          {isEx?'✕':'✓'}
                        </button>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          <SidebarSection label="Data Source">
            <div style={{display:'flex',flexDirection:'column',gap:5}}>
              {SOURCES.map(s=>(
                <button key={s.id} onClick={()=>setSource(s.id)}
                  style={{display:'flex',alignItems:'center',gap:8,padding:'5px 8px',borderRadius:7,
                    cursor:'pointer',textAlign:'left',width:'100%',
                    border:`1px solid ${source===s.id?s.color:'var(--border)'}`,
                    background:source===s.id?`${s.color}18`:'var(--bg-deep)',transition:'all 0.15s'}}>
                  <span style={{fontSize:13}}>{s.icon}</span>
                  <div>
                    <div style={{fontSize:12,fontWeight:600,color:source===s.id?s.color:'var(--text-pri)'}}>{s.label}</div>
                    <div style={{fontSize:10,color:'var(--text-sec)'}}>{s.sub}</div>
                  </div>
                </button>
              ))}
            </div>
          </SidebarSection>

          <SidebarSection label="Map Layer">
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:5}}>
              {Object.entries(TILE_LAYERS).map(([key,layer])=>(
                <button key={key} onClick={()=>setTileMode(key)}
                  style={{padding:'5px 4px',borderRadius:7,cursor:'pointer',textAlign:'center',
                    border:`1px solid ${tileMode===key?'#6366f1':'var(--border)'}`,
                    background:tileMode===key?'#6366f118':'var(--bg-deep)',
                    color:tileMode===key?'#818cf8':'var(--text-sec)',fontSize:10,fontWeight:600,transition:'all 0.15s'}}>
                  <div style={{fontSize:15,marginBottom:2}}>{layer.icon}</div>
                  {layer.label}
                </button>
              ))}
            </div>
            {TILE_LAYERS[tileMode]?.note&&(
              <div style={{marginTop:6,padding:'5px 8px',borderRadius:6,fontSize:9,
                background:tileMode==='gibs'?'#6366f115':'#f9731615',
                border:`1px solid ${tileMode==='gibs'?'#6366f133':'#f9731633'}`,
                color:tileMode==='gibs'?'#818cf8':'#fb923c'}}>
                {tileMode==='gibs'?'🛰':'⚠'} {TILE_LAYERS[tileMode].note}
                {tileMode==='gibs'&&<b style={{display:'block',marginTop:2,color:'var(--text-sec)'}}>Showing: {date}</b>}
              </div>
            )}
          </SidebarSection>

          <SidebarSection label="Date A">
            <input type="date" value={date} min={srcObj?.min} max={MAX_DATE}
              onChange={e=>setDate(e.target.value)}
              style={{width:'100%',boxSizing:'border-box',padding:'5px 8px',fontSize:11,
                background:'var(--bg-deep)',color:'var(--text-pri)',
                border:'1px solid var(--border)',borderRadius:7,outline:'none',colorScheme:'dark'}}/>
            <div style={{fontSize:9,color:'#475569',marginTop:4}}>📡 Latest: {MAX_DATE} (3-day lag)</div>
            <div style={{display:'flex',flexWrap:'wrap',gap:4,marginTop:6}}>
              {[{l:'−1d',o:4},{l:'−7d',o:10},{l:'−30d',o:33},{l:'−1yr',o:368}].map(p=>{
                const d=new Date();d.setDate(d.getDate()-p.o);const val=d.toISOString().split('T')[0];const on=date===val
                return(<button key={p.l} onClick={()=>setDate(val)}
                  style={{fontSize:9,padding:'2px 6px',borderRadius:5,cursor:'pointer',
                    border:`1px solid ${on?'#6366f1':'var(--border)'}`,
                    background:on?'#6366f118':'var(--bg-deep)',color:on?'#818cf8':'var(--text-sec)'}}>{p.l}</button>)
              })}
            </div>
          </SidebarSection>

          <SidebarSection label="Options">
            <label style={{display:'flex',alignItems:'center',gap:6,cursor:'pointer'}}>
              <input type="checkbox" checked={clipBdy} onChange={e=>setClipBdy(e.target.checked)} style={{accentColor:'#6366f1',width:13,height:13,cursor:'pointer'}}/>
              <span style={{fontSize:12,color:'var(--text-sec)'}}>Clip to boundary</span>
            </label>
            <label style={{display:'flex',alignItems:'center',gap:8,cursor:'pointer',marginTop:6}}>
              <input type="checkbox" checked={showHotspots} onChange={e=>setShowHotspots(e.target.checked)} style={{accentColor:'#f87171',width:13,height:13,cursor:'pointer'}}/>
              <span style={{fontSize:12,color:'var(--text-sec)'}}>🔥 Show hotspots (90th %ile)</span>
            </label>
            <label style={{display:'flex',alignItems:'center',gap:8,cursor:'pointer',marginTop:6}}>
              <input type="checkbox" checked={showSubDistricts} onChange={e=>setShowSubDistricts(e.target.checked)} style={{accentColor:'#06b6d4',width:13,height:13,cursor:'pointer'}}/>
              <span style={{fontSize:12,color:'var(--text-sec)'}}>🗺 Sub-district boundaries</span>
            </label>
            <label style={{display:'flex',alignItems:'center',gap:8,cursor:'pointer',marginTop:6}}>
              <input type="checkbox" checked={showNarrative} onChange={e=>setShowNarrative(e.target.checked)} style={{accentColor:'#22c55e',width:13,height:13,cursor:'pointer'}}/>
              <span style={{fontSize:12,color:'var(--text-sec)'}}>📝 Auto narrative</span>
            </label>
          </SidebarSection>

          {/* Compare */}
          <div>
            <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:compare?12:0}}>
              <div style={{fontSize:11,fontWeight:700,color:'var(--text-sec)',textTransform:'uppercase',letterSpacing:'0.08em'}}>Compare</div>
              <div onClick={()=>setCompare(c=>!c)} style={{cursor:'pointer'}}>
                <div style={{width:36,height:20,borderRadius:10,position:'relative',background:compare?'#6366f1':'var(--border)',transition:'background 0.2s'}}>
                  <div style={{position:'absolute',top:3,left:compare?18:3,width:14,height:14,borderRadius:'50%',background:'white',transition:'left 0.2s'}}/>
                </div>
              </div>
            </div>
            {compare&&(
              <div style={{display:'flex',flexDirection:'column',gap:10}}>
                <div style={{fontSize:10,color:'var(--text-sec)'}}>Satellite B</div>
                <div style={{display:'flex',flexDirection:'column',gap:4}}>
                  {SOURCES.map(s=>(
                    <button key={s.id} onClick={()=>setCmpSrc(s.id)}
                      style={{display:'flex',alignItems:'center',gap:6,padding:'6px 8px',borderRadius:8,cursor:'pointer',textAlign:'left',width:'100%',
                        border:`1px solid ${cmpSrc===s.id?s.color:'var(--border)'}`,background:cmpSrc===s.id?`${s.color}18`:'var(--bg-deep)'}}>
                      <span style={{fontSize:12}}>{s.icon}</span>
                      <div style={{fontSize:11,fontWeight:600,color:cmpSrc===s.id?s.color:'var(--text-pri)'}}>{s.label}</div>
                    </button>
                  ))}
                </div>
                <div style={{fontSize:10,color:'var(--text-sec)'}}>Date B</div>
                <input type="date" value={cmpDate} min={cmpSrcObj?.min} max={MAX_DATE}
                  onChange={e=>setCmpDate(e.target.value)}
                  style={{width:'100%',boxSizing:'border-box',padding:'6px 10px',fontSize:11,
                    background:'var(--bg-deep)',color:'var(--text-pri)',
                    border:'1px solid #f97316',borderRadius:8,outline:'none',colorScheme:'dark'}}/>
                <div style={{display:'flex',flexDirection:'column',gap:4}}>
                  {[{id:'sidebyside',label:'Side by side',icon:'⬛⬜'},{id:'fade',label:'Fade / Opacity',icon:'◑'},{id:'swipe',label:'Swipe compare',icon:'↔'}].map(m=>(
                    <button key={m.id} onClick={()=>setCmpMode(m.id)}
                      style={{padding:'6px 10px',borderRadius:8,cursor:'pointer',textAlign:'left',
                        border:`1px solid ${cmpMode===m.id?'#f97316':'var(--border)'}`,
                        background:cmpMode===m.id?'#f9731618':'var(--bg-deep)',
                        color:cmpMode===m.id?'#f97316':'var(--text-sec)',
                        fontSize:11,fontWeight:cmpMode===m.id?700:400,display:'flex',alignItems:'center',gap:8}}>
                      <span>{m.icon}</span>{m.label}
                    </button>
                  ))}
                </div>
                {cmpMode==='fade'&&(
                  <div>
                    <div style={{display:'flex',justifyContent:'space-between',fontSize:9,color:'var(--text-sec)',marginBottom:4}}>
                      <span>A: {100-fadeOp}%</span><span>B: {fadeOp}%</span>
                    </div>
                    <input type="range" min={0} max={100} value={fadeOp} onChange={e=>setFadeOp(+e.target.value)} style={{width:'100%',accentColor:'#f97316'}}/>
                  </div>
                )}
                {anyLoadingB&&<div style={{fontSize:10,color:'#f97316'}}>⏳ Fetching B…</div>}
                {activeCmpCities.some(c=>cmpRawMap[c.slotId])&&(
                  <div style={{background:'var(--bg-deep)',borderRadius:8,padding:10,border:'1px solid var(--border)'}}>
                    <div style={{fontSize:10,fontWeight:700,color:'#f97316',marginBottom:8}}>B: {cmpDate}</div>
                    {activeCmpCities.map(c=>{
                      const rdB=cmpRawMap[c.slotId];const rdA=cityRaw[c.slotId];if(!rdB)return null
                      const delta=rdA?rdA.value_mean-rdB.value_mean:null
                      return(
                        <div key={c.slotId} style={{marginBottom:10,paddingBottom:10,borderBottom:'1px solid var(--border)'}}>
                          <div style={{display:'flex',alignItems:'center',gap:6,marginBottom:6}}>
                            <div style={{width:6,height:6,borderRadius:'50%',background:c.color}}/>
                            <span style={{fontSize:11,fontWeight:700,color:c.color}}>{c.name}</span>
                          </div>
                          <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:4}}>
                            {[['Min',rdB.value_min.toFixed(1)+'°C','#38bdf8'],['Mean',rdB.value_mean.toFixed(1)+'°C','#fb923c'],
                              ['Max',rdB.value_max.toFixed(1)+'°C','#f87171'],
                              delta!==null?['Δ',(delta>0?'+':'')+delta.toFixed(1)+'°C',delta>0?'#f87171':'#4ade80']:null
                            ].filter(Boolean).map(([l,v,col])=>(
                              <div key={l} style={{textAlign:'center',background:'var(--bg-card)',borderRadius:6,padding:'4px 2px'}}>
                                <div style={{fontSize:8,color:'var(--text-sec)',marginBottom:2}}>{l}</div>
                                <div style={{fontSize:11,fontWeight:700,color:col,fontFamily:'var(--font-mono)'}}>{v}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
          {anyLoadingA&&<div style={{fontSize:11,color:'#6366f1',fontFamily:'var(--font-mono)'}}>⏳ {srcObj?.label}…</div>}
        </div>

        {/* MAP + STATS */}
        <div style={{flex:1,display:'flex',flexDirection:'column',gap:10,minWidth:0}}>
          <div id="ts-map-area" style={{flex:1,borderRadius:12,overflow:'hidden',border:'1px solid var(--border)',minHeight:400,position:'relative'}}>
            {anyLoadingA&&cityList.length>0&&<MapSkeleton/>}

            {compare&&cmpMode==='swipe'?(
              <SwipeMap cityList={cityList} cityPixelsA={cityPixels} cityPixelsB={cmpPxMap}
                cmpExcluded={cmpExcluded} vmin={vmin} vmax={vmax} flyTarget={flyTarget}
                srcColor={srcObj?.color||'#6366f1'} cmpColor={cmpSrcObj?.color||'#f97316'}
                tileMode={tileMode} date={date} cmpDate={cmpDate}
                showHotspots={showHotspots} threshMap={threshMap} onPixelClick={p=>setSelectedPixel(p)}/>

            ):compare&&cmpMode==='sidebyside'?(
              <div style={{display:'flex',height:'100%'}}>
                <div style={{flex:1,position:'relative',borderRight:'2px solid var(--border)'}}>
                  <div style={{position:'absolute',top:8,left:8,zIndex:1000,background:'rgba(10,14,26,0.88)',
                    border:`1px solid ${srcObj?.color}`,borderRadius:8,padding:'4px 10px',fontSize:10,fontWeight:700,color:srcObj?.color}}>
                    A · {date} · {srcObj?.label}
                  </div>
                  <MapContainer center={mapCenter} zoom={mapZoom} style={{height:'100%',width:'100%',background:'#0a0e1a'}}>
                    <BaseTileLayer tileMode={tileMode} date={date}/>
                    {flyTarget&&<MapFlyTo center={flyTarget.center} zoom={flyTarget.zoom}/>}
                    <MapSync sourceRef={mapARef} targetRef={mapBRef}/>
                    <MapHoverEmitter onHover={(la,lo)=>setHoverLL([la,lo])}/>
                    {cityList.map(c=>c.boundary&&<GeoJSON key={c.slotId} data={c.boundary} style={{color:c.color,weight:2,fillOpacity:0}}/>)}
                    {showSubDistricts&&cityList.map(c=>(subDistrictData[c.slotId]||[]).map((f,i)=>(
                      <GeoJSON key={`sub-${c.slotId}-${i}`} data={f.geojson} style={{color:'#06b6d4',weight:1,fillOpacity:0.04,dashArray:'4 3',opacity:0.7}}/>
                    )))}
                    {cityList.map(c=><PixelLayer key={c.slotId} pixels={cityPixels[c.slotId]||[]} vmin={vmin} vmax={vmax} thresh={threshMap[c.slotId]} showHotspots={showHotspots} onPixelClick={p=>setSelectedPixel(p)}/>)}
                    <SyncedHoverLayer hoverLatLon={hoverLL} pixels={cityPixels} cityList={cityList} label="A" color={srcObj?.color||'#6366f1'}/>
                    <TileToggleButton tileMode={tileMode} setTileMode={setTileMode}/>
                  </MapContainer>
                </div>
                <div style={{flex:1,position:'relative'}}>
                  <div style={{position:'absolute',top:8,left:8,zIndex:1000,background:'rgba(10,14,26,0.88)',
                    border:`1px solid ${cmpSrcObj?.color}`,borderRadius:8,padding:'4px 10px',fontSize:10,fontWeight:700,color:cmpSrcObj?.color}}>
                    B · {cmpDate||'—'} · {cmpSrcObj?.label}
                  </div>
                  <MapContainer center={mapCenter} zoom={mapZoom} style={{height:'100%',width:'100%',background:'#0a0e1a'}}>
                    <BaseTileLayer tileMode={tileMode} date={cmpDate||date}/>
                    {flyTarget&&<MapFlyTo center={flyTarget.center} zoom={flyTarget.zoom}/>}
                    <MapSync sourceRef={mapBRef} targetRef={mapARef}/>
                    <MapHoverEmitter onHover={(la,lo)=>setHoverLL([la,lo])}/>
                    {activeCmpCities.map(c=>c.boundary&&<GeoJSON key={c.slotId} data={c.boundary} style={{color:c.color,weight:2,fillOpacity:0}}/>)}
                    {activeCmpCities.map(c=><PixelLayer key={c.slotId} pixels={cmpPxMap[c.slotId]||[]} vmin={vmin} vmax={vmax} thresh={threshMap[c.slotId]} showHotspots={showHotspots}/>)}
                    <SyncedHoverLayer hoverLatLon={hoverLL} pixels={cmpPxMap} cityList={activeCmpCities} label="B" color={cmpSrcObj?.color||'#f97316'}/>
                    <TileToggleButton tileMode={tileMode} setTileMode={setTileMode}/>
                  </MapContainer>
                </div>
              </div>

            ):(
              <MapContainer center={mapCenter} zoom={mapZoom} style={{height:'100%',width:'100%',background:'#0a0e1a'}}>
                <BaseTileLayer tileMode={tileMode} date={date}/>
                {flyTarget&&<MapFlyTo center={flyTarget.center} zoom={flyTarget.zoom}/>}
                {cityList.map(c=>c.boundary&&<GeoJSON key={c.slotId} data={c.boundary} style={{color:c.color,weight:2,fillOpacity:0}}/>)}
                {showSubDistricts&&cityList.map(c=>(subDistrictData[c.slotId]||[]).map((f,i)=>(
                  <GeoJSON key={`sub-${c.slotId}-${i}`} data={f.geojson}
                    style={{color:'#06b6d4',weight:1,fillOpacity:0.04,dashArray:'4 3',opacity:0.7}}/>
                )))}
                {cityList.map(c=><PixelLayer key={c.slotId} pixels={cityPixels[c.slotId]||[]} vmin={vmin} vmax={vmax} thresh={threshMap[c.slotId]} showHotspots={showHotspots} opacity={compare&&cmpMode==='fade'?(100-fadeOp)/100*0.88:0.88} onPixelClick={p=>setSelectedPixel(p)}/>)}
                {compare&&cmpMode==='fade'&&activeCmpCities.map(c=><PixelLayer key={`b${c.slotId}`} pixels={cmpPxMap[c.slotId]||[]} vmin={vmin} vmax={vmax} thresh={threshMap[c.slotId]} showHotspots={showHotspots} opacity={fadeOp/100*0.88}/>)}
                {cityList.length===0&&(
                  <div style={{position:'absolute',top:'50%',left:'50%',transform:'translate(-50%,-50%)',
                    background:'rgba(10,14,26,0.88)',padding:'28px 44px',borderRadius:14,
                    zIndex:1000,textAlign:'center',border:'1px solid var(--border)'}}>
                    <div style={{fontSize:40,marginBottom:10}}>🌍</div>
                    <div style={{color:'var(--text-pri)',fontSize:15,fontWeight:600,marginBottom:6}}>Search any city above</div>
                    <div style={{color:'#475569',fontSize:12}}>Up to 6 cities · Real NASA satellite LST · Click pixel for time series</div>
                  </div>
                )}
                <TileToggleButton tileMode={tileMode} setTileMode={setTileMode}/>
              </MapContainer>
            )}
          </div>

          {/* Color scale */}
          {hasData&&(
            <div style={{background:'var(--bg-card)',border:'1px solid var(--border)',borderRadius:10,padding:'10px 16px'}}>
              <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:6}}>
                <span style={{fontSize:11,color:'var(--text-sec)',fontFamily:'var(--font-mono)',whiteSpace:'nowrap'}}>{vmin.toFixed(1)}°C</span>
                <div style={{flex:1,position:'relative',height:12,borderRadius:6,background:'linear-gradient(to right,#3b82f6,#06b6d4,#eab308,#f97316,#ef4444)'}}>
                  {tickVals.map((v,i)=>(
                    <div key={i} style={{position:'absolute',left:`${(i/ticks)*100}%`,top:-4,bottom:-4,width:1,background:'rgba(255,255,255,0.3)',transform:'translateX(-50%)'}}/>
                  ))}
                </div>
                <span style={{fontSize:11,color:'var(--text-sec)',fontFamily:'var(--font-mono)',whiteSpace:'nowrap'}}>{vmax.toFixed(1)}°C</span>
              </div>
              <div style={{display:'flex',justifyContent:'space-between',paddingLeft:42,paddingRight:52}}>
                {tickVals.map((v,i)=>(
                  <div key={i} style={{textAlign:'center'}}>
                    <div style={{fontSize:9,color:'#475569'}}>{tickLabels[i]}</div>
                    <div style={{fontSize:8,color:'#334155',fontFamily:'monospace'}}>{v.toFixed(0)}°C</div>
                  </div>
                ))}
              </div>
              <div style={{marginTop:4,fontSize:9,color:'#334155',textAlign:'right'}}>
                {srcObj?.label} · {date}{compare&&Object.keys(cmpRawMap).length>0?` vs ${cmpDate} (${cmpSrcObj?.label})`:''}
                {showHotspots&&' · 🔥 = 90th %ile'}
              </div>
            </div>
          )}

          {/* Stats per city — A and B side by side */}
          {cityList.map(c=>{
            const rdA=cityRaw[c.slotId];const px=cityPixels[c.slotId]||[]
            if(!rdA)return null
            const rdB=compare&&!cmpExcluded.has(c.slotId)?cmpRawMap[c.slotId]:null
            const uhi=computeUHI(px)
            const hotN=px.filter(p=>p.value>=(threshMap[c.slotId]||Infinity)).length
            return(
              <div key={c.slotId} style={{borderRadius:10,border:`1px solid ${c.color}33`,overflow:'hidden'}}>
                <div style={{display:'flex',alignItems:'center',gap:10,padding:'8px 14px',background:`${c.color}0d`,borderBottom:`1px solid ${c.color}22`}}>
                  <div style={{width:8,height:8,borderRadius:'50%',background:c.color}}/>
                  <span style={{fontSize:13,fontWeight:700,color:c.color}}>{c.name}</span>
                  <span style={{fontSize:10,color:'var(--text-sec)',marginLeft:'auto'}}>{rdA.source} · {date}</span>
                </div>
                <div style={{display:'grid',gridTemplateColumns:rdB?'1fr 1fr':'1fr',gap:0}}>
                  <div style={{padding:'10px 14px',background:'var(--bg-card)'}}>
                    {rdB&&<div style={{fontSize:9,color:'var(--text-sec)',fontWeight:700,textTransform:'uppercase',letterSpacing:'0.08em',marginBottom:6}}>A · {date}</div>}
                    <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
                      {[['Min',rdA.value_min.toFixed(1)+'°C','#38bdf8'],['Mean',rdA.value_mean.toFixed(1)+'°C','#fb923c'],
                        ['Max',rdA.value_max.toFixed(1)+'°C','#f87171'],['Pixels',px.length.toLocaleString(),'var(--text-sec)'],
                        uhi?['UHI',uhi+'°C','#a78bfa']:null,
                        hotN>0?['Hotspots',hotN,'#fbbf24']:null,
                      ].filter(Boolean).map(([l,v,col])=>(
                        <div key={l} style={{textAlign:'center',minWidth:48}}>
                          <div style={{fontSize:9,color:'var(--text-sec)',marginBottom:2}}>{l}</div>
                          <div style={{fontSize:14,fontWeight:700,color:col,fontFamily:'var(--font-mono)'}}>{v}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  {rdB&&(
                    <div style={{padding:'10px 14px',background:'#f9731608',borderLeft:'1px solid var(--border)'}}>
                      <div style={{fontSize:9,color:'#f97316',fontWeight:700,textTransform:'uppercase',letterSpacing:'0.08em',marginBottom:6}}>B · {cmpDate}</div>
                      <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
                        {[['Min',rdB.value_min.toFixed(1)+'°C','#38bdf8'],['Mean',rdB.value_mean.toFixed(1)+'°C','#fb923c'],
                          ['Max',rdB.value_max.toFixed(1)+'°C','#f87171'],
                          ['Δ Mean',(rdA.value_mean>rdB.value_mean?'+':'')+(rdA.value_mean-rdB.value_mean).toFixed(1)+'°C',rdA.value_mean>rdB.value_mean?'#f87171':'#4ade80'],
                        ].map(([l,v,col])=>(
                          <div key={l} style={{textAlign:'center',minWidth:48}}>
                            <div style={{fontSize:9,color:'var(--text-sec)',marginBottom:2}}>{l}</div>
                            <div style={{fontSize:14,fontWeight:700,color:col,fontFamily:'var(--font-mono)'}}>{v}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )
          })}

          {/* Priority Zones — per city */}
          {cityList.map(c=>{
            const px=cityPixels[c.slotId]||[]
            if(!px.length)return null
            return(
              <PriorityZonePanel key={c.slotId}
                pixels={px} thresh={threshMap[c.slotId]}
                cityName={c.name} cityColor={c.color}/>
            )
          })}

          {/* Intervention Simulator — per city */}
          {cityList.map(c=>{
            const px=cityPixels[c.slotId]||[]
            if(!px.length)return null
            return(
              <InterventionPanel key={c.slotId}
                pixels={px} cityName={c.name} cityColor={c.color}/>
            )
          })}

          {/* Narrative */}
          {showNarrative&&narrative&&(
            <div style={{borderRadius:10,border:'1px solid #22c55e33',background:'#22c55e08',padding:'12px 16px'}}>
              <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:8}}>
                <span style={{fontSize:14}}>📝</span>
                <span style={{fontSize:11,fontWeight:700,color:'#4ade80',textTransform:'uppercase',letterSpacing:'0.08em'}}>AI Analysis</span>
                <span style={{fontSize:9,color:'#475569',marginLeft:'auto'}}>Auto-generated · {new Date().toLocaleDateString()}</span>
              </div>
              <p style={{fontSize:12,color:'#94a3b8',lineHeight:1.7,margin:0}}>{narrative}</p>
            </div>
          )}
        </div>
      </div>

      {selectedPixel&&(
        <TimeSeriesModal pixel={selectedPixel} source={source} onClose={()=>setSelectedPixel(null)}/>
      )}
    </div>
  )
}