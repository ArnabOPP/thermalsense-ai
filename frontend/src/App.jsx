// ThermalSense AI — Main App
// Sidebar navigation + page routing.
// Person D: to add a new page, add it to PAGES and create the component in src/pages/

import { useState } from 'react'
import Overview       from './pages/Overview.jsx'
import Heatmap        from './pages/Heatmap.jsx'
import Scenarios      from './pages/Scenarios.jsx'
import Explainability from './pages/Explainability.jsx'

const PAGES = [
  { id: 'overview',       label: 'Overview',         icon: '📊', component: Overview },
  { id: 'heatmap',        label: 'Heat Map',          icon: '🗺️',  component: Heatmap },
  { id: 'scenarios',      label: 'Interventions',     icon: '🌿', component: Scenarios },
  { id: 'explainability', label: 'Explainability',    icon: '🔍', component: Explainability },
]

export default function App() {
  const [activePage, setActivePage] = useState('overview')

  const ActiveComponent = PAGES.find(p => p.id === activePage)?.component ?? Overview

  return (
    <div className="shell">
      {/* Sidebar */}
      <nav className="sidebar">
        <div className="sidebar-logo">
          <h1>ThermalSense AI</h1>
          <p>BAH 2026 · PS1 · Kolkata</p>
        </div>

        {PAGES.map(page => (
          <div
            key={page.id}
            className={`nav-item ${activePage === page.id ? 'active' : ''}`}
            onClick={() => setActivePage(page.id)}
          >
            <span className="nav-icon">{page.icon}</span>
            {page.label}
          </div>
        ))}

        <div className="isro-badge">
          🛰 ISRO · BAH 2026<br />
          Problem Statement 1<br />
          Urban Heat Mitigation
        </div>
      </nav>

      {/* Main content */}
      <main className="main">
        <ActiveComponent />
      </main>
    </div>
  )
}
