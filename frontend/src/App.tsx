import { lazy, Suspense, useEffect, useState } from 'react'
import { NavLink, Navigate, Outlet, Route, Routes, useLocation } from 'react-router-dom'
import { BarChart3, Boxes, Cpu, Gauge, Layers3, MessageSquareText, Settings, ShieldCheck, Wifi } from 'lucide-react'
import { api } from './services/api'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const Models = lazy(() => import('./pages/Models'))
const ImportModel = lazy(() => import('./pages/ImportModel'))
const Chat = lazy(() => import('./pages/Chat'))
const Evaluation = lazy(() => import('./pages/Evaluation'))
const Benchmark = lazy(() => import('./pages/Benchmark'))
const SettingsPage = lazy(() => import('./pages/SettingsPage'))

const links = [
  { to: '/dashboard', label: 'Overview', icon: BarChart3 },
  { to: '/models', label: 'Models', icon: Boxes },
  { to: '/chat', label: 'Inference lab', icon: MessageSquareText },
  { to: '/evaluation', label: 'Project evaluation', icon: Layers3 },
  { to: '/benchmark', label: 'Benchmarks', icon: Gauge },
]

function Shell() {
  const [online, setOnline] = useState(false)
  const location = useLocation()
  useEffect(() => { api.health().then(() => setOnline(true)).catch(() => setOnline(false)); const id = window.setInterval(() => api.health().then(() => setOnline(true)).catch(() => setOnline(false)), 10000); return () => window.clearInterval(id) }, [])
  const title = links.find(link => location.pathname.startsWith(link.to))?.label ?? 'Settings'
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Cpu size={21} /></div><div><strong>Quant<span>Lab</span></strong><small>LOCAL MODEL STUDIO</small></div></div>
      <div className="side-caption">WORKSPACE</div>
      <nav>{links.map(({ to, label, icon: Icon }) => <NavLink key={to} to={to} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}><Icon size={18} /><span>{label}</span>{label === 'Project evaluation' && <span className="nav-dot" />}</NavLink>)}</nav>
      <div className="side-caption tools-caption">PREFERENCES</div><NavLink to="/settings" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}><Settings size={18} /><span>Settings & hardware</span></NavLink>
      <div className="sidebar-bottom"><div className="privacy-badge"><ShieldCheck size={17} /><div><strong>Your models stay local</strong><small>No cloud. No account.</small></div></div><div className="side-version">QUANTLAB <span>v1.0</span></div></div>
    </aside>
    <main className="main-area"><header className="topbar"><div className="mobile-brand"><div className="brand-mark"><Cpu size={18} /></div>Quant<span>Lab</span></div><div className="breadcrumb">Workspace <span>/</span> <b>{title}</b></div><div className="topbar-right"><div className={`connection-pill ${online ? 'connected' : 'disconnected'}`}><span className="connection-dot" />{online ? 'LOCAL MODE' : 'BACKEND OFFLINE'}</div><div className="avatar">QL</div></div></header><div className="content"><Suspense fallback={<div className="route-loading">Loading workspace…</div>}><Outlet /></Suspense></div><footer>QUANTLAB · LOCAL-FIRST MODEL EVALUATION <span><Wifi size={13} /> INFERENCE NEVER LEAVES THIS DEVICE</span></footer></main>
  </div>
}

export default function App() {
  return <Routes><Route element={<Shell />}><Route path="/" element={<Navigate to="/dashboard" replace />} /><Route path="/dashboard" element={<Dashboard />} /><Route path="/models" element={<Models />} /><Route path="/models/import" element={<ImportModel />} /><Route path="/chat" element={<Chat />} /><Route path="/evaluation" element={<Evaluation />} /><Route path="/comparison" element={<Evaluation />} /><Route path="/benchmark" element={<Benchmark />} /><Route path="/settings" element={<SettingsPage />} /><Route path="*" element={<Navigate to="/dashboard" replace />} /></Route></Routes>
}
