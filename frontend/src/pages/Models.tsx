import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Boxes, Check, ChevronRight, CircleAlert, Cpu, FolderOpen, HardDrive, LoaderCircle, Plus, RefreshCw, Trash2 } from 'lucide-react'
import PageHeading from '../components/PageHeading'
import Notice from '../components/Notice'
import { api, formatBytes, type Model } from '../services/api'

export default function Models() {
  const [models, setModels] = useState<Model[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState<string | null>(null)
  const refresh = useCallback(() => api.models().then(setModels).catch(e => setError(e.message)), [])
  useEffect(() => { void refresh() }, [refresh])
  async function action(id: string, type: 'load' | 'unload') {
    setBusy(id); setError('')
    try { if (type === 'load') await api.loadModel(id); else await api.unloadModel(id); await refresh() } catch (e) { setError(e instanceof Error ? e.message : 'Action failed') } finally { setBusy(null) }
  }
  return <div className="page-stack">
    <PageHeading eyebrow="MODEL LIBRARY" title="Your local models" description="Import local weights, check runtime compatibility, and manage what is loaded in memory." action={<Link className="button primary" to="/models/import"><Plus size={17} /> Import model</Link>} />
    {error && <Notice>{error}</Notice>}
    <div className="info-strip"><FolderOpen size={17} /><span>Models are discovered from the backend's configured local models directory. Weights are never uploaded.</span><button className="icon-button" onClick={() => void refresh()} aria-label="Refresh models"><RefreshCw size={16} /></button></div>
    {models.length === 0 ? <section className="empty-card"><div className="empty-icon"><Boxes size={24} /></div><h2>Your model workspace is ready</h2><p>Add a local model directory or model file to begin. Start with a GGUF file or a complete Transformers folder.</p><Link className="button primary" to="/models/import"><Plus size={17} /> Import your first model</Link></section> : <div className="model-grid">{models.map(model => <article className="model-card" key={model.id}><div className="model-card-top"><div className="model-type-icon"><Cpu size={19} /></div><span className={`status-chip ${model.loaded ? 'loaded' : model.supported ? 'ready' : 'unsupported'}`}><i />{model.loaded ? 'Loaded' : model.supported ? 'Runtime ready' : 'Unavailable'}</span></div><h2>{model.name}</h2><p className="model-arch">{model.architecture ?? 'Architecture not detected'}</p><div className="tag-row"><span className="tag">{model.format}</span><span className="tag">{model.quantization ?? 'Quantization N/A'}</span></div><div className="model-facts"><div><HardDrive size={15} /><span>File size</span><strong>{formatBytes(model.size_bytes)}</strong></div><div><Cpu size={15} /><span>Runtime</span><strong>{model.runtime}</strong></div></div>{!model.supported && <div className="compat-note"><CircleAlert size={15} />{model.compatibility_message}</div>}{model.loaded && model.load_seconds != null && <div className="loaded-time"><Check size={14} /> Ready in {model.load_seconds.toFixed(2)}s · running locally</div>}<button className={`button ${model.loaded ? 'secondary' : 'primary'} full-button`} disabled={busy === model.id || (!model.loaded && !model.supported)} onClick={() => void action(model.id, model.loaded ? 'unload' : 'load')}>{busy === model.id ? <LoaderCircle className="spin" size={16} /> : model.loaded ? <Trash2 size={16} /> : <ChevronRight size={17} />}{model.loaded ? 'Unload model' : 'Load model'}</button></article>)}</div>}
  </div>
}
