import { useEffect, useState } from 'react'
import { ArrowLeft, ArrowRight, Check, CircleHelp, Cpu, FileSearch, FolderOpen, HardDrive, LoaderCircle, ScanSearch, ShieldCheck, Upload } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'
import PageHeading from '../components/PageHeading'
import Notice from '../components/Notice'
import { api, formatBytes, type LocalModelEntry, type Model } from '../services/api'

const formats = ['AUTO', 'GGUF', 'TRANSFORMERS', 'SAFETENSORS', 'PYTORCH', 'ONNX', 'TFLITE']
export default function ImportModel() {
  const [searchParams] = useSearchParams()
  const role = searchParams.get('role') === 'quantized' ? 'quantized' : searchParams.get('role') === 'original' ? 'original' : 'library'
  const [path, setPath] = useState('')
  const [format, setFormat] = useState('AUTO')
  const [model, setModel] = useState<Model | null>(null)
  const [localEntries, setLocalEntries] = useState<LocalModelEntry[]>([])
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  useEffect(() => { api.browseModelFiles().then(setLocalEntries).catch(e => setError(e.message)) }, [])
  async function inspect() {
    setBusy(true); setError(''); setMessage(''); setModel(null)
    try { const found = await api.importModel(path.trim(), format, role); setModel(found); setMessage(role === 'quantized' && found.loaded ? 'Quantized model imported and loaded locally.' : role === 'quantized' && found.auto_load_error ? `Imported, but automatic load failed: ${found.auto_load_error}` : 'Model files inspected locally. No model weights left this device.') }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not inspect model') }
    finally { setBusy(false) }
  }
  async function load() {
    if (!model) return
    setBusy(true); setError('')
    try { await api.loadModel(model.id); setModel({ ...model, loaded: true }); setMessage('Model loaded successfully on this device.') }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not load model') }
    finally { setBusy(false) }
  }
  return <div className="page-stack">
    <PageHeading eyebrow="LOCAL IMPORT" title={role === 'original' ? 'Import original model' : role === 'quantized' ? 'Import quantized model' : 'Bring a model into QuantLab'} description={role === 'quantized' ? 'Import from this PC. QuantLab automatically loads a supported quantized model after import.' : 'Select a model already stored on this PC. Files are inspected and loaded only by the local backend.'} />
    <div className="import-layout"><section className="panel import-panel"><div className="step-line"><span className="step-number">01</span><div><h2>Choose model location</h2><p>Place files under the backend local models directory, then enter a path relative to it.</p></div></div>
      <div className="local-file-browser"><div className="browser-title"><FolderOpen size={15} /><span>AVAILABLE IN LOCAL MODELS FOLDER</span><button type="button" onClick={() => void api.browseModelFiles().then(setLocalEntries).catch(e => setError(e.message))}>Refresh</button></div>{localEntries.length ? <div className="local-file-list">{localEntries.map(entry => <div className="local-file-row" key={entry.path}><div className="local-file-icon">{entry.is_directory ? <FolderOpen size={15} /> : <FileSearch size={15} />}</div><div className="local-file-name"><strong>{entry.name}</strong><small>{entry.is_directory ? 'Folder' : formatBytes(entry.size_bytes)}</small></div><button type="button" onClick={() => { setPath(entry.path); setError(''); setModel(null) }}>Select</button></div>)}</div> : <p className="browser-empty">No model files or folders found. Copy a model or archive into <code>backend/models/</code>.</p>}</div>
      <div className="drop-zone"><div className="upload-badge"><Upload size={20} /></div><strong>Local files stay on your machine</strong><span>GGUF · ONNX/TFLite · Transformers folder · ZIP archive</span><div className="path-input-wrap"><HardDrive size={16} /><input value={path} onChange={e => setPath(e.target.value)} placeholder="e.g. D__qnt.zip, model.gguf, or model-folder/" /></div><small className="field-hint">Model root: backend/models/ · ZIP model archives are safely unpacked locally. Absolute paths outside this directory are blocked.</small></div>
      <div className="form-field"><label htmlFor="format">FORMAT DETECTION</label><select id="format" value={format} onChange={e => setFormat(e.target.value)}>{formats.map(item => <option key={item} value={item}>{item === 'AUTO' ? 'Automatic detection (recommended)' : item}</option>)}</select><small className="field-hint">The format is detected from model files; a chosen type is checked against detection, never used to force a runtime.</small></div>
      <button className="button primary" disabled={busy || !path.trim()} onClick={() => void inspect()}>{busy ? <LoaderCircle className="spin" size={16} /> : <ScanSearch size={16} />} Inspect model</button>
      <div className="security-note"><ShieldCheck size={15} /> Path access is restricted to the configured local model directory.</div>
    </section><aside className="panel import-aside"><div className="aside-icon"><CircleHelp size={20} /></div><h3>How local import works</h3><ol><li>Copy or place your model file, folder, or ZIP archive inside <code>backend/models/</code>.</li><li>Enter the relative path, including the archive name (for example <code>D__qnt.zip</code>).</li><li>QuantLab safely unpacks ZIP contents locally and detects model configuration, weights, size and quantization metadata.</li><li>Load supported models to run inference and evaluation. Missing optional runtime packages are reported explicitly.</li></ol><div className="supported-mini"><span>ADAPTERS</span><div>GGUF · Transformers · ONNX · TFLite · PyTorch</div><small>Text generation is architecture/runtime dependent.</small></div></aside></div>
    {error && <Notice>{error}</Notice>}{message && <Notice kind="success">{message}</Notice>}
    {model && <section className="panel detected-panel"><div className="detected-heading"><div><div className="eyebrow">INSPECTION RESULT</div><h2>{model.name}</h2></div><span className={`status-chip ${model.supported ? 'ready' : 'unsupported'}`}><i />{model.loaded ? 'Loaded locally' : model.supported ? 'Runtime available' : 'Runtime unavailable'}</span></div><div className="detection-grid"><div><Check size={16} /><small>Format</small><strong>{model.format}</strong></div><div><Cpu size={16} /><small>Architecture</small><strong>{model.architecture ?? 'N/A'}</strong></div><div><HardDrive size={16} /><small>Model size</small><strong>{formatBytes(model.size_bytes)}</strong></div><div><FileSearch size={16} /><small>Quantization</small><strong>{model.quantization ?? 'Not detected'}</strong></div><div><Cpu size={16} /><small>Runtime</small><strong>{model.runtime}</strong></div><div><ArrowRight size={16} /><small>Compatibility</small><strong>{model.auto_load_error ?? model.compatibility_message}</strong></div></div>{model.supported && !model.loaded && <button className="button primary" disabled={busy} onClick={() => void load()}>{busy ? <LoaderCircle className="spin" size={16} /> : <ArrowRight size={16} />} Load model</button>}{model.loaded && <Link className="button primary" to={role === 'quantized' || role === 'original' ? '/evaluation' : '/chat'}>{role === 'quantized' || role === 'original' ? 'Return to Project Evaluation' : 'Start chatting'} <ArrowRight size={16} /></Link>}{!model.supported && <p className="warning-copy">{model.auto_load_error ?? model.compatibility_message}. This model declares {model.quantization ?? 'no quantization metadata'}; install the listed packages to make a runtime load attempt. Runtime/platform compatibility will still be verified at load time.</p>}</section>}
    <Link className="back-link" to="/models"><ArrowLeft size={15} /> Back to model library</Link>
  </div>
}
