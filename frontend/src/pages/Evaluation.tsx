import { useCallback, useEffect, useMemo, useState } from 'react'
import { ArrowDown, ArrowUpRight, BarChart3, Check, FileUp, Gauge, HardDrive, LoaderCircle, MemoryStick, Play, Sparkles } from 'lucide-react'
import PageHeading from '../components/PageHeading'
import Notice from '../components/Notice'
import { api, formatBytes, percent, type DatasetItem, type Model } from '../services/api'

const scoreText = (value: number | null | undefined) => value == null ? 'N/A' : `${(value * 100).toFixed(1)}%`
function Delta({ value, unit = 'pp' }: { value: number | null | undefined; unit?: string }) { if (value == null) return <span className="delta neutral">N/A</span>; const positive = value > 0; return <span className={`delta ${value === 0 ? 'neutral' : positive ? 'positive' : 'negative'}`}>{positive ? <ArrowUpRight size={14} /> : <ArrowDown size={14} />}{Math.abs(value).toFixed(1)} {unit}</span> }
export default function Evaluation() {
  const [models, setModels] = useState<Model[]>([])
  const [dataset, setDataset] = useState<DatasetItem[]>([])
  const [originalId, setOriginalId] = useState('')
  const [quantizedId, setQuantizedId] = useState('')
  const [prompt, setPrompt] = useState('Explain model quantization in simple terms.')
  const [sampleCount, setSampleCount] = useState(3)
  const [result, setResult] = useState<Record<string, any> | null>(null)
  const [busy, setBusy] = useState(false)
  const [uploadingRole, setUploadingRole] = useState<'original' | 'quantized' | null>(null)
  const [pendingOriginal, setPendingOriginal] = useState<Model | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const loaded = useMemo(() => models.filter(m => m.loaded && m.generation_supported), [models])
  const originalCandidates = useMemo(() => loaded.filter(m => m.role === 'original' || m.role === 'library'), [loaded])
  const quantizedCandidates = useMemo(() => loaded.filter(m => m.role === 'quantized' || m.role === 'library'), [loaded])
  const duplicateWeights = Boolean(originalId && quantizedId && loaded.find(m => m.id === originalId)?.content_fingerprint && loaded.find(m => m.id === originalId)?.content_fingerprint === loaded.find(m => m.id === quantizedId)?.content_fingerprint)
  const refreshModels = useCallback(async () => {
    const list = await api.models()
    setModels(list)
    const on = list.filter(m => m.loaded && m.generation_supported)
    setOriginalId(current => current && on.some(m => m.id === current) ? current : (on.find(m => m.role === 'original') ?? on.find(m => m.role !== 'quantized'))?.id ?? '')
    setQuantizedId(current => current && on.some(m => m.id === current) ? current : (on.find(m => m.role === 'quantized') ?? on.find(m => m.role !== 'original' && m.role !== 'library'))?.id ?? '')
  }, [])
  useEffect(() => { void refreshModels().catch(e => setError(e.message)); api.dataset().then(setDataset).catch(e => setError(e.message)); const timer = window.setInterval(() => { void refreshModels().catch(() => undefined) }, 5000); return () => window.clearInterval(timer) }, [refreshModels])
  async function uploadComparisonModel(file: File | undefined, role: 'original' | 'quantized') {
    if (!file) return
    setUploadingRole(role); setError(''); setNotice(''); setResult(null)
    try {
      const model = await api.uploadModel(file, role)
      await refreshModels()
      if (role === 'original') {
        setPendingOriginal(model)
        setNotice(`Original model ${model.name} was imported into the local model directory. Load it before comparing.`)
      } else if (model.loaded) {
        setQuantizedId(model.id)
        setNotice(`Quantized model ${model.name} imported and loaded locally. It is selected for comparison.`)
      } else {
        setError(model.auto_load_error ?? model.compatibility_message)
      }
    } catch (e) { setError(e instanceof Error ? e.message : `Could not import ${role} model`) }
    finally { setUploadingRole(null) }
  }
  async function loadOriginal() {
    if (!pendingOriginal) return
    setBusy(true); setError('')
    try { const loadedModel = await api.loadModel(pendingOriginal.id); await refreshModels(); setOriginalId(loadedModel.id); setPendingOriginal(null); setNotice(`Original model ${loadedModel.name} loaded locally and selected.`) }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not load the original model') }
    finally { setBusy(false) }
  }
  async function uploadDataset(file?: File) { if (!file) return; setError(''); try { const data = await api.importDataset(file); setDataset(data.dataset); setNotice(`Loaded ${data.count} evaluation examples from ${file.name}.`); setResult(null) } catch (e) { setError(e instanceof Error ? e.message : 'Dataset import failed') } }
  async function run() {
    if (!originalId || !quantizedId || originalId === quantizedId) { setError('Select two different loaded models.'); return }
    setBusy(true); setError(''); setNotice(''); setResult(null)
    try { setResult(await api.compare({ original_model_id: originalId, quantized_model_id: quantizedId, dataset: dataset.slice(0, sampleCount), prompt: prompt.trim() || undefined })); setNotice(`Comparison complete for ${Math.min(sampleCount, dataset.length)} evaluation prompts. All values are measured from this run.`) }
    catch (e) { setError(e instanceof Error ? e.message : 'Evaluation failed') } finally { setBusy(false) }
  }
  const original = result?.original?.model
  const quantized = result?.quantized?.model
  const originalEval = result?.original?.evaluation
  const quantizedEval = result?.quantized?.evaluation
  const sizeDelta = original && quantized && original.size_bytes ? (result.size_reduction_percent) : null
  const origSpeed = originalEval?.results?.map((r: any) => r.tokens_per_second).filter((x: number | null) => x != null) ?? []
  const quantSpeed = quantizedEval?.results?.map((r: any) => r.tokens_per_second).filter((x: number | null) => x != null) ?? []
  const speedChange = origSpeed.length && quantSpeed.length ? ((quantSpeed.reduce((a: number, b: number) => a + b, 0) / quantSpeed.length) / (origSpeed.reduce((a: number, b: number) => a + b, 0) / origSpeed.length) - 1) * 100 : null
  return <div className="page-stack">
    <PageHeading eyebrow="ACADEMIC DEMONSTRATION" title="Project Evaluation" description="Run the same dataset and prompt through both models. Review real quality, size and runtime measurements side by side." />
    <div className="evaluation-intro"><div className="eval-intro-icon"><Sparkles size={20} /></div><div><strong>Controlled, same-input comparison</strong><p>Exact Match and token F1 use your expected answers. Output Similarity is a lexical comparison—not an accuracy claim. Unsupported measurements remain N/A.</p></div><span className="dataset-count">{dataset.length} prompts</span></div>
    <section className="panel compare-setup">
      <div className="panel-heading"><div><h3>Configure comparison</h3><p>Upload the original model locally. A quantized model is loaded automatically after a successful import.</p></div><BarChart3 size={19} /></div>
      <div className="comparison-import-actions">
        <label className={`button secondary local-model-upload ${uploadingRole === 'original' ? 'uploading' : ''}`}><FileUp size={16} />{uploadingRole === 'original' ? 'Importing original…' : 'Upload original locally'}<input type="file" accept=".zip,.gguf,.onnx,.tflite,.lite,.safetensors,.bin,.pt,.pth" disabled={uploadingRole !== null || busy} onChange={e => { void uploadComparisonModel(e.target.files?.[0], 'original'); e.currentTarget.value = '' }} /></label>
        <label className={`button secondary local-model-upload ${uploadingRole === 'quantized' ? 'uploading' : ''}`}><FileUp size={16} />{uploadingRole === 'quantized' ? 'Importing and loading…' : 'Import quantized · auto-load'}<input type="file" accept=".zip,.gguf,.onnx,.tflite,.lite,.safetensors,.bin,.pt,.pth" disabled={uploadingRole !== null || busy} onChange={e => { void uploadComparisonModel(e.target.files?.[0], 'quantized'); e.currentTarget.value = '' }} /></label>
        <span className="local-only-hint">Files are sent only to this PC's local backend.</span>
      </div>
      {pendingOriginal && <div className="pending-original"><div><strong>{pendingOriginal.name}</strong><span>{pendingOriginal.format} · {formatBytes(pendingOriginal.size_bytes)} · imported, not loaded</span></div><button className="button primary" disabled={busy || !pendingOriginal.supported} onClick={() => void loadOriginal()}>{busy ? <LoaderCircle className="spin" size={16} /> : <Play size={15} />} Load original model</button></div>}
      <div className="compare-selects"><label><span><i className="model-dot original-dot" />ORIGINAL MODEL · LOADED</span><select value={originalId} onChange={e => { setOriginalId(e.target.value); if (e.target.value === quantizedId) setQuantizedId('') }}><option value="">Choose loaded original…</option>{originalCandidates.filter(m => m.id !== quantizedId).map(m => <option key={m.id} value={m.id}>{m.name} · {m.format}</option>)}</select></label><div className="versus">VS</div><label><span><i className="model-dot quant-dot" />QUANTIZED MODEL · AUTO-LOAD</span><select value={quantizedId} onChange={e => { setQuantizedId(e.target.value); if (e.target.value === originalId) setOriginalId('') }}><option value="">Choose loaded quantized model…</option>{quantizedCandidates.filter(m => m.id !== originalId).map(m => <option key={m.id} value={m.id}>{m.name} · {m.format}</option>)}</select></label></div>
      <div className="eval-bottom-controls"><label className="control-field"><span>Evaluation prompts</span><select value={sampleCount} onChange={e => setSampleCount(Number(e.target.value))}>{[3, 5, 10, 20].map(count => <option key={count} value={count}>{Math.min(count, dataset.length)} prompts</option>)}</select></label><label className="upload-dataset button secondary"><FileUp size={16} /> Import JSON/CSV dataset<input type="file" accept=".json,.csv,application/json,text/csv" onChange={e => void uploadDataset(e.target.files?.[0])} /></label><label className="benchmark-prompt"><span>SAME PROMPT FOR BOTH MODELS</span><textarea value={prompt} rows={2} onChange={e => setPrompt(e.target.value)} /></label><button className="button primary" disabled={busy || uploadingRole !== null || !originalId || !quantizedId || originalId === quantizedId || duplicateWeights} onClick={() => void run()}>{busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}{busy ? 'Running comparison…' : 'Run comparison'}</button></div>{duplicateWeights ? <div className="subtle-hint error-hint">These two imports have identical model weights. Upload the original pretrained file and the quantized model as separate artifacts to compare them.</div> : !originalCandidates.length ? <div className="subtle-hint">No original model is loaded. Use <strong>Upload original locally</strong>, then choose <strong>Load original model</strong>. Quantized imports are not eligible for the original slot.</div> : (loaded.length < 2 || !originalId || !quantizedId) && <div className="subtle-hint">Choose two different loaded models. Upload an original model here, then import the quantized model to auto-load it.</div>}
    </section>
    {error && <Notice>{error}</Notice>}{notice && <Notice kind="success">{notice}</Notice>}{busy && !pendingOriginal && <Notice kind="info">Running the selected local model(s) on the same input. Results will appear here when inference finishes.</Notice>}
    {result && <>
      <div className="comparison-cards"><article className="comparison-card original-card"><div className="comparison-card-label"><i className="model-dot original-dot" />ORIGINAL MODEL</div><h2>{original?.name}</h2><span className="tag">{original?.format} · {original?.quantization ?? 'Quantization N/A'}</span><div className="comparison-stats"><div><HardDrive size={15} /><span>On-disk size</span><strong>{formatBytes(original?.size_bytes)}</strong></div><div><MemoryStick size={15} /><span>RAM before → after load</span><strong>{formatBytes(result.memory?.original_before_bytes)} → {formatBytes(result.memory?.original_after_bytes)}</strong></div><div><Gauge size={15} /><span>Avg. tokens/sec</span><strong>{origSpeed.length ? (origSpeed.reduce((a: number, b: number) => a + b, 0) / origSpeed.length).toFixed(2) : 'N/A'}</strong></div><div><BarChart3 size={15} /><span>Exact Match</span><strong>{scoreText(originalEval?.metrics?.exact_match)}</strong></div><div><Sparkles size={15} /><span>Token F1</span><strong>{scoreText(originalEval?.metrics?.token_f1)}</strong></div><div><ActivityIcon /><span>Perplexity</span><strong>{originalEval?.perplexity == null ? 'N/A – unsupported' : originalEval.perplexity.toFixed(3)}</strong></div></div></article>
      <article className="comparison-card quant-card"><div className="comparison-card-label"><i className="model-dot quant-dot" />QUANTIZED MODEL</div><h2>{quantized?.name}</h2><span className="tag">{quantized?.format} · {quantized?.quantization ?? 'Quantization N/A'}</span><div className="comparison-stats"><div><HardDrive size={15} /><span>On-disk size</span><strong>{formatBytes(quantized?.size_bytes)}</strong></div><div><MemoryStick size={15} /><span>RAM before → after load</span><strong>{formatBytes(result.memory?.quantized_before_bytes)} → {formatBytes(result.memory?.quantized_after_bytes)}</strong></div><div><Gauge size={15} /><span>Avg. tokens/sec</span><strong>{quantSpeed.length ? (quantSpeed.reduce((a: number, b: number) => a + b, 0) / quantSpeed.length).toFixed(2) : 'N/A'}</strong></div><div><BarChart3 size={15} /><span>Exact Match</span><strong>{scoreText(quantizedEval?.metrics?.exact_match)}</strong></div><div><Sparkles size={15} /><span>Token F1</span><strong>{scoreText(quantizedEval?.metrics?.token_f1)}</strong></div><div><ActivityIcon /><span>Perplexity</span><strong>{quantizedEval?.perplexity == null ? 'N/A – unsupported' : quantizedEval.perplexity.toFixed(3)}</strong></div></div></article></div>
      <section className="panel deltas-panel"><div className="panel-heading"><div><h3>Measured change</h3><p>Quantized model relative to original · positive speed means faster</p></div><ArrowUpRight size={18} /></div><div className="delta-grid"><div><small>EXACT MATCH DIFFERENCE</small><strong>{result.accuracy_difference == null ? 'N/A' : `${(result.accuracy_difference * 100).toFixed(1)} pp`}</strong><Delta value={result.accuracy_difference == null ? null : result.accuracy_difference * 100} /></div><div><small>SIZE REDUCTION</small><strong>{sizeDelta == null ? 'N/A' : `${sizeDelta.toFixed(1)}%`}</strong><Delta value={sizeDelta} unit="%" /></div><div><small>RAM INCREASE CHANGE</small><strong>{result.memory?.original_ram_bytes == null || result.memory?.quantized_ram_bytes == null ? 'N/A' : `${((result.memory.original_ram_bytes - result.memory.quantized_ram_bytes) / 1024 ** 2).toFixed(1)} MB`}</strong><Delta value={result.memory?.original_ram_bytes == null || result.memory?.quantized_ram_bytes == null ? null : (result.memory.original_ram_bytes - result.memory.quantized_ram_bytes) / 1024 ** 2} unit="MB" /></div><div><small>INFERENCE SPEED CHANGE</small><strong>{speedChange == null ? 'N/A' : `${speedChange.toFixed(1)}%`}</strong><Delta value={speedChange} unit="%" /></div></div><div className="interpretation">Exact Match difference is shown in percentage points (pp). RAM figures are measured process RSS increases during each model load, not isolated GPU/model allocations. Perplexity is available only on compatible runtimes.</div></section>
      {result.prompt_comparison && <section className="panel output-panel"><div className="panel-heading"><div><h3>Same prompt · output comparison</h3><p>Similarity is lexical Jaccard overlap and is not an accuracy metric.</p></div><span className="similarity-pill">Similarity {percent(result.prompt_comparison.output_similarity)}</span></div><div className="prompt-display"><span>PROMPT</span><p>{result.prompt_comparison.prompt}</p></div><div className="output-columns"><div><div className="output-heading"><i className="model-dot original-dot" />ORIGINAL OUTPUT</div><div className="output-text">{result.prompt_comparison.original_output || 'No text returned.'}</div></div><div><div className="output-heading"><i className="model-dot quant-dot" />QUANTIZED OUTPUT</div><div className="output-text">{result.prompt_comparison.quantized_output || 'No text returned.'}</div></div></div></section>}
      <section className="panel sample-panel"><div className="panel-heading"><div><h3>Evaluation sample results</h3><p>Per-question responses compared with expected answers</p></div><span className="tag">{dataset.length} examples</span></div><div className="sample-table"><div className="sample-header"><span>QUESTION</span><span>ORIGINAL</span><span>QUANTIZED</span></div>{(originalEval?.results ?? []).slice(0, 8).map((row: any, index: number) => <div className="sample-row" key={`${row.id}-${index}`}><strong>{row.question}</strong><span>{row.output}</span><span>{quantizedEval?.results?.[index]?.output}</span></div>)}</div></section>
    </>}
    <div className="fine-print"><Check size={14} /> No model is labeled more accurate unless the provided expected-answer metrics demonstrate it.</div>
  </div>
}
function ActivityIcon() { return <Gauge size={15} /> }
