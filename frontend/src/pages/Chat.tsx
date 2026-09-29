import { useEffect, useRef, useState } from 'react'
import { Bot, Check, ChevronDown, Clock3, Cpu, MessageSquarePlus, Send, Square, Zap } from 'lucide-react'
import { Link } from 'react-router-dom'
import PageHeading from '../components/PageHeading'
import Notice from '../components/Notice'
import { api, type Model } from '../services/api'

type Message = { role: 'user' | 'assistant'; content: string; seconds?: number; tokens?: number | null; speed?: number | null }
export default function Chat() {
  const [models, setModels] = useState<Model[]>([])
  const [modelId, setModelId] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [prompt, setPrompt] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const abort = useRef<AbortController | null>(null)
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => { api.models().then(list => { const active = list.filter(m => m.loaded && m.generation_supported); setModels(active); setModelId(active[0]?.id ?? '') }).catch(e => setError(e.message)) }, [])
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy])
  const selected = models.find(m => m.id === modelId)
  async function send() {
    const current = prompt.trim()
    if (!current || !modelId || busy) return
    setError(''); setPrompt(''); setMessages(prev => [...prev, { role: 'user', content: current }, { role: 'assistant', content: '' }]); setBusy(true)
    const controller = new AbortController(); abort.current = controller
    const history = messages.map(m => ({ role: m.role, content: m.content })).filter(m => m.content.trim())
    const chatMessages = [...history, { role: 'user', content: current }]
    const request = selected?.chat_template_available
      ? { model_id: modelId, messages: chatMessages }
      : { model_id: modelId, prompt: `Question: ${current}\nAnswer:` }
    try {
      const response = await api.generateStream(request, token => setMessages(prev => prev.map((message, index) => index === prev.length - 1 ? { ...message, content: message.content + token } : message)), controller.signal)
      setMessages(prev => prev.map((message, index) => index === prev.length - 1 ? { ...message, seconds: response.generation_time, tokens: response.tokens_generated, speed: response.tokens_per_second } : message))
    }
    catch (e) { setMessages(prev => prev.filter((message, index) => !(index === prev.length - 1 && message.role === 'assistant' && !message.content))); if (!(e instanceof DOMException && e.name === 'AbortError')) setError(e instanceof Error ? e.message : 'Generation failed') }
    finally { setBusy(false); abort.current = null }
  }
  function cancel() { abort.current?.abort() }
  return <div className="page-stack chat-page">
    <PageHeading eyebrow="LOCAL TEXT GENERATION" title="Inference lab" description="Chat directly with a model running on your machine. Prompts and responses stay in this local session." action={<button className="button secondary" onClick={() => { setMessages([]); setError('') }}><MessageSquarePlus size={16} /> New conversation</button>} />
    {error && <Notice>{error}</Notice>}
    {!models.length ? <section className="empty-card chat-empty"><div className="empty-icon"><Bot size={24} /></div><h2>No model loaded</h2><p>Load a locally supported text-generation model before opening the inference lab.</p><Link className="button primary" to="/models">Go to models <ChevronDown size={16} /></Link></section> : <div className="chat-layout"><section className="panel chat-panel"><div className="chat-toolbar"><label className="chat-model-select"><Cpu size={16} /><select value={modelId} onChange={e => setModelId(e.target.value)}>{models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select><ChevronDown size={14} /></label><div className="chat-ready"><span className="health-dot" />LOCAL INFERENCE</div></div>
      <div className="chat-messages">{messages.length === 0 ? <div className="chat-welcome"><div className="welcome-mark"><Bot size={23} /></div><h2>Start a local conversation</h2><p>Ask a question to see the selected model's real output and generation measurements.</p><div className="suggestion-row">{['Explain quantization simply', 'What affects tokens per second?', 'How does perplexity work?'].map(text => <button key={text} onClick={() => setPrompt(text)}>{text}<Send size={13} /></button>)}</div></div> : messages.map((message, index) => <div className={`message-row ${message.role}`} key={index}><div className={`message-avatar ${message.role}`}>{message.role === 'user' ? 'YOU' : <Bot size={16} />}</div><div className="message-content"><span className="message-role">{message.role === 'user' ? 'YOU' : selected?.name ?? 'MODEL'}</span><div className="message-bubble">{message.content}</div>{message.role === 'assistant' && <div className="response-stats"><span><Clock3 size={13} />{message.seconds?.toFixed(2)}s</span><span><Zap size={13} />{message.speed == null ? 'N/A' : `${message.speed.toFixed(1)} tok/s`}</span><span><Check size={13} />{message.tokens == null ? 'Token count N/A' : `${message.tokens} tokens`}</span></div>}</div></div>)}{busy && <div className="message-row assistant"><div className="message-avatar assistant"><Bot size={16} /></div><div className="message-content"><span className="message-role">{selected?.name}</span><div className="message-bubble typing"><i /><i /><i /></div></div></div>}<div ref={end} /></div>
      <div className="composer-area"><div className="composer"><textarea value={prompt} onChange={e => setPrompt(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send() } }} placeholder="Message your local model…" rows={2} /><div className="composer-bottom"><span>Enter to send · Shift+Enter for new line</span>{busy ? <button className="send-button stop" onClick={cancel}><Square size={14} fill="currentColor" /> Stop</button> : <button className="send-button" disabled={!prompt.trim()} onClick={() => void send()}><Send size={15} /> Send</button>}</div></div><div className="chat-disclaimer">Model responses may be incorrect. Inference runs on your local device.</div></div></section>
      <aside className="panel chat-settings"><div className="panel-heading"><div><h3>Active model</h3><p>Generation uses fixed evaluation settings</p></div><Cpu size={16} /></div><div className="model-info-box"><div className="eyebrow">ACTIVE MODEL</div><strong>{selected?.name}</strong><div className="info-pair"><span>ARCHITECTURE</span><b>{selected?.architecture ?? 'N/A'}</b></div><div className="info-pair"><span>FORMAT</span><b>{selected?.format}</b></div><div className="info-pair"><span>QUANTIZATION</span><b>{selected?.quantization ?? 'N/A'}</b></div><div className="info-pair"><span>CHAT TEMPLATE</span><b>{selected?.chat_template_available ? 'Available' : 'Not detected'}</b></div></div>{selected && !selected.chat_template_available && <Notice kind="info">This model has no chat template. It appears to be a base text-completion model, not a chat/instruction-tuned assistant; it may continue or repeat text instead of answering reliably. Quantization does not add instruction-following capability. Use a locally imported instruction-tuned model for Q&amp;A.</Notice>}<div className="stream-note">GGUF streams token-by-token and can be interrupted. Other adapters may return a complete response after generation.</div></aside></div>}
  </div>
}
