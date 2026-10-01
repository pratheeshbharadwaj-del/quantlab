// Resolution order: runtime-injected config (set by docker-entrypoint.sh in
// production containers, see frontend/public/config.js) > build-time env var
// (VITE_API_URL, useful for `vite build` outside Docker) > local dev default.
const runtimeApiUrl = (window as unknown as { __QNT_API_URL__?: string }).__QNT_API_URL__
const API = (runtimeApiUrl && runtimeApiUrl.length > 0 ? runtimeApiUrl : undefined) ?? import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail ?? `Request failed (${response.status})`)
  return body as T
}

export type Model = { id: string; name: string; format: string; architecture: string | null; parameter_count: number | null; quantization: string | null; size_bytes: number; content_fingerprint?: string | null; chat_template_available?: boolean; chat_format_note?: string | null; runtime: string; role?: 'original' | 'quantized' | 'library'; auto_load_error?: string; required_packages?: string[]; missing_runtime_packages?: string[]; runtime_installed: boolean; supported: boolean; generation_supported: boolean; compatibility_message: string; loaded?: boolean; path: string; load_seconds?: number; runtime_metadata?: Record<string, unknown> }
export type DatasetItem = { id?: string | number; question: string; expected_answer?: string }
export type EvalResult = { id: string; model_id: string; model_name: string; sample_count: number; metrics: { exact_match: number | null; token_f1: number | null; accuracy?: number | null }; perplexity: number | null; average_generation_time: number; results: Array<{ question: string; expected_answer?: string; output: string }> }
export type InferenceDevice = 'auto' | 'cpu' | 'gpu'
export type SystemStatus = { cpu_percent: number; cpu_cores: number | null; ram_total_bytes: number; ram_available_bytes: number; ram_used_bytes: number; gpu: { available: boolean; name: string | null; memory_total_bytes: number | null; memory_used_bytes: number | null; supported_runtimes: string[] }; inference_device: InferenceDevice; operating_system: string; python_version: string; backend_runtime: string; process_rss_bytes: number }
export type LocalModelEntry = { name: string; path: string; is_directory: boolean; size_bytes: number | null }

export const api = {
  health: () => request<{ status: string; mode: string; model_directory: string }>('/health'),
  models: () => request<Model[]>('/models'),
  browseModelFiles: (path = '') => request<LocalModelEntry[]>(`/models/files${path ? `?path=${encodeURIComponent(path)}` : ''}`),
  importModel: (path: string, format_hint = 'AUTO', role: 'original' | 'quantized' | 'library' = 'library') => request<Model>('/models/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path, format_hint, role }) }),
  uploadModel: async (file: File, role: 'original' | 'quantized') => {
    const form = new FormData(); form.append('file', file); form.append('role', role)
    return request<Model>('/models/upload', { method: 'POST', body: form })
  },
  loadModel: (model_id: string) => request<Model>('/models/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_id }) }),
  unloadModel: (model_id?: string) => request<{ unloaded: string[] }>('/models/unload', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(model_id ? { model_id } : {}) }),
  system: () => request<SystemStatus>('/system/status'),
  setInferenceDevice: (device: InferenceDevice) => request<{ inference_device: InferenceDevice }>('/system/device', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ device }) }),
  dataset: () => request<DatasetItem[]>('/evaluation/dataset'),
  importDataset: async (file: File) => {
    const form = new FormData(); form.append('file', file)
    return request<{ dataset: DatasetItem[]; count: number }>('/evaluation/dataset/import', { method: 'POST', body: form })
  },
  generateStream: async (payload: { model_id: string; prompt?: string; messages?: Array<{ role: string; content: string }> }, onToken: (text: string) => void, signal?: AbortSignal) => {
    const response = await fetch(`${API}/generate/stream`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal })
    if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail ?? `Request failed (${response.status})`) }
    if (!response.body) throw new Error('Streaming response body is unavailable in this browser.')
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let completed: { generation_time: number; tokens_generated: number | null; tokens_per_second: number | null } | null = null
    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const event = JSON.parse(line.slice(6)) as { type: string; text?: string; message?: string; generation_time?: number; tokens_generated?: number | null; tokens_per_second?: number | null }
        if (event.type === 'token' && event.text) onToken(event.text)
        if (event.type === 'error') throw new Error(event.message ?? 'Generation failed')
        if (event.type === 'done') completed = { generation_time: event.generation_time ?? 0, tokens_generated: event.tokens_generated ?? null, tokens_per_second: event.tokens_per_second ?? null }
      }
      if (done) break
    }
    if (!completed) throw new Error('Generation ended without a completion event.')
    return completed
  },
  compare: (payload: { original_model_id: string; quantized_model_id: string; dataset?: DatasetItem[]; prompt?: string }) => request<Record<string, any>>('/evaluation/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  benchmark: (payload: { model_id: string; prompt: string; runs: number }) => request<Record<string, any>>('/benchmark/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
}

export const formatBytes = (value?: number | null) => value == null ? 'N/A' : value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(2)} GB` : value >= 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(1)} MB` : `${(value / 1024).toFixed(1)} KB`
export const percent = (value?: number | null) => value == null ? 'N/A' : `${(value * 100).toFixed(1)}%`
