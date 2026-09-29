import { AlertCircle, CheckCircle2 } from 'lucide-react'
export default function Notice({ children, kind = 'error' }: { children: string; kind?: 'error' | 'success' | 'info' }) { return <div className={`notice ${kind}`}>{kind === 'success' ? <CheckCircle2 size={17} /> : <AlertCircle size={17} />}<span>{children}</span></div> }
