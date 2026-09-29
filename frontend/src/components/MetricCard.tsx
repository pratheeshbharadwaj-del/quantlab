import type { ReactNode } from 'react'

type Props = { label: string; value: ReactNode; detail?: string; icon: ReactNode; accent?: string }
export default function MetricCard({ label, value, detail, icon, accent = 'blue' }: Props) {
  return <section className="metric-card"><div className={`metric-icon ${accent}`}>{icon}</div><div><div className="metric-label">{label}</div><div className="metric-value">{value}</div>{detail && <div className="metric-detail">{detail}</div>}</div></section>
}
