import type { ReactNode } from 'react'

/** Pill segmented control (single choice). `block` stretches it to the container. */
export function Segmented<T extends string>({ value, options, onChange, label, block, disabled }: {
  value: T; options: { id: T; label: ReactNode; disabled?: boolean }[]; onChange: (v: T) => void; label: string; block?: boolean; disabled?: boolean
}) {
  return (
    <div className={`jt-seg${block ? ' block' : ''}`} role="group" aria-label={label}>
      {options.map((o) => (
        <button type="button" key={o.id} className={o.id === value ? 'on' : ''} aria-pressed={o.id === value} disabled={disabled || o.disabled} onClick={() => onChange(o.id)}>{o.label}</button>
      ))}
    </div>
  )
}

/** Range slider with a large serif read-out and optional tick marks. */
export function Slider({ id, label, min, max, step = 0.001, value, onChange, marks = [], text }: {
  id: string; label: string; min: number; max: number; step?: number; value: number; onChange: (v: number) => void; marks?: { at: number; t: string }[]; text: ReactNode
}) {
  const f = (v: number) => `${(100 * (v - min)) / (max - min)}%`
  return (
    <div className="jt-slider">
      <div className="jt-slider-h"><label htmlFor={id}>{label}</label><b>{text}</b></div>
      <div className="jt-slider-box" style={{ ['--p' as string]: f(value) }}>
        <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
        {marks.map((m) => <span key={m.t} className="jt-slider-mark" style={{ left: f(m.at) }}><i />{m.t}</span>)}
      </div>
    </div>
  )
}
