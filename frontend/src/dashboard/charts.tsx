import { useEffect, useRef, useState, type ReactNode, type RefObject } from 'react'

export function useWidth<T extends HTMLElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T | null>(null)
  const [w, setW] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(() => setW(Math.round(el.getBoundingClientRect().width)))
    ro.observe(el)
    setW(Math.round(el.getBoundingClientRect().width))
    return () => ro.disconnect()
  }, [])
  return [ref, w]
}

const exps = (lo: number, hi: number) => {
  const out: number[] = []
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) out.push(e)
  return out
}
const tickLabel = (e: number) => (
  <>
    10<tspan dy={-4} fontSize="8">{String(e).replace('-', '−')}</tspan>
  </>
)

export interface Series {
  id: string
  color: string
  dash?: string
  pts: [number, number][]
  axis?: 'r'
}
export interface DualAxes {
  l: { max: number; fmt: (v: number) => ReactNode; color: string }
  r: { max: number; fmt: (v: number) => ReactNode; color: string }
}
export interface Mark {
  x: number
  y: number
  color: string
  label: string
  axis?: 'r'
}

/** Log-y line chart with a hover read-out. x in data units, y > 0. */
export function LineChart({
  series, marks, xDomain, xTicks, xFmt, yDomain, dual, height = 280, tipX, tipRows,
}: {
  dual?: DualAxes
  series: Series[]
  marks: Mark[]
  xDomain: [number, number]
  xTicks: number[]
  xFmt: (x: number) => string
  yDomain: [number, number]
  height?: number
  tipX: (x: number) => string
  tipRows: (x: number) => { label: string; value: ReactNode; color: string }[]
}) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [hx, setHx] = useState<number | null>(null)
  const W = Math.max(w, 260)
  const m = { l: 48, r: dual ? 52 : 14, t: 14, b: 30 }
  const iw = W - m.l - m.r
  const ih = height - m.t - m.b
  const X = (x: number) => m.l + ((x - xDomain[0]) / (xDomain[1] - xDomain[0])) * iw
  const Y = (y: number, ax?: 'r') =>
    dual ? m.t + (1 - y / (ax === 'r' ? dual.r.max : dual.l.max)) * ih : m.t + (1 - (Math.log10(y) - yDomain[0]) / (yDomain[1] - yDomain[0])) * ih
  const path = (pts: [number, number][], ax?: 'r') => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(y, ax).toFixed(1)}`).join('')
  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const x = xDomain[0] + ((e.clientX - r.left) / r.width) * (xDomain[1] - xDomain[0])
    setHx(Math.min(xDomain[1], Math.max(xDomain[0], x)))
  }
  const rows = hx === null ? [] : tipRows(hx)
  const left = hx === null ? 0 : X(hx)
  return (
    <div className="chart jt-chart" ref={ref} style={{ height }}>
      {w > 0 && (
        <svg width={W} height={height} role="img" aria-label="Line chart on a logarithmic probability axis">
          {dual && [0, 1, 2, 3, 4].map((i) => (
            <g key={i}>
              <line x1={m.l} x2={W - m.r} y1={Y((dual.l.max * i) / 4)} y2={Y((dual.l.max * i) / 4)} className="jt-grid" />
              <text x={m.l - 8} y={Y((dual.l.max * i) / 4) + 3} textAnchor="end" className="jt-tick" style={{ fill: dual.l.color }}>{dual.l.fmt((dual.l.max * i) / 4)}</text>
              <text x={W - m.r + 8} y={Y((dual.l.max * i) / 4) + 3} textAnchor="start" className="jt-tick" style={{ fill: dual.r.color }}>{dual.r.fmt((dual.r.max * i) / 4)}</text>
            </g>
          ))}
          {!dual && exps(yDomain[0], yDomain[1]).map((e) => (
            <g key={e}>
              <line x1={m.l} x2={W - m.r} y1={Y(10 ** e)} y2={Y(10 ** e)} className="jt-grid" />
              <text x={m.l - 8} y={Y(10 ** e) + 3} textAnchor="end" className="jt-tick">
                {tickLabel(e)}
              </text>
            </g>
          ))}
          {xTicks.map((x) => (
            <text key={x} x={X(x)} y={height - 9} textAnchor="middle" className="jt-tick">
              {xFmt(x)}
            </text>
          ))}
          {series.map((s) => (
            <path key={s.id} d={path(s.pts, s.axis)} fill="none" stroke={s.color} strokeWidth={2.2} strokeDasharray={s.dash} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {hx !== null && <line x1={left} x2={left} y1={m.t} y2={m.t + ih} className="jt-cross" />}
          {marks.map((k) => (
            <g key={k.label} transform={`translate(${X(k.x)},${Y(k.y, k.axis)})`}>
              <circle r={8} fill={k.color} opacity={0.2} />
              <circle r={4.5} fill="var(--panel)" stroke={k.color} strokeWidth={2} />
            </g>
          ))}
          <rect x={m.l} y={m.t} width={iw} height={ih} fill="transparent" onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={() => setHx(null)} />
        </svg>
      )}
      {hx !== null && (
        <div className="jt-tip" style={{ left: Math.min(Math.max(left, 90), W - 100), top: 6 }}>
          <b>{tipX(hx)}</b>
          {rows.map((r) => (
            <div key={r.label} className="tip-row">
              <i style={{ background: r.color }} />
              <span>{r.label}</span>
              <em>{r.value}</em>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export interface Bar {
  id: string
  label: string
  value: number
  display: ReactNode
  sub?: ReactNode
  active?: boolean
}
/** Vertical bars. log=true plots log10(value) between floor and the largest value. */
export function VBars({ bars, log, height = 200, floor, onPick }: { bars: Bar[]; log?: boolean; height?: number; floor?: number; onPick?: (id: string) => void }) {
  const hi = Math.max(...bars.map((b) => b.value))
  const lo = floor ?? Math.min(...bars.map((b) => b.value)) / 10
  const frac = (v: number) => (log ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi * 1.15) - Math.log10(lo)) : v / (hi * 1.1))
  return (
    <div className="vbars" style={{ height }} role="list">
      {bars.map((b) => (
        <button key={b.id} type="button" role="listitem" className={`vb${b.active ? ' on' : ''}`} onClick={() => onPick?.(b.id)} disabled={!onPick} aria-pressed={onPick ? b.active : undefined}>
          <span className="vb-val">{b.display}</span>
          <span className="vb-track">
            <span className="vb-fill" style={{ height: `${Math.max(3, Math.min(100, frac(b.value) * 100))}%` }} />
          </span>
          <span className="vb-lab">{b.label}</span>
          {b.sub && <span className="vb-sub">{b.sub}</span>}
        </button>
      ))}
    </div>
  )
}

/** Donut with a centre read-out; arcs are fractions of 1, r is the inset of each ring. */
export function Donut({ size = 150, arcs, center, caption }: { size?: number; arcs: { frac: number; color: string; r: number }[]; center: ReactNode; caption: string }) {
  const c = size / 2
  const sw = 11
  return (
    <div className="donut" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={caption}>
        {arcs.map((a, i) => {
          const rad = c - sw / 2 - a.r
          const L = 2 * Math.PI * rad
          return (
            <g key={i} transform={`rotate(-90 ${c} ${c})`}>
              <circle cx={c} cy={c} r={rad} fill="none" stroke="var(--track)" strokeWidth={sw} />
              <circle cx={c} cy={c} r={rad} fill="none" stroke={a.color} strokeWidth={sw} strokeLinecap="round" strokeDasharray={`${Math.max(0, a.frac) * L} ${L}`} />
            </g>
          )
        })}
      </svg>
      <div className="donut-c">{center}</div>
    </div>
  )
}
