import { useMemo } from 'react'
import { answers, type Grid } from '../lib/model'
import type { MapData, Pool, Solution, Story } from './data'
import { pctText } from './format'

export const C = { teal: '#46d9c4', green: '#9be37f', terra: '#ff7a4d', grey: '#7d8f93', ink: '#e8f3f1' }

/** Both-bonds-wiped-out probability against the earthquake shared share, at the current rain share. */
export function Spark({ grid, eq, rain, brief, data }: { grid: Grid; eq: number; rain: number; brief: number; data: number }) {
  const W = 260, H = 62
  const pts = useMemo(() => {
    const xs = grid.eq_axis
    return xs.map((e) => ({ e, p: answers(grid, e, rain).jointExhaustion }))
  }, [grid, rain])
  const lo = Math.log10(Math.min(...pts.map((p) => p.p))), hi = Math.log10(Math.max(...pts.map((p) => p.p)))
  const x0 = grid.eq_axis[0], x1 = grid.eq_axis[grid.eq_axis.length - 1]
  const X = (e: number) => 4 + ((e - x0) / (x1 - x0)) * (W - 8)
  const Y = (p: number) => 6 + (1 - (Math.log10(p) - lo) / (hi - lo || 1)) * (H - 14)
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.e).toFixed(1)},${Y(p.p).toFixed(1)}`).join('')
  const cur = answers(grid, eq, rain).jointExhaustion
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Probability that both bonds are wiped out, against the earthquake shared share">
      <defs>
        <linearGradient id="sp" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={C.terra} stopOpacity="0.35" /><stop offset="1" stopColor={C.terra} stopOpacity="0" /></linearGradient>
      </defs>
      {[brief, data].map((m, i) => (
        <g key={i}><line x1={X(m)} x2={X(m)} y1="4" y2={H - 12} className="sp-mark" /><text x={X(m)} y={H - 1} textAnchor="middle" className="sp-t">{i ? 'records' : 'brief'}</text></g>
      ))}
      <path d={`${d}L${X(x1)},${H - 12}L${X(x0)},${H - 12}Z`} fill="url(#sp)" />
      <path d={d} className="sp-line" />
      <circle cx={X(eq)} cy={Y(cur)} r="8" className="sp-halo" />
      <circle cx={X(eq)} cy={Y(cur)} r="3.4" className="sp-dot" />
    </svg>
  )
}

export function Gauge({ value }: { value: number }) {
  const R = 40, circ = 2 * Math.PI * R, arc = circ * 0.75
  return (
    <svg className="gauge" viewBox="0 0 110 110" role="img" aria-label={`Shared pool pays ${pctText(value)} of expected claims`}>
      <g transform="rotate(135 55 55)">
        <circle cx="55" cy="55" r={R} className="g-track" strokeDasharray={`${arc} ${circ}`} />
        <circle cx="55" cy="55" r={R} className="g-val" strokeDasharray={`${arc * value} ${circ}`} />
      </g>
      <text x="55" y="58" textAnchor="middle" className="g-num">{(value * 100).toFixed(1)}<tspan className="g-pc">%</tspan></text>
    </svg>
  )
}

function logBar(v: number, lo: number, hi: number) {
  return Math.min(1, Math.max(0.02, (Math.log10(v) - lo) / (hi - lo)))
}

export function PoolBars({ pools, sel }: { pools: Pool[]; sel: string }) {
  const lo = Math.log10(Math.min(...pools.map((p) => p.seconds))) - 0.4, hi = Math.log10(Math.max(...pools.map((p) => p.seconds)))
  return (
    <div className="pbars">
      {pools.map((p) => (
        <div key={p.id} className={p.id === sel ? 'pb on' : 'pb'}>
          <span className="pb-l">{p.n}</span>
          <span className="pb-track"><i style={{ width: `${100 * logBar(p.seconds, lo, hi)}%` }} /></span>
        </div>
      ))}
      <div className="pb-cap">exact solve time, log scale</div>
    </div>
  )
}

/** Dependence graph of the selected pool: real links for the CAREC pool, the Kyrgyz Republic and Tajikistan pair, node count for the rest. */
export function NodeRing({ pool, sol }: { pool: Pool; sol: Solution }) {
  const n = pool.n
  const S = 118, c = S / 2, r = S / 2 - 12
  const names = pool.id === 'carec' ? sol.carec_order : pool.id === 'pair' ? ['KG', 'TJ'] : null
  const pos = (i: number) => {
    const a = (i / n) * 2 * Math.PI - Math.PI / 2
    return [c + (n === 2 ? (i ? 1 : -1) * r * 0.7 : r * Math.cos(a)), c + (n === 2 ? 0 : r * Math.sin(a))]
  }
  const links: [number, number][] = []
  if (pool.id === 'carec') for (const [a, b] of sol.carec_edges) links.push([sol.carec_order.indexOf(a), sol.carec_order.indexOf(b)])
  else if (pool.id === 'pair') links.push([0, 1])
  return (
    <svg className="ring" viewBox={`0 0 ${S} ${S}`} role="img" aria-label={`${pool.n} countries, ${pool.edges} shared-event links`}>
      {pool.id === 'ap' && <circle cx={c} cy={c} r={r} className="rg-guide" />}
      {links.map(([a, b], i) => { const [x1, y1] = pos(a), [x2, y2] = pos(b); return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} className={pool.id === 'pair' ? 'rg-l hot' : 'rg-l'} /> })}
      {Array.from({ length: n }, (_, i) => {
        const [x, y] = pos(i)
        const hot = pool.id === 'pair' || (names && (names[i] === 'KG' || names[i] === 'TJ'))
        return <g key={i}><circle cx={x} cy={y} r={n > 12 ? 2.6 : 4} className={hot ? 'rg-n hot' : 'rg-n'} />{names && n <= 10 && <text x={x} y={y + (y < c ? -7 : 11)} textAnchor="middle" className="rg-t">{names[i]}</text>}</g>
      })}
    </svg>
  )
}

const TARGET_LABEL: Record<string, string> = {
  joint_exhaustion: 'Both bonds exhausted', joint_attachment: 'Both bonds pay', exceedance: 'Pool loss above US$80m',
  stop_loss: 'Pool stop-loss', expected_loss: 'Expected loss',
}
export const TARGETS: [string, string][] = Object.entries(TARGET_LABEL)

export function ErrBars({ sol, scenario }: { sol: Solution; scenario: string }) {
  const cells = sol.e1[scenario] ?? sol.e1['adb-base']
  const vals = Object.values(cells).flatMap((c) => [c.mc, c.rdpo])
  const lo = Math.log10(Math.min(...vals)) - 0.2, hi = Math.log10(Math.max(...vals)) + 0.05
  return (
    <div className="ebars">
      {TARGETS.map(([k, name]) => {
        const c = cells[k]
        if (!c) return null
        return (
          <div className="eb" key={k}>
            <div className="eb-h"><span>{name}</span><b>{'×'}{(c.mc / c.rdpo).toFixed(c.mc / c.rdpo >= 10 ? 0 : 1)}</b></div>
            <div className="eb-t"><i className="mc" style={{ width: `${100 * logBar(c.mc, lo, hi)}%` }} /><em>{pctText(c.mc, 2)}</em></div>
            <div className="eb-t"><i className="rd" style={{ width: `${100 * logBar(c.rdpo, lo, hi)}%` }} /><em>{pctText(c.rdpo, 2)}</em></div>
          </div>
        )
      })}
      <div className="eb-key"><span><i className="mc" />plain Monte Carlo</span><span><i className="rd" />RDPO</span><span>relative error, {sol.budget.toLocaleString('en-US')} paths</span></div>
    </div>
  )
}

export function QChart({ story, sel }: { story: Story; sel: number | null }) {
  const W = 290, H = 118
  const q = story.quantum
  const all = q.flatMap((r) => [r.s_star, ...(r.ci ?? [])])
  const lo = Math.floor(Math.log10(Math.min(...all))), hi = Math.ceil(Math.log10(Math.max(...all)))
  const X = (i: number) => 28 + (i / (q.length - 1)) * (W - 44)
  const Y = (v: number) => 10 + (1 - (Math.log10(v) - lo) / (hi - lo)) * (H - 36)
  const ticks: number[] = []
  for (let e = lo; e <= hi; e += 2) ticks.push(e)
  return (
    <svg className="qchart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Quantum speed-up needed to break even, by number of countries">
      {ticks.map((e) => <g key={e}><line x1="26" x2={W - 6} y1={Y(Math.pow(10, e))} y2={Y(Math.pow(10, e))} className="q-grid" /><text x="22" y={Y(Math.pow(10, e)) + 3} textAnchor="end" className="q-t">10<tspan dy="-3" fontSize="7">{e}</tspan></text></g>)}
      <path d={q.map((r, i) => `${i ? 'L' : 'M'}${X(i)},${Y(r.s_star)}`).join('')} className="q-line" />
      {q.map((r, i) => (
        <g key={r.n} className={sel === r.n ? 'q-pt on' : 'q-pt'}>
          {r.ci && <line x1={X(i)} x2={X(i)} y1={Y(r.ci[0])} y2={Y(r.ci[1])} className="q-ci" />}
          <circle cx={X(i)} cy={Y(r.s_star)} r={sel === r.n ? 4.5 : 3} />
          <text x={X(i)} y={H - 6} textAnchor="middle" className="q-t">{r.n}</text>
        </g>
      ))}
    </svg>
  )
}

export function Timeline({ map, onHover, hover }: { map: MapData; onHover: (i: number | null) => void; hover: number | null }) {
  const ev = useMemo(() => {
    const byYear = new Map<number, number>()
    const sorted = map.quakes.map((q, i) => ({ q, i })).sort((a, b) => a.q[3] - b.q[3] || a.q[2] - b.q[2])
    const count = new Map<number, number>()
    sorted.forEach(({ q }) => count.set(q[3], (count.get(q[3]) ?? 0) + 1))
    return sorted.map(({ q, i }) => {
      const k = byYear.get(q[3]) ?? 0
      byYear.set(q[3], k + 1)
      return { i, mag: q[2], year: q[3], both: q[4] === 1, t: q[3] + (k + 0.5) / (count.get(q[3]) ?? 1) }
    })
  }, [map])
  const y0 = 1965, y1 = 2017
  const x = (t: number) => `${(100 * (t - y0)) / (y1 - y0)}%`
  const mags = ev.map((e) => e.mag)
  const mlo = Math.min(...mags), mhi = Math.max(...mags)
  const h = (m: number) => 18 + ((m - mlo) / (mhi - mlo)) * 62
  const hv = hover !== null ? ev.find((e) => e.i === hover) : null
  const nBoth = ev.filter((e) => e.both).length
  const ticks = [1970, 1980, 1990, 2000, 2010]
  return (
    <div className="tl">
      <div className="tl-head">
        <span className="lbl">Events</span>
        <span className="tl-sub">Earthquake main shocks M{mlo.toFixed(1)}+, {y0}–{y1 - 1}: {ev.length} in total, <b>{nBoth}</b> reached both countries</span>
        <span className="tl-key"><i className="a" />one country<i className="b" />both</span>
      </div>
      <div className="tl-strip" onMouseLeave={() => onHover(null)}>
        {ticks.map((t) => <div key={t} className="tl-tick" style={{ left: x(t) }}><span>{t}</span></div>)}
        {ev.map((e) => (
          <button type="button" key={e.i} className={`ev${e.both ? ' both' : ''}${hover === e.i ? ' on' : ''}`} style={{ left: x(e.t), height: h(e.mag) }}
            aria-label={`Magnitude ${e.mag.toFixed(1)}, ${e.year}${e.both ? ', reached both countries' : ''}`}
            onMouseEnter={() => onHover(e.i)} onFocus={() => onHover(e.i)} onBlur={() => onHover(null)} />
        ))}
        {hv && (
          <div className={`tl-tip${hv.both ? ' both' : ''}`} style={{ left: x(hv.t), ['--ty' as string]: `${h(hv.mag)}px` }}>
            <b>M{hv.mag.toFixed(1)}</b> {hv.year}<small>{hv.both ? 'reached both countries' : 'one country'}</small>
          </div>
        )}
      </div>
    </div>
  )
}

