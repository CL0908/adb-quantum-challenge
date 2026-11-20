import { Fragment, useEffect, useState, type ReactNode } from 'react'
import { useJSON, useStory, useValues, V, type Story } from '../lib/data'
import { useInView } from './Reveal'

/* ---------- helpers ---------- */

function useNarrow(px = 700) {
  const [n, setN] = useState(false)
  useEffect(() => {
    const m = window.matchMedia(`(max-width: ${px}px)`)
    const f = () => setN(m.matches)
    f()
    m.addEventListener('change', f)
    return () => m.removeEventListener('change', f)
  }, [px])
  return n
}

const pctS = (x: number) => `${(100 * x).toFixed(100 * x < 10 ? 2 : 1)}%`
const sup = (e: number) => <sup>{String(e).replace('-', '−')}</sup>

function Card({ id, span, title, sub, children, className = '' }: { id?: string; span?: 1 | 2 | 3; title: string; sub: string; children: ReactNode; className?: string }) {
  const [ref, shown] = useInView<HTMLElement>()
  return (
    <article ref={ref} id={id} className={`bcard s${span ?? 1} ${shown ? 'in' : ''} ${className}`}>
      <header className="bhead">
        <h3>{title}</h3>
        <p>{sub}</p>
      </header>
      <div className="bvis">{children}</div>
    </article>
  )
}

/* ---------- one model, every source ---------- */

type Pill = { label: string; rate: number; tone: 'navy' | 'terra' | 'teal'; to: ('KG' | 'TJ')[]; soft?: boolean }

function Sources({ s }: { s: Story }) {
  const narrow = useNarrow(720)
  const d = s.decomp
  const pills: Pill[] = [
    { label: narrow ? 'Kyrgyz solo' : 'Kyrgyz solo quakes', rate: d.kg_solo, tone: 'navy', to: ['KG'] },
    { label: 'Shared quakes', rate: d.shared, tone: 'terra', to: ['KG', 'TJ'] },
    { label: narrow ? 'Tajik solo' : 'Tajik solo quakes', rate: d.tj_solo, tone: 'navy', to: ['TJ'] },
    { label: narrow ? 'Rain, one country' : 'Rain, one country', rate: d.rain_kg_or_tj_only, tone: 'teal', to: ['KG', 'TJ'], soft: true },
    { label: 'Rain, both', rate: d.rain_both, tone: 'teal', to: ['KG', 'TJ'] },
  ]
  const W = narrow ? 340 : 700, H = narrow ? 372 : 300
  const pw = narrow ? 200 : 214, ph = narrow ? 38 : 40, gap = narrow ? 48 : 58, y0 = narrow ? 2 : 8
  const nx = narrow ? 250 : 336, nw = narrow ? 90 : 132, nh = narrow ? 46 : 56
  const ny = { KG: narrow ? 52 : 82, TJ: narrow ? 158 : 192 }
  const jx = narrow ? 176 : 556, jw = narrow ? 164 : 144, jh = narrow ? 56 : 70, jy = narrow ? 300 : 115
  const py = (i: number) => y0 + i * gap + ph / 2
  const link = (x1: number, y1: number, x2: number, y2: number) => {
    const dx = (x2 - x1) * 0.5
    return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`
  }
  const lines: { d: string; tone: string; soft: boolean; key: string }[] = []
  pills.forEach((p, i) =>
    p.to.forEach((c, j) => {
      const off = p.to.length > 1 ? (j === 0 ? -11 : 11) : 0
      lines.push({ d: link(pw, py(i), nx, ny[c] + off), tone: p.tone, soft: !!p.soft, key: `${i}${c}` })
    }),
  )
  const toJoint = (['KG', 'TJ'] as const).map((c, j) => {
    if (narrow) return { d: `M${nx + nw / 2},${ny[c] + nh / 2} C${nx + nw / 2},${jy - 40} ${jx + jw * (0.3 + 0.4 * j)},${jy - 50} ${jx + jw * (0.3 + 0.4 * j)},${jy}`, key: c }
    return { d: link(nx + nw, ny[c], jx, jy + jh / 2 + (j ? 14 : -14)), key: c }
  })
  const colour = { navy: 'var(--navy)', terra: 'var(--terra)', teal: 'var(--teal)' }
  return (
    <svg className={`chain ${narrow ? "nw" : ""}`} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Source decomposition: three kinds of earthquake and two kinds of rain feed the two countries, which together decide the joint payout.">
      <g className="links">
        {lines.map((l, i) => (
          <Fragment key={l.key}>
            <path d={l.d} className="lk" />
            <path d={l.d} className={`lk flow ${l.soft ? 'soft' : ''}`} style={{ stroke: colour[l.tone as keyof typeof colour], animationDelay: `${i * 140}ms` }} />
          </Fragment>
        ))}
        {toJoint.map((l) => (
          <Fragment key={l.key}>
            <path d={l.d} className="lk" />
            <path d={l.d} className="lk flow" style={{ stroke: 'var(--navy)' }} />
          </Fragment>
        ))}
      </g>
      {pills.map((p, i) => (
        <g key={p.label} transform={`translate(0 ${y0 + i * gap})`}><g className="pillg" style={{ transitionDelay: `${i * 70}ms` }}>
          <title>{`${p.label}: ${pctS(p.rate)} chance in a year`}</title>
          <rect className="pill" width={pw} height={ph} rx={ph / 2} />
          <circle cx={19} cy={ph / 2} r={4.5} fill={colour[p.tone]} />
          <text x={32} y={ph / 2 + 4.5} className="plab">{p.label}</text>
          <text x={pw - 16} y={ph / 2 + 4.5} className="prate" textAnchor="end">{pctS(p.rate)}</text>
        </g></g>
      ))}
      {(['KG', 'TJ'] as const).map((c) => (
        <g key={c} transform={`translate(${nx} ${ny[c] - nh / 2})`}><g className="nodeg">
          <rect width={nw} height={nh} rx={14} className="cnode" />
          <text x={nw / 2} y={nh / 2 + 4.5} textAnchor="middle" className="ctext">{c === 'KG' ? (narrow ? 'Kyrgyzstan' : 'Kyrgyz Republic') : 'Tajikistan'}</text>
        </g></g>
      ))}
      <g transform={`translate(${jx} ${jy})`}><g className="nodeg last">
        <rect width={jw} height={jh} rx={16} className="jnode" />
        <text x={jw / 2} y={jh / 2 - 2} textAnchor="middle" className="jtext">Joint payout</text>
        <text x={jw / 2} y={jh / 2 + 15} textAnchor="middle" className="jsub">both bonds exhausted</text>
      </g></g>
    </svg>
  )
}

/* ---------- exact up to 28 countries ---------- */

function Ring({ n }: { n: number }) {
  const R = 56, c = 70
  const pts = Array.from({ length: n }, (_, i) => {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2
    return [c + R * Math.cos(a), c + R * Math.sin(a)] as const
  })
  return (
    <svg className="ring" viewBox="0 0 140 140" role="img" aria-label={`A ring of ${n} countries`}>
      <circle cx={c} cy={c} r={R} className="ringline" />
      {pts.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={i < 2 ? 5 : 3.6} className={`rn ${i < 2 ? 'hot' : ''}`} style={{ transitionDelay: `${i * 28}ms` }} />
      ))}
      <text x={c} y={c + 8} textAnchor="middle" className="ringn">{n}</text>
    </svg>
  )
}

function TimeLine({ data }: { data: [number, number][] }) {
  const W = 300, H = 104, L = 6, Rr = 14, T = 12, B = 22
  const xs = data.map((d) => d[0]), ys = data.map((d) => d[1])
  const x = (n: number) => L + ((n - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - L - Rr)
  const y = (t: number) => T + (1 - t / Math.max(...ys)) * (H - T - B)
  const line = data.map(([n, t], i) => `${i ? 'L' : 'M'}${x(n).toFixed(1)},${y(t).toFixed(1)}`).join('')
  const area = `${line}L${x(xs[xs.length - 1])},${H - B}L${x(xs[0])},${H - B}Z`
  const [hi, setHi] = useState(data.length - 1)
  const move = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const u = ((e.clientX - r.left) / r.width) * W
    let best = 0
    data.forEach(([n], i) => { if (Math.abs(x(n) - u) < Math.abs(x(data[best][0]) - u)) best = i })
    setHi(best)
  }
  const [hn, ht] = data[hi]
  return (
    <svg className="tline" viewBox={`0 0 ${W} ${H}`} onPointerMove={move} onPointerLeave={() => setHi(data.length - 1)} role="img"
      aria-label={`Exact solve time on one core against number of countries: from ${ys[0].toFixed(1)} s at ${xs[0]} to ${ys[ys.length - 1].toFixed(0)} s at ${xs[xs.length - 1]}.`}>
      <path d={area} className="tarea" />
      <path d={line} className="tpath" pathLength={1} />
      <line x1={x(hn)} x2={x(hn)} y1={y(ht)} y2={H - B} className="tguide" />
      <circle cx={x(hn)} cy={y(ht)} r={4.5} className="tdot" />
      <text x={Math.min(Math.max(x(hn), 40), W - 40)} y={y(ht) - 9} textAnchor="middle" className="tlab">{ht.toFixed(ht < 10 ? 1 : 0)} s · {hn}</text>
      <text x={L} y={H - 5} className="tax">2</text>
      <text x={W - Rr} y={H - 5} textAnchor="end" className="tax">countries</text>
    </svg>
  )
}

/* ---------- heatmap ---------- */

const SCEN: [string, string][] = [
  ['adb-base', 'Brief base'], ['adb-independent', 'Independent'], ['adb-common-shock', 'Common shock'], ['adb-rain-stress', 'Rain stress'], ['adb-rain-cover', 'Rain cover'],
]
const TGT: [string, string, string][] = [
  ['expected_loss', 'Loss', 'expected loss'], ['joint_attachment', 'Both pay', 'both bonds pay'], ['exceedance', 'Pool tail', 'pool above US$80m'],
  ['stop_loss', 'Stop-loss', 'stop-loss'], ['joint_exhaustion', 'Both out', 'both bonds exhausted'],
]
const fx = (f: number) => `${f >= 10 ? f.toFixed(0) : f.toFixed(1)}×`

function Heat({ s }: { s: Story }) {
  const [sel, setSel] = useState<[number, number] | null>(null)
  const P = s.heat.paths, Tm = s.heat.time
  const info = (() => {
    if (!sel) return null
    const [r, c] = sel
    const a = P[SCEN[r][0]][TGT[c][0]], b = Tm[SCEN[r][0]][TGT[c][0]]
    return { a, b, text: `${SCEN[r][1]}, ${TGT[c][2]}` }
  })()
  return (
    <div className="heat">
      <div className="hm-grid" role="grid" aria-label="Improvement factor over the best of four methods at equal paths">
        <span />
        {TGT.map(([k, short], c) => <span key={k} className={`hm-col ${c === 4 ? 'key' : ''}`}>{short}</span>)}
        {SCEN.map(([sk, sl], r) => (
          <Fragment key={sk}>
            <span className="hm-row">{sl}</span>
            {TGT.map(([tk], c) => {
              const f = P[sk][tk]
              const win = f > 1
              const t = win ? Math.min(1, 0.16 + Math.log10(f) / 2) : 0
              const key = c === 4
              return (
                <button key={tk} type="button" role="gridcell" className={`hm ${win ? '' : 'lose'} ${key ? 'key' : ''} ${sel && sel[0] === r && sel[1] === c ? 'on' : ''}`}
                  style={{ ['--t' as string]: t, transitionDelay: `${(r + c) * 40}ms` }}
                  onMouseEnter={() => setSel([r, c])} onFocus={() => setSel([r, c])} onMouseLeave={() => setSel(null)} onBlur={() => setSel(null)}
                  aria-label={`${sl}, ${TGT[c][2]}: ${fx(f)}`}>
                  {fx(f)}
                </button>
              )
            })}
          </Fragment>
        ))}
      </div>
      <p className="hm-note" aria-live="polite">
        {info ? <><b>{info.text}.</b> {fx(info.a)} on equal paths{`, ${fx(info.b)} on equal CPU time`}</>
          : <>RDPO beats the best of four methods in <b><V k="EQT_A_PATH_BRIEF" /></b> cells</>}
      </p>
    </div>
  )
}

/* ---------- honest about CPU time ---------- */

function parse(sv: string | undefined) {
  const n = (sv ?? '').match(/\d+/g)?.map(Number) ?? [0]
  return { lo: n[0], hi: n[n.length - 1], text: sv ?? '' }
}

function Checks() {
  const v = useValues()
  const rows = [
    { label: 'Benchmark', n: v.EQ0_N, p: v.EQ0_PATH, t: v.EQ0_TIME },
    { label: 'Data pools', n: v.EQ1_N, p: v.EQ1_PATH, t: v.EQ1_TIME },
    { label: 'Scaling', n: v.EQ2_N, p: v.EQ2_PATH, t: v.EQ2_TIME },
    { label: 'Dense graphs', n: v.EQ3_N, p: v.EQ3_PATH, t: v.EQ3_TIME },
  ]
  return (
    <div className="checks">
      {rows.map((r, i) => {
        const N = Number(r.n) || 1
        const p = parse(r.p), t = parse(r.t)
        return (
          <div className="crow" key={r.label} style={{ ['--d' as string]: `${i * 110}ms` }}>
            <div className="clab">{r.label}<span>{r.n} cells</span></div>
            <div className="cbars">
              <div className="cbar paths"><i style={{ ['--w' as string]: p.hi / N }} /><b>{p.text}</b></div>
              <div className="cbar cpu"><i className="range" style={{ ['--w' as string]: t.hi / N }} /><i style={{ ['--w' as string]: t.lo / N }} /><b>{t.text}</b></div>
            </div>
          </div>
        )
      })}
      <div className="clegend"><span><i className="d paths" />Same paths</span><span><i className="d cpu" />Same CPU time</span></div>
    </div>
  )
}

/* ---------- a bar for quantum ---------- */

const MAXLOG = 9
const at = (x: number) => `${(100 * Math.min(1, Math.max(0, Math.log10(x) / MAXLOG))).toFixed(2)}%`

function Bar({ s }: { s: Story }) {
  const o = useValues()
  const q = s.quantum
  const k = q.find((r) => r.n === 28) ?? q[q.length - 1]
  return (
    <div className="qbar">
      <div className="qscale">
        <div className="qline" />
        <div className="qgap" style={{ width: at(k.s_star) }} />
        {[0, 3, 6, 9].map((e) => (
          <div key={e} className="qtick" style={{ left: `${(100 * e) / MAXLOG}%` }}><i /><span>{e === 0 ? '1×' : <>10{sup(e)}×</>}</span></div>
        ))}
        {q.filter((r) => r !== k).map((r) => (
          <div key={r.n} className={`qdot other n${r.n}`} style={{ left: at(r.s_star) }} title={`${r.n} countries: ${r.s_star.toExponential(1)}×`}><span>{r.n}</span></div>
        ))}
        {k.ci && <div className="qci" style={{ left: at(k.ci[0]), width: `calc(${at(k.ci[1])} - ${at(k.ci[0])})` }} />}
        <div className="qdot key" style={{ left: at(k.s_star) }}>
          <div className="qlabel"><small>{k.n} countries</small><b><V k="Q_SSTAR_28" />×</b>{o.Q_SSTAR_28_CI ? <em>{o.Q_SSTAR_28_CI}</em> : null}</div>
        </div>
        <div className="qdot start" style={{ left: 0 }}><div className="qlabel left"><small>Projected hardware</small></div></div>
      </div>
      <dl className="qstats">
        <div><dt>Classical, measured</dt><dd><V k="Q_CLS_28" /></dd></div>
        <div><dt>Quantum, projected</dt><dd><V k="Q_TIME_28" /></dd></div>
        <p className="qnote">Dots: ring pools of {q.filter((r) => r !== k).map((r) => r.n).join(', ')} and {k.n} countries.</p>
      </dl>
    </div>
  )
}

/* ---------- regional pools ---------- */

interface CarecMap {
  bbox: number[]
  carec: { countries: Record<string, { name: string; rings: number[][][]; c: number[]; p25: number | null }>; edges: [string, string][] }
}

function PoolMap() {
  const map = useJSON<CarecMap>('map.json')
  const X0 = 40, X1 = 90, Y0 = 28, Y1 = 55
  const cos = Math.cos((41 * Math.PI) / 180), sx = 1000 / ((X1 - X0) * cos), h = (Y1 - Y0) * sx
  const px = (lo: number) => (lo - X0) * cos * sx
  const py = (la: number) => (Y1 - la) * sx
  const cs = map?.carec.countries
  return (
    <div className="pools">
      {cs ? (
        <svg viewBox={`0 0 1000 ${h.toFixed(0)}`} role="img" aria-label="Map of the ten CAREC countries, linked where earthquakes reached both">
          {Object.entries(cs).map(([iso, c]) => (
            <path key={iso} className={`pland ${iso === 'KG' || iso === 'TJ' ? 'hot' : ''}`} d={c.rings.map((r) => 'M' + r.map(([lo, la]) => `${px(lo).toFixed(1)},${py(la).toFixed(1)}`).join('L') + 'Z').join('')} />
          ))}
          {map!.carec.edges.map(([a, b], i) => cs[a] && cs[b] && (
            <line key={a + b} className="pedge" style={{ animationDelay: `${i * 90}ms` }} x1={px(cs[a].c[0])} y1={py(cs[a].c[1])} x2={px(cs[b].c[0])} y2={py(cs[b].c[1])} />
          ))}
          {Object.entries(cs).map(([iso, c]) => (
            <circle key={iso} className={`pnode ${iso === 'KG' || iso === 'TJ' ? 'hot' : ''}`} cx={px(c.c[0])} cy={py(c.c[1])} r={iso === 'KG' || iso === 'TJ' ? 15 : 11} />
          ))}
        </svg>
      ) : <div className="map-loading">Loading map</div>}
      <dl className="qstats compact">
        <div><dt>10 countries</dt><dd><V k="CAREC_MPS_S" /></dd></div>
        <div><dt><V k="AP_N" /> countries</dt><dd><V k="AP_MPS_SEC" /></dd></div>
      </dl>
    </div>
  )
}

/* ---------- the grid ---------- */

export default function Bento() {
  const s = useStory()
  return (
    <section className="sec" id="method">
      <div className="wrap wide">
        <h2 className="h2">Exact where possible, learned where not, quantum where it pays.</h2>
        <div className="panel">
          <div className="bento">
            <Card span={2} title="One model, every source" sub="Five causes of loss, per year, feeding the two-country bond.">
              {s ? <Sources s={s} /> : <div className="bskel" />}
            </Card>
            <Card id="exact" title="Exact to 28 countries" sub="Solve time on one CPU core.">
              {s ? (
                <>
                  <div className="exactrow">
                    <Ring n={s.scaling.ring[s.scaling.ring.length - 1][0]} />
                    <div className="bigstat"><b><V k="TN_RING28_S" /></b><span>bond dimension <V k="TN_RING28_BOND" /></span></div>
                  </div>
                  <TimeLine data={s.scaling.ring} />
                </>
              ) : <div className="bskel" />}
            </Card>
            <Card title="Learned sampler wins" sub="Accuracy gain at equal paths, by question.">
              {s ? <Heat s={s} /> : <div className="bskel" />}
            </Card>
            <Card id="accounting" title="Honest about CPU time" sub="Cells won, same paths against same CPU.">
              <Checks />
            </Card>
            <Card id="pools" title="Regional pools" sub="Ten CAREC countries, linked by shared quakes.">
              <PoolMap />
            </Card>
            <Card id="quantum" span={3} title="A bar for quantum" sub="Hardware speed-up needed over today’s projections to break even.">
              {s ? <Bar s={s} /> : <div className="bskel" />}
            </Card>
          </div>
        </div>
      </div>
    </section>
  )
}
