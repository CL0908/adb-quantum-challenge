import { Component, Suspense, lazy, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { answers, oneIn, pct, type Grid } from '../lib/model'
import type { MapData } from '../lib/map'
import { useData, type Solution, type Story } from './data'
import type { AnchorDef, Insets } from './Terrain'
import { ErrBars, Gauge, NodeRing, PoolBars, QChart, Spark, Timeline } from './Charts'
import Run from './Run'
import { prob, seconds, sci } from './format'
import { SiteNav } from '../shared/SiteNav'
import { Segmented, Slider } from '../shared/Controls'
import '../shared/theme.css'

const Terrain = lazy(() => import('./Terrain'))

class Boundary extends Component<{ fallback: ReactNode; children: ReactNode }, { bad: boolean }> {
  state = { bad: false }
  static getDerivedStateFromError() { return { bad: true } }
  render() { return this.state.bad ? this.props.fallback : this.props.children }
}

const ANCHORS: AnchorDef[] = [
  { id: 'kg', lon: 76.4, lat: 41.9 },
  { id: 'tj', lon: 71.7, lat: 38.4 },
  { id: 'both', lon: 72.9, lat: 39.85 },
]

const STAGES = [
  { n: 1, name: 'Source model', line: 'Independent Poisson sources' },
  { n: 2, name: 'Exact engine', line: 'Tensor network, exact answers' },
  { n: 3, name: 'Learned sampler', line: 'RDPO for the rarest events' },
  { n: 4, name: 'Quantum stage', line: 'Where a quantum computer could help' },
]

export default function App() {
  const map = useData<MapData>('map.json')
  const grid = useData<Grid>('grid.json')
  const sol = useData<Solution>('solution.json')
  const story = useData<Story>('story.json')
  const [eq, setEq] = useState(0.0745)
  const [rain, setRain] = useState(0.25)
  const [stage, setStage] = useState(2)
  const [poolId, setPoolId] = useState('pair')
  const [scenario, setScenario] = useState('adb-base')
  const [hover, setHover] = useState<number | null>(null)
  const [noGL, setNoGL] = useState(false)
  const [ready, setReady] = useState(false)
  // nothing data-driven shows until the data and the first placed frame are in, so the page never flashes a half-built layout
  const [late, setLate] = useState(false)
  useEffect(() => { const t = window.setTimeout(() => setLate(true), 4000); return () => window.clearTimeout(t) }, [])
  const anchorEls = useRef<Record<string, HTMLElement | null>>({})
  const root = useRef<HTMLDivElement>(null)

  const ans = useMemo(() => (grid ? answers(grid, eq, rain) : null), [grid, eq, rain])
  const getInsets = (): Insets => {
    const r = root.current
    const out: Insets = { l: 0, r: 0, t: 0, b: 0 }
    if (!r) return out
    const box = r.getBoundingClientRect()
    r.querySelectorAll<HTMLElement>('[data-inset]').forEach((n) => {
      const b = n.getBoundingClientRect()
      const k = n.dataset.inset
      if (k === 'l') out.l = Math.max(out.l, b.right - box.left)
      if (k === 'r') out.r = Math.max(out.r, box.right - b.left)
      if (k === 'b') out.b = Math.max(out.b, box.bottom - b.top)
      if (k === 't') out.t = Math.max(out.t, b.bottom - box.top)
    })
    return out
  }
  const pool = sol?.pools.find((p) => p.id === poolId)
  const preset = grid ? (Math.abs(eq - grid.brief.eq) < 0.0006 && Math.abs(rain - grid.brief.rain) < 0.0006 ? 'brief' : Math.abs(eq - grid.data.eq) < 0.0006 && Math.abs(rain - grid.data.rain) < 0.0006 ? 'data' : '') : ''
  // between grid nodes the numbers are interpolated from the exact grid, so only nodes and presets are labelled exact
  const onGrid = !!grid && ([grid.brief, grid.data].some((p) => Math.abs(eq - p.eq) < 1e-9 && Math.abs(rain - p.rain) < 1e-9) || (grid.eq_axis.some((x) => Math.abs(x - eq) < 1e-9) && grid.rain_axis.some((x) => Math.abs(x - rain) < 1e-9)))
  const e1 = sol?.e1[scenario]?.joint_exhaustion
  const qLast = story?.quantum[story.quantum.length - 1]
  const qSel = poolId === 'pair' ? 2 : poolId === 'ap' ? 28 : null

  const setAnchor = (id: string) => (el: HTMLElement | null) => { anchorEls.current[id] = el }
  const m = sol?.margins

  const mode = (() => {
    switch (stage) {
      case 1: return sol ? <>Fed by {sol.sources.length} Poisson sources</> : null
      case 2: { const pair = sol?.pools.find((p) => p.id === 'pair'); return pair ? (onGrid ? <>Exact, two countries, {seconds(pair.seconds)}</> : <>Interpolated between exact runs</>) : null }
      case 3: return e1 ? <>Run scenario, relative RMSE: RDPO {(e1.rdpo * 100).toFixed(2)}% vs Monte Carlo {(e1.mc * 100).toFixed(2)}%</> : null
      default: return qLast ? <>Quantum break-even at {qLast.n} countries {'×'}{sci(qLast.s_star)}</> : null
    }
  })()

  return (
    <div className={`sol st${stage}${((ready || noGL) && sol && grid) || late ? ' is-ready' : ''}`} ref={root}>
      <div className="map-host" aria-hidden={noGL}>
        {map && !noGL && (
          <Boundary fallback={null}>
            <Suspense fallback={null}>
              <Terrain map={map} eq={eq} stage={stage} anchors={ANCHORS} anchorEls={anchorEls} getInsets={getInsets} onFail={() => setNoGL(true)} onReady={() => requestAnimationFrame(() => requestAnimationFrame(() => setReady(true)))} />
            </Suspense>
          </Boundary>
        )}
        {noGL && <div className="nogl">3D view unavailable in this browser. All numbers below are unaffected.</div>}
        <div className={ready || noGL ? 'veil gone' : 'veil'}><span>Building the relief model</span></div>
        <div className="vignette" />
      </div>

      <SiteNav theme="dark" current="solution" />
      <header className="top" data-inset="t">
        <h1>Kyrgyz Republic <i>{'·'}</i> <em>Tajikistan</em></h1>
        <p>How the Disaster Relief Bonds are priced exactly, from sources to tail</p>
      </header>

      <aside className="jt-panel rail" data-inset="l" aria-label="Dependence controls">
        <div className="lbl">Dependence</div>
        <div style={{ margin: '12px 0 14px' }}>
          <Segmented block label="Presets" value={preset as 'brief' | 'data' | ''} disabled={!grid} onChange={(v) => { if (grid) { const g = v === 'brief' ? grid.brief : grid.data; setEq(g.eq); setRain(g.rain) } }}
            options={[{ id: 'brief', label: 'Brief' }, { id: 'data', label: 'Public records' }]} />
        </div>
        {grid && (
          <>
            <Slider id="eq" label="Earthquakes that hit both" min={0} max={0.6} value={eq} onChange={setEq} text={pct(eq)}
              marks={[{ at: grid.brief.eq, t: 'brief' }, { at: grid.data.eq, t: 'records' }]} />
            <Slider id="rain" label="Extreme rain that hits both" min={0} max={0.3} value={rain} onChange={setRain} text={pct(rain)}
              marks={[{ at: grid.data.rain, t: 'records' }, { at: grid.brief.rain, t: 'brief' }]} />
          </>
        )}
        <dl className="reads">
          <div><dt>Pool loss above US$80m</dt><dd>{ans ? prob(ans.exceed, 3) : '–'}<small>{ans ? oneIn(ans.exceed) : ''}</small></dd></div>
        </dl>
        <p className="note">{onGrid ? 'Exact' : 'Interpolated from the exact grid'}, three-year term.</p>
      </aside>

      <div className="anchor a-tl" ref={setAnchor('kg')}>
        <i className="dot" /><i className="lead" />
        <section className={`jt-panel card c-kg${stage === 1 ? ' hi' : ''}`} aria-label="Kyrgyz Republic">
          <div className="lbl teal">Kyrgyz Republic</div>
          <div className="pair"><div><span>Attachment</span><b>{m ? prob(m.KG.attach, 3) : '–'}</b></div><div><span>Exhaustion</span><b>{m ? prob(m.KG.exhaust, 3) : '–'}</b></div></div>
          {stage === 1 && sol && <div className="mini">{sol.sources.filter((s) => s.scope !== 'TJ').map((s) => <div key={s.id}><span>{s.kind === 'eq' ? 'Earthquake' : 'Rain'}{s.scope === 'both' ? ', shared' : ''}</span><b>{(s.rate * 100).toFixed(2)}% a year</b></div>)}</div>}
          <div className="cap">Within the 3-year term</div>
        </section>
      </div>
      <div className="anchor a-bl" ref={setAnchor('tj')}>
        <i className="dot" /><i className="lead" />
        <section className={`jt-panel card c-tj${stage === 1 ? ' hi' : ''}`} aria-label="Tajikistan">
          <div className="lbl green">Tajikistan</div>
          <div className="pair"><div><span>Attachment</span><b>{m ? prob(m.TJ.attach, 3) : '–'}</b></div><div><span>Exhaustion</span><b>{m ? prob(m.TJ.exhaust, 3) : '–'}</b></div></div>
          {stage === 1 && sol && <div className="mini">{sol.sources.filter((s) => s.scope !== 'KG').map((s) => <div key={s.id}><span>{s.kind === 'eq' ? 'Earthquake' : 'Rain'}{s.scope === 'both' ? ', shared' : ''}</span><b>{(s.rate * 100).toFixed(2)}% a year</b></div>)}</div>}
          <div className="cap">Within the 3-year term</div>
        </section>
      </div>
      <div className="anchor a-br" ref={setAnchor('both')}>
        <i className="dot hot" /><i className="lead" />
        <section className={`jt-panel card c-both${stage >= 2 ? ' hi' : ''}`} aria-label="Both bonds wiped out">
          <div className="lbl terra">Both bonds wiped out</div>
          <div className="big">{ans ? prob(ans.jointExhaustion, 3) : '–'}</div>
          <div className="one">{ans ? oneIn(ans.jointExhaustion) : ''} three-year terms</div>
          {grid && <Spark grid={grid} eq={eq} rain={rain} brief={grid.brief.eq} data={grid.data.eq} />}
          <div className="mode">{mode}</div>
        </section>
      </div>

      <aside className="jt-panel arch" data-inset="r" aria-label="Run estimate and architecture">
        {sol ? <Run sol={sol} scenario={scenario} setScenario={setScenario} /> : <div className="run" />}
        <div className="lbl arch-lbl">Architecture</div>
        <ol className="stages">
          {STAGES.map((s) => (
            <li key={s.n} className={stage === s.n ? 'on' : ''}>
              <button type="button" aria-expanded={stage === s.n} onClick={() => setStage(s.n)}>
                <span className="num">{s.n}</span><span className="tt"><b>{s.name}</b><small>{s.line}</small></span>
              </button>
              {stage === s.n && sol && (
                <div className="body">
                  {s.n === 1 && (
                    <>
                      <p>Losses come from {sol.sources.length} independent Poisson sources. A source that hits both countries is what makes the bonds fail together.</p>
                      <ul className="src-list">
                        {sol.sources.map((x) => (
                          <li key={x.id} className={x.scope === 'both' ? 'both' : ''}>
                            <span>{x.label}</span><b>{(x.rate * 100).toFixed(2)}%</b>
                            <i style={{ width: `${(100 * x.rate) / Math.max(...sol.sources.map((y) => y.rate))}%` }} />
                          </li>
                        ))}
                      </ul>
                      <small className="muted">Expected events per year, as a percentage.</small>
                    </>
                  )}
                  {s.n === 2 && pool && (
                    <>
                      <Segmented block label="Pool" value={poolId} onChange={setPoolId} options={sol.pools.map((p) => ({ id: p.id, label: p.label }))} />
                      <div className="pool-row">
                        <NodeRing pool={pool} sol={sol} />
                        <div className="facts">
                          <div><span>Exact solve time</span><b className="num-l">{seconds(pool.seconds)}</b></div>
                          <div><span>Tensor width</span><b>{pool.bond ?? 'not needed'}</b></div>
                          <div><span>Countries, links</span><b>{pool.n}, {pool.edges}</b></div>
                        </div>
                      </div>
                      <PoolBars pools={sol.pools} sel={poolId} />
                      <p className="muted">{pool.sub}. Method: {pool.method}, no sampling error.</p>
                    </>
                  )}
                  {s.n === 3 && (
                    <>
                      <p>A learned sampler concentrates paths on the rare joint events. Error per query, set against plain Monte Carlo with the same paths.</p>
                      <ErrBars sol={sol} scenario={scenario} />
                    </>
                  )}
                  {s.n === 4 && story && (
                    <>
                      <p>Hardware speed-up needed over the projected fault-tolerant quantum baseline to match the measured classical solver.</p>
                      {qLast && <div className="q-head"><b>{'×'}{sci(qLast.s_star)}</b><span>for {qLast.n} countries{qLast.ci ? <>, 95% interval {sci(qLast.ci[0])} to {sci(qLast.ci[1])}</> : null}</span></div>}
                      <QChart story={story} sel={qSel} />
                      <small className="muted">Required speed-up against the number of countries, log scale.</small>
                    </>
                  )}
                </div>
              )}
            </li>
          ))}
        </ol>
      </aside>

      <section className="jt-panel bar" data-inset="b" aria-label="Events and pooling">
        <div className="gauge-box">
          <Gauge value={ans ? ans.delivered : 0} />
          <div className="g-txt"><div className="lbl">Pooling</div><p>A shared US$80m pool pays <b>{ans ? (ans.delivered * 100).toFixed(1) : '–'}%</b> of expected claims</p></div>
        </div>
        {map && <Timeline map={map} hover={hover} onHover={setHover} />}
      </section>
    </div>
  )
}
