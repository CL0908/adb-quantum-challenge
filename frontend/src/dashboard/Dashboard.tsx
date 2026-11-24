import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { DivBars, Donut, LineChart, VBars, type Bar, type DBar, type Mark, type Series } from './charts'
import { delta, moduleCmd, pct, prob, probText, relErr, sci, sciSvg, TARGET_HINT, TARGET_LABEL, usd } from './fmt'
import { SiteNav } from '../shared/SiteNav'
import { Segmented } from '../shared/Controls'
import type { DashData, Grid, Pool, Scenario, TargetName } from './types'

type PoolId = 'kgtj' | 'carec' | 'ap'
type RainMode = 'both' | 'brief' | 'data'
type Show = 'joint' | 'exceed' | 'both'
type Metric = 'joint_exhaustion' | 'exceedance' | 'stop_loss'

const POOL_IDS: PoolId[] = ['kgtj', 'carec', 'ap']
const COL = { teal: 'var(--c1)', terra: 'var(--c2)', sand: 'var(--hi)', blue: 'var(--c3)', mute: 'var(--c-neutral)' }
const COVERAGE_OK = 0.95 // the range is a 95% interval, so 95% is the target it is judged against

interface Row {
  key: string
  target: TargetName
  label: string
  hint: string
  thr: number | null
  value: number
  money: boolean
  brief: number | null
  rel: number | null
  cov: number | null
  reps: number | null
  budget: number | null
}

const isMoney = (t: TargetName) => t === 'expected_loss' || t === 'stop_loss'
const fmtValue = (r: Row): ReactNode => (r.money ? usd(r.value) : prob(r.value))
const thrLabel = (t: number | null) => (t === null ? '' : ` US$${t}m`)

function useData(): { d: DashData | null; g: Grid | null; err: boolean } {
  const [d, setD] = useState<DashData | null>(null)
  const [g, setG] = useState<Grid | null>(null)
  const [err, setErr] = useState(false)
  useEffect(() => {
    let live = true
    const get = (f: string) => fetch(`/data/${f}`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(f))))
    Promise.all([get('dashboard.json'), get('grid.json')])
      .then(([a, b]) => {
        if (!live) return
        setD(a)
        setG(b)
      })
      .catch(() => live && setErr(true))
    return () => {
      live = false
    }
  }, [])
  return { d, g, err }
}

function rowsFor(pool: Pool, poolId: PoolId, scen: string): Row[] {
  if (poolId === 'kgtj' && pool.scenarios) {
    const sc = pool.scenarios[scen]
    return (Object.keys(sc.targets) as TargetName[]).map((t) => {
      const c = sc.targets[t]
      const thr = t === 'exceedance' || t === 'stop_loss' ? sc.threshold_musd : null
      return {
        key: t, target: t, label: TARGET_LABEL[t], hint: TARGET_HINT[t], thr, value: c.value, money: isMoney(t),
        brief: c.brief_rounded, rel: c.rdpo_rel_rmse, cov: c.rdpo_coverage, reps: c.reps, budget: c.budget,
      }
    })
  }
  return (pool.targets ?? []).map((c) => ({
    key: `${c.target}|${c.threshold_musd}`, target: c.target, label: TARGET_LABEL[c.target], hint: TARGET_HINT[c.target],
    thr: c.threshold_musd, value: c.value, money: isMoney(c.target), brief: null, rel: c.rdpo_rel_rmse, cov: c.rdpo_coverage,
    reps: c.reps, budget: c.budget,
  }))
}

function BigNumber({ text }: { text: string }) {
  const m = text.match(/^(.*?)(\.\d.*)$/)
  return (
    <span className="big">
      {m ? m[1] : text}
      {m && <span className="dim">{m[2]}</span>}
    </span>
  )
}
const Pill = ({ tone, children }: { tone: 'up' | 'down' | 'flat' | 'teal' | 'warn' | 'sand'; children: ReactNode }) => <span className={`pill ${tone}`}>{children}</span>

function Seg<T extends string>({ value, onChange, options, label }: { value: T; onChange: (v: T) => void; options: { v: T; label: string }[]; label: string }) {
  return <Segmented label={label} value={value} onChange={onChange} options={options.map((o) => ({ id: o.v, label: o.label }))} />
}

function Select({ id, label, value, onChange, children, disabled }: { id: string; label: string; value: string; onChange: (v: string) => void; children: ReactNode; disabled?: boolean }) {
  return (
    <label className={`sel${disabled ? ' off' : ''}`} htmlFor={id}>
      <span className="sel-l">{label}</span>
      <select className="sel-s" id={id} value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
        {children}
      </select>
    </label>
  )
}

/** A band is one question a banker asks, in reading order: the question, then the answer in one line. */
function BandHead({ n, tag, id, q, a }: { n: number; tag: string; id: string; q: string; a: ReactNode }) {
  return (
    <div className="band-h">
      <span className="band-n">{n} · {tag}</span>
      <h2 id={id} className="band-q">{q}</h2>
      <p className="band-a">{a}</p>
    </div>
  )
}

const fmtDate = (iso: string) => {
  const [y, m, d] = iso.split('-').map(Number)
  return `${d} ${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m - 1]} ${y}`
}

export default function Dashboard() {
  const { d, g, err } = useData()
  const [poolId, setPoolId] = useState<PoolId>('kgtj')
  const [scen, setScen] = useState('adb-base')
  const [rainMode, setRainMode] = useState<RainMode>('both')
  const [metric, setMetric] = useState<Metric>('joint_exhaustion')
  const [show, setShow] = useState<Show>('joint')
  const [picked, setPicked] = useState(false) // the hand on the scenario rows leaves once a scenario has been picked
  if (err) return <><SiteNav theme="light" current="dashboard" /><div className="shell state">The data files could not be loaded. Reload the page, or open the site home and come back.</div></>
  if (!d || !g) return <><SiteNav theme="light" current="dashboard" /><div className="shell state" aria-busy="true">Loading the pool data</div></>

  const pool = d.pools[poolId]
  const isKG = poolId === 'kgtj'
  const base = d.pools.kgtj.scenarios![d.scenario_order[0]]
  const sc = isKG ? pool.scenarios![scen] : null
  const pickScen = (s: string) => { setScen(s); setPicked(true) }

  // one-line answers under each band's question, computed from the same data the band shows
  const ss = d.pools.kgtj.scenarios!
  const top = d.scenario_order.slice(1).map((id) => ({ id, r: ss[id].targets.joint_exhaustion.value / base.targets.joint_exhaustion.value }))
    .sort((a, b) => Math.abs(Math.log(b.r)) - Math.abs(Math.log(a.r)))[0]
  const times = (r: number) => { const k = r >= 1 ? r : 1 / r; return `${k.toFixed(k < 10 ? 1 : 0)}× ${r >= 1 ? 'higher' : 'lower'}` }
  // the quake share alone: brief point against the public-records quake share, rain held at the brief's value
  // (the public-records point also lowers the shared rain, which would mix two effects into one number)
  const near = (axis: number[], v: number) => axis.reduce((b, x, i) => (Math.abs(x - v) < Math.abs(axis[b] - v) ? i : b), 0)
  const quakeOnly = g.values.joint_exhaustion[near(g.eq_axis, g.data.eq)][near(g.rain_axis, g.brief.rain)] / g.brief.joint_exhaustion // exact brief point over the nearest grid node to 48%
  const jx = sc?.targets.joint_exhaustion

  return (
    <>
      <SiteNav theme="light" current="dashboard" />
      <div className="page">
      <div className="shell">
        <div className="body">
          <header className="topbar" id="pools">
            <div className="tb-l">
              <h1>Banker <em>view</em></h1>
              <Select id="pool" label="Pool" value={poolId} onChange={(v) => setPoolId(v as PoolId)}>
                {POOL_IDS.map((id) => (
                  <option key={id} value={id}>
                    {d.pools[id].label}
                  </option>
                ))}
              </Select>
              {!isKG && <span className="note">The five brief scenarios are defined for the two-country bond.</span>}
            </div>
            <div className="tb-r">
              <span className="asof" title={`Latest committed result file used here: ${d.asof.file}`}>
                Data as of <b>{fmtDate(d.asof.date)}</b> · commit <code>{d.asof.commit}</code>
              </span>
              <span className="pillbtn">{pool.n_countries} countries</span>
            </div>
          </header>

          <main id="overview">
            {/* 1 · now: the size of the risk under the selected scenario */}
            <section className="band" aria-labelledby="b1-h">
              <BandHead n={1} tag="Now" id="b1-h" q="How big is the risk?"
                a={isKG && jx ? <>Under the <b className="t">{sc!.label.toLowerCase()}</b> scenario both bonds are wiped out together in <b>{oneIn(jx.value)}</b> three-year terms.</> : <>{pool.label}: the shared limit and the loss it leaves uncovered, at the US${pool.T_moderate}m threshold.</>} />
              {isKG && sc ? (
                <>
                  <div className="sc-tabs">
                    <Segmented label="Scenario" value={scen} onChange={pickScen} options={d.scenario_order.map((s) => ({ id: s, label: ss[s].label }))} />
                    <span className="sc-desc">{sc.desc}</span>
                  </div>
                  <KeyFigures sc={sc} />
                </>
              ) : (
                <div className="panel kpi"><Kpi d={d} g={g} pool={pool} poolId={poolId} scen={sc} /></div>
              )}
            </section>

            {/* 2 · stress: how the brief's scenarios move it */}
            <section className="band" id="scenarios" aria-labelledby="b2-h">
              <BandHead n={2} tag="Stress" id="b2-h" q={isKG ? 'What changes under the brief’s scenarios?' : 'How rare is each event?'}
                a={isKG ? <><b className="t">{ss[top.id].label}</b> moves the joint risk most: <b>{times(top.r)}</b> than the base case. Pick a scenario to update the whole page.</> : <>Exact probabilities on a log scale, for this pool.</>} />
              <div className={isKG ? 'duo' : ''}>
                <section className="panel bars" aria-label={isKG ? 'The brief’s five scenarios' : 'Reference probabilities'}>
                  <div className="ph">
                    <div>
                      <h2>{isKG ? 'Five scenarios against the base' : 'Reference probabilities'}</h2>
                      <p className="sub">{isKG ? 'Log scale, largest change first.' : 'Exact, log scale: how rare each event is for this pool.'}</p>
                    </div>
                    {isKG && (
                      <Seg<Metric> label="Metric" value={metric} onChange={setMetric} options={[
                        { v: 'joint_exhaustion', label: 'Both exhausted' },
                        { v: 'exceedance', label: 'Pool > limit' },
                        { v: 'stop_loss', label: 'Stop-loss' },
                      ]} />
                    )}
                  </div>
                  {isKG && !picked && <p className="th-note">Click a scenario row to switch the whole page to it</p>}
                  <ScenarioBars d={d} pool={pool} isKG={isKG} metric={metric} scen={scen} setScen={pickScen} hint={isKG && !picked ? top.id : undefined} />
                </section>
                {isKG && (
                  <section className="panel kpi" aria-labelledby="kpi-h">
                    <div className="ph">
                      <div>
                        <h2 id="kpi-h">Shared limit and loss</h2>
                        <p className="sub">One US${base.threshold_musd}m limit shared by both bonds, {sc!.label.toLowerCase()} scenario.</p>
                      </div>
                    </div>
                    <Kpi d={d} g={g} pool={pool} poolId={poolId} scen={sc} />
                  </section>
                )}
              </div>
            </section>

            {/* 3 · drivers: what pushes the tail up */}
            {poolId !== 'ap' && (
            <section className="band" aria-labelledby="b3-h">
              <BandHead n={3} tag="Drivers" id="b3-h" q={isKG ? 'What pushes the joint risk up?' : 'How fast does the tail thin out?'}
                a={isKG ? <>How often one earthquake hits both countries. Raising that share from the brief&rsquo;s {pct(g.brief.eq, 1)} to the {pct(g.data.eq, 0)} seen in public records, with shared rain held at {pct(g.brief.rain, 0)}, makes the joint risk <b>{times(quakeOnly)}</b>.</> : <>Exact loss exceedance down to 10⁻¹², with the two thresholds the register reads.</>} />
              <section className="panel trend" aria-labelledby="trend-h">
                <div className="ph">
                  <div>
                    <h2 id="trend-h">{isKG ? 'Tail risk vs dependence' : poolId === 'carec' ? 'Loss exceedance curve' : 'Estimator error by method'}</h2>
                    <p className="sub">
                      {isKG
                        ? `${show === 'exceed' ? 'Chance the pool pays more than US$80m' : show === 'joint' ? 'Chance both bonds are exhausted' : 'Chance the pool pays more than US$80m, and chance both bonds are exhausted,'} as the share of quakes that hit both countries rises.`
                        : poolId === 'carec'
                          ? `Exact probability that the pooled loss in one term exceeds each threshold (down to 10⁻¹²). Computed by ${pool.exact_label.replace('exact, ', '')}, not simulated.`
                          : `Relative error of each estimator at ${pool.targets![0].budget?.toLocaleString('en-US')} paths, ${pool.reps} replications. Exact references come from ${pool.exact_label}.`}
                    </p>
                    {isKG && <p className="scope">Base calibration across dependence: the scenario picked above does not change this chart.</p>}
                  </div>
                  {isKG && (
                    <div className="segs">
                    <Seg<Show> label="Probability shown" value={show} onChange={setShow} options={[
                      { v: 'joint', label: 'Both exhausted' },
                      { v: 'exceed', label: 'Pool > US$80m' },
                      { v: 'both', label: 'Both, two axes' },
                    ]} />
                    <Seg<RainMode> label="Rain shared share" value={rainMode} onChange={setRainMode} options={[
                      { v: 'both', label: 'Compare' },
                      { v: 'brief', label: `Brief ${pct(g.brief.rain, 0)}` },
                      { v: 'data', label: `Station data ${pct(0.02, 0)}` },
                    ]} />
                    </div>
                  )}
                </div>
                {isKG ? <TrendKG g={g} mode={rainMode} show={show} /> : <TrendCurve pool={pool} />}
              </section>
            </section>
            )}

            {/* 4 · trust: exact values, and how close the fast estimator gets to them */}
            <section className="band" aria-labelledby="b4-h">
              <BandHead n={poolId === 'ap' ? 3 : 4} tag="Trust" id="b4-h" q="Can these numbers be trusted?"
                a={isKG && jx ? <>Every figure above is an exact calculation. The fast estimator (RDPO) lands within <b className="t">{relErr(jx.rdpo_rel_rmse)}</b> of it on the joint risk, and its 95% range held the exact value in {Math.round(jx.rdpo_coverage * jx.reps)} of {jx.reps} runs.</> : <>Every figure is exact; the fast estimators are checked against it below.</>} />
              {poolId === 'ap' && (
                <section className="panel trend" aria-label="Estimator error by method">
                  <div className="ph"><div><h2>Estimator error by method</h2><p className="sub">{`Relative error of each estimator at ${pool.targets![0].budget?.toLocaleString('en-US')} paths, ${pool.reps} replications. Exact references come from ${pool.exact_label}.`}</p></div></div>
                  <MethodError pool={pool} />
                </section>
              )}
              <Register pool={pool} poolId={poolId} scen={scen} scLabel={sc?.label} scDesc={sc?.desc} />
              <div className="prov">
              <section className="panel runs" aria-labelledby="runs-h">
                <div className="ph">
                  <div>
                    <h2 id="runs-h">Model runs</h2>
                    <p className="sub">Every number here comes from a committed run that anyone can repeat.</p>
                  </div>
                </div>
                <details className="runs-d">
                <summary>For reviewers: files, commits and commands</summary>
                <ol className="timeline">
                  {pool.runs.map((r) => (
                    <li key={r.file}>
                      <b>{r.file.replace('results/', '')}</b>
                      <span className="use">{r.use}</span>
                      <span className="meta">
                        {r.script.replace('src/', '')} · commit <code>{r.commit}</code> · {fmtDate(r.date)}
                        {r.seed !== undefined && <> · seed {r.seed}</>}
                        {r.reps !== undefined && <> · {r.reps} reps</>}
                      </span>
                      <code className="cmd">{moduleCmd(r.script)}</code>
                    </li>
                  ))}
                </ol>
                </details>
              </section>

              <section className="panel methods" aria-labelledby="meth-h">
                <h2 id="meth-h">Methods</h2>
                <p>
                  Every probability marked exact comes from a full calculation, not a simulation. RDPO is the fast estimator whose error is tracked in the register.
                </p>
                <a className="jt-btn solid" href="/solution/">How the solution works</a>
                <a className="lnk" href="/#method">Method on the home page</a>
              </section>
              </div>
            </section>
          </main>
        </div>
      </div>
    </div>
    </>
  )
}

function TrendKG({ g, mode, show: which }: { g: Grid; mode: RainMode; show: Show }) {
  const ri = (r: number) => g.rain_axis.findIndex((x) => Math.abs(x - r) < 1e-9)
  const rBrief = ri(0.25)
  const rData = ri(0.02)
  const pick = (m: 'exceed' | 'joint_exhaustion', r: number) => g.eq_axis.map((e, i) => [e * 100, g.values[m][i][r]] as [number, number])
  const maxOf = (m: 'exceed' | 'joint_exhaustion') => Math.max(...[rBrief, rData].flatMap((r) => pick(m, r).map((p) => p[1])))
  // round tick steps (1, 2, 2.5, 5 x 10^k) on both axes, with the number of gridlines that fits the left axis tightest
  const step = (v: number, ks = [1, 2, 2.5, 5, 10]) => {
    const e = 10 ** Math.floor(Math.log10(v))
    return ks.map((k) => k * e).find((x) => x >= v * (1 - 1e-9))!
  }
  const lv = maxOf('exceed') * 1.04, rv = maxOf('joint_exhaustion') * 1.04
  // one probability on one axis by default; both together keep the two-axis view
  const fit = (v: number, ks?: number[]) => [4, 5, 6].reduce((b, k) => (step(v / k, ks) * k < step(v / b, ks) * b ? k : b), 4)
  const n = which === 'joint' ? fit(rv, [1, 2, 5, 10]) : fit(lv)
  const pctAxis = { max: step(lv / n) * n, fmt: (v: number) => `${+(v * 100).toPrecision(3)}%`, color: COL.teal }
  const sciAxis = { max: step(rv / n, [1, 2, 5, 10]) * n, fmt: (v: number) => sciSvg(v, 2), color: COL.terra }
  const dual = which === 'both' ? { n, l: pctAxis, r: sciAxis } : which === 'exceed' ? { n, l: pctAxis, r: { ...pctAxis, fmt: () => '' } } : { n, l: sciAxis, r: { ...sciAxis, fmt: () => '' } }
  const jAxis = which === 'both' ? ('r' as const) : undefined
  const wantE = which !== 'joint', wantJ = which !== 'exceed'
  const yDomain: [number, number] = [0, 1]
  const show = (which: 'brief' | 'data') => mode === 'both' || mode === which
  const series: Series[] = []
  const marks: Mark[] = []
  if (show('brief')) {
    if (wantE) { series.push({ id: 'eb', color: COL.teal, pts: pick('exceed', rBrief) }); marks.push({ x: g.brief.eq * 100, y: g.brief.exceed, color: COL.teal, label: 'brief-e' }) }
    if (wantJ) { series.push({ id: 'jb', color: COL.terra, axis: jAxis, pts: pick('joint_exhaustion', rBrief) }); marks.push({ x: g.brief.eq * 100, y: g.brief.joint_exhaustion, axis: jAxis, color: COL.terra, label: 'brief-j' }) }
  }
  if (show('data')) {
    const dash = mode === 'both' ? '5 4' : undefined
    if (wantE) { series.push({ id: 'ed', color: COL.teal, dash, pts: pick('exceed', rData) }); marks.push({ x: g.data.eq * 100, y: g.data.exceed, color: COL.sand, label: 'data-e' }) }
    if (wantJ) { series.push({ id: 'jd', color: COL.terra, dash, axis: jAxis, pts: pick('joint_exhaustion', rData) }); marks.push({ x: g.data.eq * 100, y: g.data.joint_exhaustion, axis: jAxis, color: COL.sand, label: 'data-j' }) }
  }
  const at = (x: number) => Math.min(g.eq_axis.length - 1, Math.max(0, Math.round(x)))
  const tipRows = (x: number) => {
    const i = at(x)
    const rows: { label: string; value: ReactNode; color: string }[] = []
    const add = (tag: string, r: number) => {
      if (wantE) rows.push({ label: `Pool > US$80m, ${tag}`, value: prob(g.values.exceed[i][r]), color: COL.teal })
      if (wantJ) rows.push({ label: `Both exhausted, ${tag}`, value: prob(g.values.joint_exhaustion[i][r]), color: COL.terra })
    }
    if (show('brief')) add(`rain ${pct(g.rain_axis[rBrief], 0)}`, rBrief)
    if (show('data')) add(`rain ${pct(g.rain_axis[rData], 0)}`, rData)
    return rows
  }
  return (
    <>
      <LineChart series={series} marks={marks} xDomain={[0, 60]} xTicks={[0, 10, 20, 30, 40, 50, 60]} xFmt={(x) => `${x}%`} yDomain={yDomain} dual={dual} height={290}
        tipX={(x) => `${at(x)}% of quakes hit both countries`} tipRows={tipRows} />
      <div className="jt-legend">
        {wantE && <span><i className="ln" style={{ background: COL.teal }} />Pr(pool pays more than US$80m){which === 'both' ? ', left axis' : ''}</span>}
        {wantJ && <span><i className="ln" style={{ background: COL.terra }} />Pr(both bonds exhausted){which === 'both' ? ', right axis' : ''}</span>}
        {mode === 'both' && <span><i className="ln dash" />dashed: rain share 2% (grid slice)</span>}
        {show('brief') && <span><i className="dot" style={{ borderColor: COL.teal }} />brief: {pct(g.brief.eq, 1)} quake share, {pct(g.brief.rain, 0)} rain</span>}
        {show('data') && <span><i className="dot" style={{ borderColor: COL.sand }} />public records, exact point: {pct(g.data.eq, 0)} quake, {pct(g.data.rain, 2)} rain</span>}
      </div>
      <p className="src">Exact: every point is a full calculation on the two-country grid, no sampling. Hover for values.</p>
    </>
  )
}

function TrendCurve({ pool }: { pool: Pool }) {
  const c = pool.curve!
  const FLOOR = 1e-12
  const pts = c.x.map((x, i) => [x, c.y[i]] as [number, number]).filter((p) => p[1] >= FLOOR)
  const yDomain: [number, number] = [-12, Math.ceil(Math.log10(Math.max(...c.y)))]
  const marks: Mark[] = [pool.T_moderate!, pool.T_rare!].map((t) => {
    const row = pool.targets!.find((r) => r.target === 'exceedance' && r.threshold_musd === t)!
    return { x: t, y: row.value, color: t === pool.T_moderate ? COL.teal : COL.sand, label: `T${t}` }
  })
  const nearest = (x: number) => c.x.reduce((b, v) => (Math.abs(v - x) < Math.abs(b - x) ? v : b), c.x[0])
  const xmin = Math.min(...c.x)
  const xmax = Math.max(...pts.map((p) => p[0]))
  const ticks = c.x.filter((x) => x % 100 === 0 && x <= xmax)
  return (
    <>
      <LineChart series={[{ id: 'c', color: COL.teal, pts }]} marks={marks} xDomain={[xmin, xmax]} xTicks={ticks} xFmt={(x) => `${x}`} yDomain={yDomain} height={270}
        tipX={(x) => `Loss above US$${nearest(x)}m`} tipRows={(x) => [{ label: 'Probability in one term', value: prob(c.y[c.x.indexOf(nearest(x))]), color: COL.teal }]} />
      <div className="jt-legend">
        <span><i className="ln" style={{ background: COL.teal }} />Pr(pooled loss &gt; threshold), threshold in US$m</span>
        {marks.map((m) => (
          <span key={m.label}><i className="dot" style={{ borderColor: m.color }} />US${m.x}m: {prob(m.y)}</span>
        ))}
      </div>
    </>
  )
}

function MethodError({ pool }: { pool: Pool }) {
  const rows = pool.targets!.filter((t) => t.rdpo_rel_rmse !== null && (t.target === 'exceedance' || t.target === 'stop_loss' || t.target === 'joint_exhaustion'))
  const lo = -3
  const hi = 0.15
  const w = (v: number) => `${Math.max(2, ((Math.log10(Math.max(v, 1e-3)) - lo) / (hi - lo)) * 100)}%`
  const meth: { k: 'mc_rel_rmse' | 'twist_rel_rmse' | 'rdpo_rel_rmse'; label: string; color: string }[] = [
    { k: 'mc_rel_rmse', label: 'Monte Carlo', color: COL.mute },
    { k: 'twist_rel_rmse', label: 'Exponential twisting', color: COL.blue },
    { k: 'rdpo_rel_rmse', label: 'RDPO', color: COL.teal },
  ]
  return (
    <>
      <div className="merr">
        {rows.map((r) => (
          <div className="merr-row" key={`${r.target}${r.threshold_musd}`}>
            <div className="merr-l">
              {TARGET_LABEL[r.target].replace('threshold', `US$${r.threshold_musd}m`)}
              <small>{prob(r.value) !== undefined && r.target !== 'stop_loss' ? <>exact {prob(r.value)}</> : <>exact {usd(r.value)}</>}</small>
            </div>
            <div className="merr-b">
              {meth.map((m) => {
                const v = r[m.k]
                return (
                  <div className="mb" key={m.k}>
                    <span className="mb-n">{m.label}</span>
                    <span className="mb-t">
                      {v === null ? null : v >= 0.999 ? <span className="mb-f hatch" style={{ width: '100%' }} /> : <span className="mb-f" style={{ width: w(v), background: m.color }} />}
                    </span>
                    <span className="mb-v">{v === null ? 'not run' : v >= 0.999 ? 'no hits' : relErr(v)}</span>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="jt-legend">
        <span>Bars on a log scale; shorter is better. &ldquo;No hits&rdquo; means plain Monte Carlo saw no loss above the threshold in any replication.</span>
      </div>
    </>
  )
}

function Kpi({ d, g, pool, poolId, scen }: { d: DashData; g: Grid; pool: Pool; poolId: PoolId; scen: Scenario | null }) {
  const base = d.pools.kgtj.scenarios![d.scenario_order[0]]
  let a: { label: string; text: string; pill: ReactNode }
  let b: { label: string; text: string; pill: ReactNode; extra?: ReactNode }
  let donut: ReactNode

  if (poolId === 'kgtj' && scen) {
    const dEL = delta(scen.targets.expected_loss.value, base.targets.expected_loss.value)
    const dSL = delta(scen.targets.stop_loss.value, base.targets.stop_loss.value)
    const pillOf = (x: ReturnType<typeof delta>) => (x ? <Pill tone={x.sign > 0 ? 'up' : x.sign < 0 ? 'down' : 'flat'}>{x.sign === 0 ? x.text : `${x.text} vs brief`}</Pill> : null)
    a = { label: 'Expected loss per term', text: usd(scen.targets.expected_loss.value), pill: pillOf(dEL) }
    const rec = delta(g.data.shortfall, base.targets.stop_loss.value)
    b = {
      label: `Stop-loss above US$${scen.threshold_musd}m`, text: usd(scen.targets.stop_loss.value), pill: pillOf(dSL),
      extra: rec && <Pill tone="sand">{`Public records: ${usd(g.data.shortfall)} (${rec.text})`}</Pill>,
    }
    donut = (
      <>
        <Donut caption="Share of expected claims a shared limit still pays" arcs={[{ frac: scen.delivered, color: COL.teal, r: 0 }, { frac: g.data.delivered, color: COL.sand, r: 15 }]}
          center={<><b>{pct(scen.delivered)}</b><small>paid</small></>} />
        <ul className="dl">
          <li><i style={{ background: COL.teal }} />{scen.label} scenario <b>{pct(scen.delivered)}</b></li>
          <li><i style={{ background: COL.sand }} />Public records, fixed calibration <b>{pct(g.data.delivered)}</b></li>
          <li className="dim">Brief base <b>{pct(base.delivered)}</b></li>
        </ul>
      </>
    )
  } else {
    const el = pool.targets!.find((t) => t.target === 'expected_loss')
    const ex = pool.targets!.find((t) => t.target === 'exceedance' && t.threshold_musd === pool.T_moderate)!
    const sl = pool.targets!.find((t) => t.target === 'stop_loss' && t.threshold_musd === pool.T_moderate)!
    const pill = (t: { rdpo_rel_rmse: number | null; budget: number | null }) => (
      <Pill tone="teal">{t.rdpo_rel_rmse === null ? 'exact reference' : `exact, RDPO ${relErr(t.rdpo_rel_rmse)}`}</Pill>
    )
    a = el
      ? { label: 'Expected loss per term', text: usd(el.value), pill: pill(el) }
      : { label: `Pr(pooled loss > US$${pool.T_moderate}m)`, text: probText(ex.value), pill: pill(ex) }
    b = { label: `Stop-loss above US$${pool.T_moderate}m`, text: usd(sl.value), pill: pill(sl) }
    donut =
      pool.delivered !== null && pool.delivered !== undefined ? (
        <>
          <Donut caption="Share of expected claims a shared limit still pays" arcs={[{ frac: pool.delivered, color: COL.teal, r: 0 }]} center={<><b>{pct(pool.delivered, 2)}</b><small>paid</small></>} />
          <ul className="dl">
            <li><i style={{ background: COL.teal }} />Pooled limit US${pool.T_moderate}m <b>{pct(pool.delivered, 2)}</b></li>
          </ul>
        </>
      ) : (
        <p className="empty">The exported 28-country run holds no expected-loss reference, so the paid share is not shown for this pool.</p>
      )
  }
  return (
    <div className="kpi-body">
      <div className="kpi-d">{donut}</div>
      <div className="kpi-n">
        <div className="num">
          <span className="num-l">{a.label}</span>
          <BigNumber text={a.text} />
          <span className="pills">{a.pill}</span>
        </div>
        <div className="num">
          <span className="num-l">{b.label}</span>
          <BigNumber text={b.text} />
          <span className="pills">{b.pill}{b.extra}</span>
        </div>
      </div>
    </div>
  )
}

function oneIn(p: number) {
  const n = 1 / p, m = n >= 1000 ? 100 : 1
  return `1 in ${(Math.round(n / m) * m).toLocaleString('en-US')}`
}

function KeyFigures({ sc }: { sc: Scenario }) {
  const j = sc.targets.joint_exhaustion, e = sc.targets.exceedance
  return (
    <section className="keyfig" aria-label={`Key figures, ${sc.label} scenario`}>
      <div><span>Both bonds exhausted</span><b>{prob(j.value)}</b><small>{oneIn(j.value)} three-year terms, exact</small></div>
      <div><span>Pool pays over US${sc.threshold_musd}m</span><b>{prob(e.value)}</b><small>{oneIn(e.value)} terms, exact</small></div>
      <div><span>Claims one shared limit pays</span><b>{pct(sc.delivered)}</b><small>of expected claims, US${sc.threshold_musd}m limit</small></div>
    </section>
  )
}

function ScenarioBars({ d, pool, isKG, metric, scen, setScen, hint }: { d: DashData; pool: Pool; isKG: boolean; metric: Metric; scen: string; setScen: (s: string) => void; hint?: string }) {
  if (isKG) {
    const ss = pool.scenarios!
    const ref = d.scenario_order[0]
    const money = metric === 'stop_loss'
    const baseV = ss[ref].targets[metric].value
    const rows: DBar[] = d.scenario_order.map((id) => {
      const c = ss[id].targets[metric]
      return {
        id, label: ss[id].label, ratio: c.value / baseV, active: id === scen, ref: id === ref, hint: id === hint,
        display: money ? usd(c.value) : prob(c.value),
        sub: <>brief {money ? usd(c.brief_rounded) : sci(c.brief_rounded, 2)}</>,
      }
    }).sort((a, b) => (a.ref ? -1 : b.ref ? 1 : Math.abs(Math.log(b.ratio)) - Math.abs(Math.log(a.ratio))))
    return <><DivBars rows={rows} onPick={setScen} /><p className="src">Exact values for a three-year term; the brief&rsquo;s Table 5 figure under each name.</p></>
  }
  const bars: Bar[] = pool
    .targets!.filter((t) => t.target === 'exceedance' || t.target === 'joint_exhaustion')
    .map((t) => ({
      id: `${t.target}${t.threshold_musd}`, label: t.target === 'joint_exhaustion' ? 'Both exhausted' : `Loss > US$${t.threshold_musd}m`,
      value: t.value, display: prob(t.value), sub: sci(t.value),
    }))
  return <VBars bars={bars} log floor={1e-12} height={210} />
}

function Register({ pool, poolId, scen, scLabel, scDesc }: { pool: Pool; poolId: PoolId; scen: string; scLabel?: string; scDesc?: string }) {
  const rows = useMemo(() => rowsFor(pool, poolId, scen), [pool, poolId, scen])
  const kg = poolId === 'kgtj'
  const budget = rows.find((r) => r.budget)?.budget ?? null
  return (
    <section className="panel reg" aria-labelledby="reg-h">
      <div className="ph">
        <div>
          <h2 id="reg-h">Risk register</h2>
          <p className="sub">
            {kg ? `${scLabel} scenario: ${scDesc}.` : `${pool.label}, calibrated to public records.`} The fast estimate is checked against the exact value: its error, and how often its 95% range held the exact value over repeated runs.
          </p>
        </div>
        <a className="lnk reg-run" href="/solution/">Run an estimate yourself on the Solution page</a>
      </div>
      <div className="tbl" role="table" aria-label="Risk register">
        <div className="tr th" role="row">
          <span role="columnheader">Target</span>
          <span role="columnheader">Exact value</span>
          <span role="columnheader">Method</span>
          <span role="columnheader">Fast estimate check</span>
          <span role="columnheader">Status</span>
        </div>
        {rows.map((r) => {
          const ok = r.cov !== null && r.cov >= COVERAGE_OK
          return (
            <div className="tr" role="row" key={r.key}>
              <span role="cell" className="c-t" data-l="Target">
                <b>{r.label}{r.thr !== null && r.target !== 'expected_loss' ? <span className="thr">{thrLabel(r.thr)}</span> : null}</b>
                <small>{r.hint}</small>
              </span>
              <span role="cell" className="c-v" data-l="Exact value">
                <b>{fmtValue(r)}</b>
                {r.brief !== null && <small>brief {r.money ? usd(r.brief) : r.brief >= 0.001 ? `${+(r.brief * 100).toPrecision(2)}%` : sci(r.brief, 2)}</small>}
              </span>
              <span role="cell" data-l="Method">
                <span>{`exact, full calculation (${pool.exact_label.replace('exact, ', '').replace(/ \(.*\)/, '')})`}</span>
              </span>
              <span role="cell" data-l="Fast estimate check">
                {r.rel === null ? <small>no run</small> : (
                  <>
                    <b>{relErr(r.rel)}</b>
                    {r.cov !== null && <span className="covbar" aria-hidden="true"><i style={{ width: `${Math.min(100, r.cov * 100)}%` }} className={ok ? '' : 'low'} /><u /></span>}
                    <small style={r.cov !== null && !ok ? { color: 'var(--hi)' } : undefined}>{r.cov === null ? 'range check n/a' : `range held the exact value in ${Math.round(r.cov * (r.reps ?? 0))} of`} {r.reps} runs (target 95%){budget === null ? '' : `, ${budget.toLocaleString('en-US')} paths`}</small>
                  </>
                )}
              </span>
              <span role="cell" data-l="Status">
                <Pill tone="teal">exact</Pill>
              </span>
            </div>
          )
        })}
      </div>
    </section>
  )
}
