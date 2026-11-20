import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { Donut, LineChart, VBars, type Bar, type Mark, type Series } from './charts'
import { delta, moduleCmd, pct, prob, probText, relErr, sci, TARGET_HINT, TARGET_LABEL, usd } from './fmt'
import type { DashData, Grid, Pool, Scenario, TargetName } from './types'

type PoolId = 'kgtj' | 'carec' | 'ap'
type RainMode = 'both' | 'brief' | 'data'
type Metric = 'joint_exhaustion' | 'exceedance' | 'stop_loss'

const POOL_IDS: PoolId[] = ['kgtj', 'carec', 'ap']
const COL = { teal: '#34D1BC', terra: '#F0846A', lime: '#D4F26B', mute: '#8B9A98' }
const COVERAGE_OK = 0.9

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

const Icon = ({ d }: { d: ReactNode }) => (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {d}
  </svg>
)
const NAV = [
  { id: 'overview', label: 'Overview', href: '#overview', d: <path d="M4 11.5 12 5l8 6.5V19a1 1 0 0 1-1 1h-4v-5H9v5H5a1 1 0 0 1-1-1z" /> },
  { id: 'scenarios', label: 'Scenarios', href: '#scenarios', d: <><path d="M5 20V10M12 20V4M19 20v-7" /></> },
  { id: 'pools', label: 'Pools', href: '#pools', d: <><circle cx="8" cy="9" r="3.2" /><circle cx="16.5" cy="10" r="2.6" /><path d="M2.8 19c.4-3 2.5-4.6 5.2-4.6s4.8 1.6 5.2 4.6M14 15.2c2.8-.5 5.6.6 6.4 3.8" /></> },
  { id: 'methods', label: 'Methods', href: '/solution/', d: <><path d="M9 3h6M10 3v6L5 18.5A1.5 1.5 0 0 0 6.3 21h11.4a1.5 1.5 0 0 0 1.3-2.5L14 9V3" /><path d="M7.5 15h9" /></> },
  { id: 'documents', label: 'Documents', href: '/#documents', d: <><path d="M7 3h7l4 4v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z" /><path d="M14 3v4h4M9 12h6M9 16h6" /></> },
]

function BigNumber({ text }: { text: string }) {
  const m = text.match(/^(.*?)(\.\d.*)$/)
  return (
    <span className="big">
      {m ? m[1] : text}
      {m && <span className="dim">{m[2]}</span>}
    </span>
  )
}
const Pill = ({ tone, children }: { tone: 'up' | 'down' | 'flat' | 'teal' | 'warn' | 'lime'; children: ReactNode }) => <span className={`pill ${tone}`}>{children}</span>

function Seg<T extends string>({ value, onChange, options, label }: { value: T; onChange: (v: T) => void; options: { v: T; label: string }[]; label: string }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.v} type="button" className={o.v === value ? 'on' : ''} aria-pressed={o.v === value} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

function Select({ id, label, value, onChange, children, disabled }: { id: string; label: string; value: string; onChange: (v: string) => void; children: ReactNode; disabled?: boolean }) {
  return (
    <label className={`sel${disabled ? ' off' : ''}`} htmlFor={id}>
      <span className="sel-l">{label}</span>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
        {children}
      </select>
    </label>
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
  const [active, setActive] = useState('overview')

  useEffect(() => {
    const onHash = () => setActive(location.hash.slice(1) || 'overview')
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  if (err) return <div className="shell state">The data files could not be loaded. Reload the page, or open the site home and come back.</div>
  if (!d || !g) return <div className="shell state" aria-busy="true">Loading the pool data</div>

  const pool = d.pools[poolId]
  const isKG = poolId === 'kgtj'
  const base = d.pools.kgtj.scenarios![d.scenario_order[0]]
  const sc = isKG ? pool.scenarios![scen] : null

  return (
    <div className="page">
      <div className="shell">
        <nav className="rail" aria-label="Sections">
          <a className="mark" href="/" aria-label="Back to the Joint-Tail site" title="Joint-Tail home">
            <svg viewBox="0 0 32 32" width="30" height="30" aria-hidden="true"><circle cx="16" cy="16" r="14" fill="#1b2a2b" stroke="#34D1BC" strokeWidth="1.5" /><path d="M7 21 13 11l4 6 3-4 5 8z" fill="#D4F26B" /></svg>
          </a>
          <div className="rail-nav">
            {NAV.map((n) => (
              <a key={n.id} href={n.href} className={active === n.id && n.href.startsWith('#') ? 'on' : ''} aria-label={n.label} title={n.label} onClick={() => n.href.startsWith('#') && setActive(n.id)}>
                <Icon d={n.d} />
                <span className="rail-t">{n.label}</span>
              </a>
            ))}
          </div>
        </nav>

        <div className="body">
          <header className="topbar" id="pools">
            <div className="tb-l">
              <h1>Banker view</h1>
              <Select id="pool" label="Pool" value={poolId} onChange={(v) => setPoolId(v as PoolId)}>
                {POOL_IDS.map((id) => (
                  <option key={id} value={id}>
                    {d.pools[id].label}
                  </option>
                ))}
              </Select>
              <Select id="scen" label="Scenario" value={isKG ? scen : d.scenario_order[0]} onChange={setScen} disabled={!isKG}>
                {d.scenario_order.map((s) => (
                  <option key={s} value={s}>
                    {d.pools.kgtj.scenarios![s].label}
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

          <div className="cols">
            <main className="main" id="overview">
              <section className="panel trend" aria-labelledby="trend-h">
                <div className="ph">
                  <div>
                    <h2 id="trend-h">{isKG ? 'Tail risk vs dependence' : poolId === 'carec' ? 'Loss exceedance curve' : 'Estimator error by method'}</h2>
                    <p className="sub">
                      {isKG
                        ? 'Chance the pool pays more than US$80m, and chance both bonds are exhausted, as the share of quakes that hit both countries rises.'
                        : poolId === 'carec'
                          ? `Exact probability that the pooled loss in one term exceeds each threshold (down to 10⁻¹²). Computed by ${pool.exact_label.replace('exact, ', '')}, not simulated.`
                          : `Relative error of each estimator at ${pool.targets![0].budget?.toLocaleString('en-US')} paths, ${pool.reps} replications. Exact references come from ${pool.exact_label}.`}
                    </p>
                  </div>
                  {isKG && (
                    <Seg<RainMode> label="Rain shared share" value={rainMode} onChange={setRainMode} options={[
                      { v: 'both', label: 'Compare' },
                      { v: 'brief', label: `Brief ${pct(g.brief.rain, 0)}` },
                      { v: 'data', label: `Station data ${pct(0.02, 0)}` },
                    ]} />
                  )}
                </div>
                {isKG ? <TrendKG g={g} mode={rainMode} /> : poolId === 'carec' ? <TrendCurve pool={pool} /> : <MethodError pool={pool} />}
              </section>

              <div className="duo">
                <section className="panel kpi" aria-labelledby="kpi-h">
                  <div className="ph">
                    <div>
                      <h2 id="kpi-h">Shared limit and loss</h2>
                      <p className="sub">{isKG ? `One US$${base.threshold_musd}m limit shared by both bonds.` : `Pooled cover read at the US$${pool.T_moderate}m threshold.`}</p>
                    </div>
                  </div>
                  <Kpi d={d} g={g} pool={pool} poolId={poolId} scen={sc} />
                </section>

                <section className="panel bars" id="scenarios" aria-labelledby="bars-h">
                  <div className="ph">
                    <div>
                      <h2 id="bars-h">{isKG ? 'The brief’s five scenarios' : 'Reference probabilities'}</h2>
                      <p className="sub">{isKG ? 'Exact values; the brief’s Table 5 figure under each bar. Click a bar to select the scenario.' : 'Exact, log scale: how rare each event is for this pool.'}</p>
                    </div>
                    {isKG && (
                      <Seg<Metric> label="Metric" value={metric} onChange={setMetric} options={[
                        { v: 'joint_exhaustion', label: 'Both exhausted' },
                        { v: 'exceedance', label: 'Pool > limit' },
                        { v: 'stop_loss', label: 'Stop-loss' },
                      ]} />
                    )}
                  </div>
                  <ScenarioBars d={d} pool={pool} isKG={isKG} metric={metric} scen={scen} setScen={setScen} />
                </section>
              </div>

              <Register pool={pool} poolId={poolId} scen={scen} scLabel={sc?.label} scDesc={sc?.desc} />
            </main>

            <aside className="side" aria-label="Provenance">
              <section className="panel runs" aria-labelledby="runs-h">
                <div className="ph">
                  <div>
                    <h2 id="runs-h">Model runs</h2>
                    <p className="sub">The committed runs behind these numbers. Rerun any of them to reproduce.</p>
                  </div>
                </div>
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
              </section>

              <section className="panel methods" aria-labelledby="meth-h">
                <h2 id="meth-h">Methods</h2>
                <p>
                  Every probability marked exact comes from a full calculation, not a simulation. RDPO is the fast estimator whose error is tracked in the register.
                </p>
                <a className="btn" href="/solution/">How the solution works</a>
                <a className="lnk" href="/#method">Method on the home page</a>
              </section>
            </aside>
          </div>
        </div>
      </div>
    </div>
  )
}

function TrendKG({ g, mode }: { g: Grid; mode: RainMode }) {
  const ri = (r: number) => g.rain_axis.findIndex((x) => Math.abs(x - r) < 1e-9)
  const rBrief = ri(0.25)
  const rData = ri(0.02)
  const pick = (m: 'exceed' | 'joint_exhaustion', r: number) => g.eq_axis.map((e, i) => [e * 100, g.values[m][i][r]] as [number, number])
  const maxOf = (m: 'exceed' | 'joint_exhaustion') => Math.max(...[rBrief, rData].flatMap((r) => pick(m, r).map((p) => p[1])))
  const nice = (v: number) => {
    const e = 10 ** Math.floor(Math.log10(v))
    return [1, 2, 4, 5, 8, 10].map((k) => k * e).find((x) => x >= v * 1.04)!
  }
  const dual = {
    l: { max: nice(maxOf('exceed')), fmt: (v: number) => `${+(v * 100).toPrecision(2)}%`, color: COL.teal },
    r: { max: nice(maxOf('joint_exhaustion')), fmt: (v: number) => (v === 0 ? '0' : sci(v, 1)), color: COL.terra },
  }
  const yDomain: [number, number] = [0, 1]
  const show = (which: 'brief' | 'data') => mode === 'both' || mode === which
  const series: Series[] = []
  const marks: Mark[] = []
  if (show('brief')) {
    series.push({ id: 'eb', color: COL.teal, pts: pick('exceed', rBrief) }, { id: 'jb', color: COL.terra, axis: 'r', pts: pick('joint_exhaustion', rBrief) })
    marks.push({ x: g.brief.eq * 100, y: g.brief.exceed, color: COL.teal, label: 'brief-e' }, { x: g.brief.eq * 100, y: g.brief.joint_exhaustion, axis: 'r', color: COL.terra, label: 'brief-j' })
  }
  if (show('data')) {
    const dash = mode === 'both' ? '5 4' : undefined
    series.push({ id: 'ed', color: COL.teal, dash, pts: pick('exceed', rData) }, { id: 'jd', color: COL.terra, dash, axis: 'r', pts: pick('joint_exhaustion', rData) })
    marks.push({ x: g.data.eq * 100, y: g.data.exceed, color: COL.lime, label: 'data-e' }, { x: g.data.eq * 100, y: g.data.joint_exhaustion, axis: 'r', color: COL.lime, label: 'data-j' })
  }
  const at = (x: number) => Math.min(g.eq_axis.length - 1, Math.max(0, Math.round(x)))
  const tipRows = (x: number) => {
    const i = at(x)
    const rows: { label: string; value: ReactNode; color: string }[] = []
    const add = (tag: string, r: number) => {
      rows.push({ label: `Pool > US$80m, ${tag}`, value: prob(g.values.exceed[i][r]), color: COL.teal })
      rows.push({ label: `Both exhausted, ${tag}`, value: prob(g.values.joint_exhaustion[i][r]), color: COL.terra })
    }
    if (show('brief')) add(`rain ${pct(g.rain_axis[rBrief], 0)}`, rBrief)
    if (show('data')) add(`rain ${pct(g.rain_axis[rData], 0)}`, rData)
    return rows
  }
  return (
    <>
      <LineChart series={series} marks={marks} xDomain={[0, 60]} xTicks={[0, 10, 20, 30, 40, 50, 60]} xFmt={(x) => `${x}%`} yDomain={yDomain} dual={dual} height={290}
        tipX={(x) => `${at(x)}% of quakes hit both countries`} tipRows={tipRows} />
      <div className="legend">
        <span><i className="ln" style={{ background: COL.teal }} />Pr(pool pays more than US$80m), left axis</span>
        <span><i className="ln" style={{ background: COL.terra }} />Pr(both bonds exhausted), right axis</span>
        {mode === 'both' && <span><i className="ln dash" />dashed: station-data rain share</span>}
        {show('brief') && <span><i className="dot" style={{ borderColor: COL.teal }} />brief: {pct(g.brief.eq, 1)} quake share, {pct(g.brief.rain, 0)} rain</span>}
        {show('data') && <span><i className="dot" style={{ borderColor: COL.lime }} />public records: {pct(g.data.eq, 0)} quake, {pct(g.data.rain, 1)} rain</span>}
      </div>
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
    return { x: t, y: row.value, color: t === pool.T_moderate ? COL.teal : COL.lime, label: `T${t}` }
  })
  const nearest = (x: number) => c.x.reduce((b, v) => (Math.abs(v - x) < Math.abs(b - x) ? v : b), c.x[0])
  const xmin = Math.min(...c.x)
  const xmax = Math.max(...pts.map((p) => p[0]))
  const ticks = c.x.filter((x) => x % 100 === 0 && x <= xmax)
  return (
    <>
      <LineChart series={[{ id: 'c', color: COL.teal, pts }]} marks={marks} xDomain={[xmin, xmax]} xTicks={ticks} xFmt={(x) => `${x}`} yDomain={yDomain} height={270}
        tipX={(x) => `Loss above US$${nearest(x)}m`} tipRows={(x) => [{ label: 'Probability in one term', value: prob(c.y[c.x.indexOf(nearest(x))]), color: COL.teal }]} />
      <div className="legend">
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
    { k: 'twist_rel_rmse', label: 'Exponential twisting', color: COL.terra },
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
      <div className="legend">
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
      extra: rec && <Pill tone="lime">{`Public records: ${usd(g.data.shortfall)} (${rec.text})`}</Pill>,
    }
    donut = (
      <>
        <Donut caption="Share of expected claims a shared limit still pays" arcs={[{ frac: scen.delivered, color: COL.teal, r: 0 }, { frac: g.data.delivered, color: COL.lime, r: 15 }]}
          center={<><b>{pct(scen.delivered)}</b><small>paid</small></>} />
        <ul className="dl">
          <li><i style={{ background: COL.teal }} />{scen.label} scenario <b>{pct(scen.delivered)}</b></li>
          <li><i style={{ background: COL.lime }} />Public records <b>{pct(g.data.delivered)}</b></li>
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

function ScenarioBars({ d, pool, isKG, metric, scen, setScen }: { d: DashData; pool: Pool; isKG: boolean; metric: Metric; scen: string; setScen: (s: string) => void }) {
  if (isKG) {
    const ss = pool.scenarios!
    const bars: Bar[] = d.scenario_order.map((id) => {
      const c = ss[id].targets[metric]
      const money = metric === 'stop_loss'
      return {
        id, label: ss[id].label, value: c.value, active: id === scen,
        display: money ? usd(c.value) : prob(c.value),
        sub: <>brief {money ? usd(c.brief_rounded) : sci(c.brief_rounded, 2)}</>,
      }
    })
    return <VBars bars={bars} log={metric !== 'stop_loss'} onPick={setScen} height={210} />
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
  return (
    <section className="panel reg" aria-labelledby="reg-h">
      <div className="ph">
        <div>
          <h2 id="reg-h">Risk register</h2>
          <p className="sub">
            {kg ? `${scLabel} scenario: ${scDesc}.` : `${pool.label}, calibrated to public records.`} Coverage is the share of replications whose 95% interval holds the exact value.
          </p>
        </div>
      </div>
      <div className="tbl" role="table" aria-label="Risk register">
        <div className="tr th" role="row">
          <span role="columnheader">Target</span>
          <span role="columnheader">Exact value</span>
          <span role="columnheader">Method</span>
          <span role="columnheader">Error of RDPO</span>
          <span role="columnheader">Coverage</span>
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
                {r.brief !== null && <small>brief {r.money ? usd(r.brief) : sci(r.brief, 2)}</small>}
              </span>
              <span role="cell" data-l="Method">
                <span className="chip">exact</span>
                <small>{pool.exact_label.replace('exact, ', '')}</small>
              </span>
              <span role="cell" data-l="Error of RDPO">
                {r.rel === null ? <small>no run</small> : (
                  <>
                    <b>{relErr(r.rel)}</b>
                    <small>{r.budget?.toLocaleString('en-US')} paths, {r.reps} reps</small>
                  </>
                )}
              </span>
              <span role="cell" data-l="Coverage">{r.cov === null ? <small>n/a</small> : <b>{pct(r.cov, 0)}</b>}</span>
              <span role="cell" data-l="Status">
                {r.rel === null ? <Pill tone="flat">exact only</Pill> : ok ? <Pill tone="teal">{`estimated ${relErr(r.rel)}`}</Pill> : <Pill tone="warn">{`${relErr(r.rel)}, coverage ${pct(r.cov ?? 0, 0)}`}</Pill>}
              </span>
            </div>
          )
        })}
      </div>
    </section>
  )
}
