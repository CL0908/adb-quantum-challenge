import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { useJSON } from '../lib/data'

const BUDGETS = [2048, 4096, 8192, 16384, 32768, 65536]
const SCEN: [string, string][] = [
  ['adb-base', 'Base'], ['adb-independent', 'Independent (no shared shocks)'], ['adb-common-shock', 'Common shock (shared earthquakes ×2)'],
  ['adb-rain-stress', 'Rain stress (rain rates ×2)'], ['adb-rain-cover', 'Rain cover (US$60m / US$20m limits)'],
]
const TARGETS: [string, string][] = [
  ['joint_exhaustion', 'Both bonds exhausted'], ['joint_attachment', 'Both bonds pay'], ['exceedance', 'Pool loss above US$80m'],
  ['stop_loss', 'Pool stop-loss above US$80m'], ['expected_loss', 'Expected principal loss'],
]

interface Result {
  estimate: number; ci: [number, number]; exact: number; money: boolean; paths_used: number; seconds?: number
  mc: { estimate: number; ci: [number, number]; hits: number }
}

const sup = (e: number) => <sup>{String(e).replace('-', '−')}</sup>
function fmt(x: number | null | undefined, money: boolean): ReactNode {
  if (x === null || x === undefined || Number.isNaN(x)) return '–'
  if (money) return 'US$' + (x >= 100 ? x.toFixed(1) : x.toFixed(3)) + 'm'
  if (x === 0) return '0'
  if (x >= 1e-3) return (x * 100).toPrecision(3) + '%'
  const e = Math.floor(Math.log10(x))
  return <>{(x / Math.pow(10, e)).toPrecision(3)}×10{sup(e)}</>
}
const rangeOf = (c: [number, number], money: boolean) => <>95% interval {fmt(c[0], money)} to {fmt(c[1], money)}</>

/** Interval plot: log axis for probabilities, linear for money. Returns the layout in percentages. */
function layout(r: Result, money: boolean) {
  const vals = [r.exact, r.estimate, ...r.ci, ...r.mc.ci, r.mc.estimate]
  let tx: (v: number) => number
  const ticks: { p: number; label: ReactNode }[] = []
  if (!money) {
    const pv = vals.filter((v) => v > 0)
    let l = Math.floor(Math.log10(Math.min(...pv)) - 0.15)
    const h = Math.min(0, Math.ceil(Math.log10(Math.max(...pv)) + 0.05))
    if (h - l < 1) l = h - 1
    tx = (v) => (v <= 0 ? 0 : Math.min(1, Math.max(0, (Math.log10(v) - l) / (h - l))))
    const step = Math.max(1, Math.ceil((h - l) / 6))
    for (let e = l; e <= h; e += step) ticks.push({ p: tx(Math.pow(10, e)), label: <>10{sup(e)}</> })
  } else {
    let lo = Math.min(...vals), hi = Math.max(...vals)
    const pad = (hi - lo) * 0.18 || hi * 0.1
    lo = Math.max(0, lo - pad); hi += pad
    tx = (v) => Math.min(1, Math.max(0, (v - lo) / (hi - lo)))
    const raw = (hi - lo) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw)))
    const st = [1, 2, 5, 10].map((k) => k * mag).find((k) => k >= raw)!
    for (let t = Math.ceil(lo / st) * st; t <= hi; t += st) ticks.push({ p: tx(t), label: 'US$' + t.toFixed(st < 0.1 ? 3 : st < 1 ? 2 : 1) + 'm' })
  }
  return { tx, ticks }
}

function Row({ name, sub, color, est, ci, money, extra }: { name: string; sub: string; color: string; est: number; ci: [number, number]; money: boolean; extra?: ReactNode }) {
  return (
    <div className="mrow" style={{ ['--c' as string]: color }}>
      <div className="nm">{name}<small>{sub}</small></div>
      <div className="val">{fmt(est, money)}</div>
      <div className="ci">{rangeOf(ci, money)}{extra ? <><br />{extra}</> : null}</div>
    </div>
  )
}

function Plot({ r }: { r: Result }) {
  const money = r.money
  const { tx, ticks } = layout(r, money)
  const bar = (y: number, name: string, color: string, ci: [number, number], est: number) => {
    const a = tx(ci[0]), b = tx(ci[1]), clipped = !money && ci[0] <= 0
    return (
      <>
        <div className="rl" style={{ top: y - 22, color }}>{name}</div>
        <div className="iv" style={{ ['--c' as string]: color, left: `${100 * a}%`, width: `max(${100 * (b - a)}%, 8px)`, top: y - 6 }} />
        {clipped && <div className="arrowl" style={{ ['--c' as string]: color, left: `${100 * a}%`, top: y }}>◀ 0</div>}
        {(est > 0 || money) && <div className="pt" style={{ ['--c' as string]: color, left: `${100 * tx(est)}%`, top: y }} />}
      </>
    )
  }
  return (
    <div className="plot" role="img" aria-label={`Interval plot. Exact ${money ? r.exact.toFixed(3) + ' million US dollars' : (r.exact * 100).toPrecision(3) + ' percent'}. RDPO interval narrower than Monte Carlo's.`}>
      <div className="axis">
        {ticks.map((t, i) => (
          <span key={i}><div className="tick" style={{ left: `${100 * t.p}%` }} /><div className="tl" style={{ left: `${100 * t.p}%` }}>{t.label}</div></span>
        ))}
        {bar(52, 'Plain Monte Carlo', '#4F5563', r.mc.ci, r.mc.estimate)}
        {bar(104, 'RDPO', '#C8553A', r.ci, r.estimate)}
        <div className="ex" style={{ left: `${100 * tx(r.exact)}%` }} />
        <div className="exl" style={{ left: `${100 * tx(r.exact)}%` }}>exact</div>
      </div>
    </div>
  )
}

export default function TryIt() {
  const fallback = useJSON<Record<string, Result>>('fallback.json')
  const fbRef = useRef<Record<string, Result> | null>(null)
  useEffect(() => { fbRef.current = fallback }, [fallback])
  const [scenario, setScenario] = useState('adb-base')
  const [target, setTarget] = useState('joint_exhaustion')
  const [bi, setBi] = useState(3)
  const [res, setRes] = useState<{ r: Result; live: boolean; seed: boolean } | null>(null)
  const [status, setStatus] = useState('Loading…')
  const [warn, setWarn] = useState(false)
  const seq = useRef(0)

  const run = useCallback(async (randomSeed: boolean) => {
    const my = ++seq.current
    const budget = BUDGETS[bi]
    setStatus('Running…'); setWarn(false)
    const body = { scenario, target, budget, threshold_musd: 80, seed: randomSeed ? Math.floor(Math.random() * 1e6) : 0 }
    const ctl = new AbortController()
    const timer = setTimeout(() => ctl.abort(), 20000)
    try {
      const r = await fetch('api/estimate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: ctl.signal })
      clearTimeout(timer)
      if (!r.ok || !(r.headers.get('content-type') ?? '').includes('json')) throw new Error('function unavailable')
      const j = (await r.json()) as Result
      if (my !== seq.current) return
      setRes({ r: j, live: true, seed: randomSeed })
      setStatus(`Live: computed on the server in ${(j.seconds ?? 0).toFixed(2)} s (RDPO, training included)${randomSeed ? ', new random seed.' : '.'}`)
    } catch {
      clearTimeout(timer)
      let fb = fbRef.current
      if (!fb) {
        try { fb = (await (await fetch('data/fallback.json')).json()) as Record<string, Result> } catch { fb = null }
      }
      if (my !== seq.current) return
      const j = fb?.[`${scenario}|${target}|${budget}`]
      if (!j) { setWarn(true); setStatus('Could not compute this case. Please try again.'); return }
      setRes({ r: j, live: false, seed: false })
      setWarn(true)
      setStatus('The live function is not reachable from this page, so this is a precomputed answer (seed 0, same code and settings).')
    }
  }, [scenario, target, bi])

  useEffect(() => { void run(false) }, [run])

  const r = res?.r
  const money = r?.money ?? false
  const wMC = r ? r.mc.ci[1] - r.mc.ci[0] : 0, wR = r ? r.ci[1] - r.ci[0] : 0
  const inside = r ? r.exact >= r.ci[0] && r.exact <= r.ci[1] : false
  const ratio = wR > 0 ? wMC / wR : Infinity

  return (
    <section className="sec" id="try">
      <div className="wrap">
        <p className="eyebrow">Try the estimator</p>
        <h2 className="h2">Pick a scenario from the brief and compare three estimators.</h2>
        <p className="lede">The two-country benchmark, solved live: the exact answer, a plain Monte Carlo run, and our learned sampler (RDPO) with the same number of simulated paths. Pool questions use the brief’s US$80m threshold.</p>
        <div className="try-panel">
          <form className="try-controls" onSubmit={(e) => { e.preventDefault(); void run(true) }} autoComplete="off">
            <label>Scenario
              <select value={scenario} onChange={(e) => setScenario(e.target.value)}>{SCEN.map(([k, n]) => <option key={k} value={k}>{n}</option>)}</select></label>
            <label>Target
              <select value={target} onChange={(e) => setTarget(e.target.value)}>{TARGETS.map(([k, n]) => <option key={k} value={k}>{n}</option>)}</select></label>
            <label><span>Budget: <b>{BUDGETS[bi].toLocaleString('en-US')}</b> paths</span>
              <input type="range" min={0} max={5} step={1} value={bi} onChange={(e) => setBi(+e.target.value)} /></label>
            <button className="btn-dark" type="submit">Run again</button>
          </form>
          <div className="try-result" aria-live="polite">
            <div className={`status ${warn ? 'warn' : ''}`}>{status}</div>
            {r && (
              <>
                <Row name="Exact" sub="dense enumeration" color="#1F3557" est={r.exact} ci={[r.exact, r.exact]} money={money} extra="no sampling error" />
                <Row name="Plain Monte Carlo" sub={`${r.paths_used.toLocaleString('en-US')} paths`} color="#4F5563" est={r.mc.estimate} ci={r.mc.ci} money={money}
                  extra={money ? null : r.mc.hits === 0 ? 'No path hit the event.' : `${r.mc.hits} of ${r.paths_used.toLocaleString('en-US')} paths hit the event.`} />
                <Row name="RDPO" sub={`${r.paths_used.toLocaleString('en-US')} paths, training included`} color="#C8553A" est={r.estimate} ci={r.ci} money={money} />
                <Plot r={r} />
                <p className="note">
                  RDPO’s interval is {Number.isFinite(ratio) ? ratio.toFixed(ratio >= 10 ? 0 : 1) : '∞'}× narrower than plain Monte Carlo’s on the same budget
                  {inside ? ' and contains the exact value.' : '; this run’s interval misses the exact value (about 1 run in 20 should).'} At equal CPU time simple methods win many cells; see honest accounting above.
                </p>
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
