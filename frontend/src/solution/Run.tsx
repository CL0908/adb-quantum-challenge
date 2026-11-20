import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { useData, type RunResult, type Solution } from './data'
import { money, prob, seconds } from './format'
import { C, TARGETS } from './Charts'

const BUDGETS = [2048, 4096, 8192, 16384, 32768, 65536]
export const SCEN: [string, string][] = [
  ['adb-base', 'Base'], ['adb-independent', 'Independent'], ['adb-common-shock', 'Common shock'],
  ['adb-rain-stress', 'Rain stress'], ['adb-rain-cover', 'Rain cover'],
]

const fmt = (x: number | null | undefined, isMoney: boolean): ReactNode => (isMoney && x !== null && x !== undefined ? money(x) : prob(x))

function IntervalPlot({ r }: { r: RunResult }) {
  const isMoney = r.money
  const vals = [r.estimate, ...r.ci, r.mc.estimate, ...r.mc.ci, ...(r.exact ? [r.exact] : [])].filter((v) => (isMoney ? true : v > 0))
  let tx: (v: number) => number
  if (isMoney) {
    const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.12 || hi * 0.1
    tx = (v) => Math.min(1, Math.max(0, (v - (lo - pad)) / (hi - lo + 2 * pad)))
  } else {
    const lo = Math.log10(Math.min(...vals)) - 0.1, hi = Math.log10(Math.max(...vals)) + 0.1
    tx = (v) => (v <= 0 ? 0 : Math.min(1, Math.max(0, (Math.log10(v) - lo) / (hi - lo))))
  }
  const rows: [string, string, [number, number], number][] = [
    ['Plain Monte Carlo', C.grey, r.mc.ci, r.mc.estimate], ['RDPO', C.teal, r.ci, r.estimate],
  ]
  return (
    <div className="iplot" role="img" aria-label="Interval plot of the exact value, plain Monte Carlo and RDPO">
      {r.exact !== null && <div className="ip-exact" style={{ left: `${100 * tx(r.exact)}%` }}><span>exact</span></div>}
      {rows.map(([n, col, ci, est]) => (
        <div className="ip-row" key={n}>
          <i className="ip-bar" style={{ background: col, left: `${100 * tx(ci[0])}%`, width: `max(${100 * (tx(ci[1]) - tx(ci[0]))}%, 5px)` }} />
          <b className="ip-pt" style={{ borderColor: col, left: `${100 * tx(est)}%` }} />
        </div>
      ))}
    </div>
  )
}

export default function Run({ sol, scenario, setScenario }: { sol: Solution; scenario: string; setScenario: (s: string) => void }) {
  const fb = useData<Record<string, RunResult>>('fallback.json')
  const [target, setTarget] = useState('joint_exhaustion')
  const [bi, setBi] = useState(3)
  const [live, setLive] = useState<{ key: string; r: RunResult } | null>(null)
  const [status, setStatus] = useState<'idle' | 'running' | 'failed'>('idle')
  const seq = useRef(0)
  const key = `${scenario}|${target}|${BUDGETS[bi]}`
  const shown: { r: RunResult; live: boolean } | null = live && live.key === key ? { r: live.r, live: true } : fb && fb[key] ? { r: fb[key], live: false } : null
  useEffect(() => { setStatus('idle') }, [key])

  const run = useCallback(async () => {
    const my = ++seq.current
    setStatus('running')
    const ctl = new AbortController()
    const timer = setTimeout(() => ctl.abort(), 25000)
    try {
      const res = await fetch('/api/estimate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ctl.signal,
        body: JSON.stringify({ scenario, target, budget: BUDGETS[bi], threshold_musd: 80, seed: Math.floor(Math.random() * 1e6) }),
      })
      clearTimeout(timer)
      if (!res.ok || !(res.headers.get('content-type') ?? '').includes('json')) throw new Error('unavailable')
      const j = (await res.json()) as RunResult
      if (my !== seq.current) return
      setLive({ key, r: j })
      setStatus('idle')
    } catch {
      clearTimeout(timer)
      if (my === seq.current) setStatus('failed')
    }
  }, [scenario, target, bi, key])

  const r = shown?.r
  const exactMs = sol.pools[0].seconds
  return (
    <div className="run">
      <div className="run-head"><span className="lbl">Run estimate</span><span className="run-sub">one query, exact against sampling</span></div>
      <div className="run-sel">
        <label>Scenario<select className="jt-select" value={scenario} onChange={(e) => setScenario(e.target.value)}>{SCEN.map(([k, n]) => <option key={k} value={k}>{n}</option>)}</select></label>
        <label>Target<select className="jt-select" value={target} onChange={(e) => setTarget(e.target.value)}>{TARGETS.map(([k, n]) => <option key={k} value={k}>{n}</option>)}</select></label>
        <label>Budget<select className="jt-select" value={bi} onChange={(e) => setBi(Number(e.target.value))}>{BUDGETS.map((b, i) => <option key={b} value={i}>{b.toLocaleString('en-US')} paths</option>)}</select></label>
      </div>
      <button type="button" className="jt-btn solid go" onClick={run} disabled={status === 'running'}>{status === 'running' ? 'Running on the server' : 'Run estimate'}</button>
      {r ? (
        <div className="run-out">
          <div className={shown!.live ? 'src live' : 'src pre'}>{shown!.live ? 'Live result from /api/estimate' : status === 'failed' ? 'Live API unavailable, showing the precomputed run for these settings' : 'Precomputed run for these settings, press Run for a live one'}</div>
          <table className="rt">
            <tbody>
              {r.exact !== null && <tr><th><i style={{ background: C.ink }} />Exact</th><td>{fmt(r.exact, r.money)}</td><td className="d">{seconds(exactMs)}</td></tr>}
              <tr><th><i style={{ background: C.grey }} />Plain Monte Carlo</th><td>{fmt(r.mc.estimate, r.money)}</td><td className="d">{fmt(r.mc.ci[0], r.money)} to {fmt(r.mc.ci[1], r.money)}</td></tr>
              <tr className="hl"><th><i style={{ background: C.teal }} />RDPO</th><td>{fmt(r.estimate, r.money)}</td><td className="d">{fmt(r.ci[0], r.money)} to {fmt(r.ci[1], r.money)}</td></tr>
            </tbody>
          </table>
          <IntervalPlot r={r} />
          <div className="run-foot">95% intervals, {r.paths_used.toLocaleString('en-US')} paths{r.seconds !== undefined ? `, RDPO ${seconds(r.seconds)} including training` : ''}</div>
        </div>
      ) : <div className="run-out"><div className="src pre">{fb ? 'No precomputed run for these settings; press Run' : 'Loading precomputed runs'}</div></div>}
    </div>
  )
}
