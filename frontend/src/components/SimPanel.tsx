import { useCallback, useEffect, useRef, useState } from 'react'
import { useJSON, useValues, Fancy } from '../lib/data'
import { oneIn } from '../lib/model'
import Reveal from './Reveal'

interface MC {
  n: number
  exact_joint_exhaustion: number
  exact_exceed: number
  levels_musd: number[]
  law: number[][]
  cells: [number, number, number][]
  hits_joint_exhaustion: number
  hits_exceed: number
  zero_runs: number
  repeats: number
  zero_share: number
}

const SIZE = 460
const PAD = { l: 52, r: 14, t: 14, b: 46 }
const MAX = 80

function sampleCells(mc: MC): [number, number, number][] {
  // Plain Monte Carlo in the browser: draw n terms from the exact 10x10 term law of the brief's base case.
  const flat = mc.law.flat()
  const cum: number[] = []
  flat.reduce((s, p, i) => (cum[i] = s + p), 0)
  const counts = new Map<number, number>()
  for (let k = 0; k < mc.n; k++) {
    const u = Math.random() * cum[cum.length - 1]
    let lo = 0, hi = cum.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (cum[mid] < u) lo = mid + 1
      else hi = mid
    }
    counts.set(lo, (counts.get(lo) ?? 0) + 1)
  }
  return [...counts].map(([i, c]) => [mc.levels_musd[Math.floor(i / 10)], mc.levels_musd[i % 10], c])
}

export default function SimPanel() {
  const mc = useJSON<MC>('mc_sample.json')
  const v = useValues()
  const canvas = useRef<HTMLCanvasElement>(null)
  const raf = useRef(0)
  const [cells, setCells] = useState<[number, number, number][] | null>(null)
  const [shown, setShown] = useState(0)
  const [runs, setRuns] = useState({ total: 0, zero: 0 })
  const [isSeed, setIsSeed] = useState(true)

  useEffect(() => {
    if (mc) setCells(mc.cells)
  }, [mc])

  const dots = useRef<{ x: number; y: number; hit: number }[]>([])
  const draw = useCallback((upto: number) => {
    const c = canvas.current
    if (!c) return
    const dpr = window.devicePixelRatio || 1
    c.width = SIZE * dpr
    c.height = SIZE * dpr
    const g = c.getContext('2d')!
    g.scale(dpr, dpr)
    const X = (m: number) => PAD.l + ((SIZE - PAD.l - PAD.r) * m) / MAX
    const Y = (m: number) => SIZE - PAD.b - ((SIZE - PAD.t - PAD.b) * m) / MAX
    g.clearRect(0, 0, SIZE, SIZE)
    g.fillStyle = '#fbf9f4'
    g.fillRect(X(0), Y(MAX), X(MAX) - X(0), Y(0) - Y(MAX))
    // corner: both wiped out
    g.fillStyle = 'rgba(181,71,46,0.12)'
    g.fillRect(X(MAX - 10), Y(MAX), X(MAX) - X(MAX - 10), Y(MAX - 10) - Y(MAX))
    g.strokeStyle = '#B5472E'
    g.lineWidth = 1.5
    g.strokeRect(X(MAX - 10), Y(MAX), X(MAX) - X(MAX - 10), Y(MAX - 10) - Y(MAX))
    // pool pays more than US$80m: above the diagonal
    g.setLineDash([5, 4])
    g.strokeStyle = '#1F3557'
    g.lineWidth = 1
    g.beginPath(); g.moveTo(X(0), Y(MAX)); g.lineTo(X(MAX), Y(0)); g.stroke()
    g.setLineDash([])
    // axes
    g.fillStyle = '#5b6678'
    g.font = '12px "Plus Jakarta Sans", system-ui, sans-serif'
    g.textAlign = 'center'
    for (const t of [0, 20, 40, 60, 80]) {
      g.fillText(String(t), X(t), SIZE - PAD.b + 16)
      g.textAlign = 'right'
      g.fillText(String(t), PAD.l - 8, Y(t) + 4)
      g.textAlign = 'center'
    }
    g.fillText('Kyrgyz Republic loss (US$m)', (X(0) + X(MAX)) / 2, SIZE - 10)
    g.save(); g.translate(13, (Y(0) + Y(MAX)) / 2); g.rotate(-Math.PI / 2); g.fillText('Tajikistan loss (US$m)', 0, 0); g.restore()
    // simulated terms
    const d = dots.current
    for (let i = 0; i < Math.min(upto, d.length); i++) {
      const p = d[i]
      g.fillStyle = p.hit ? '#B5472E' : p.x + p.y > MAX + 1e-6 ? 'rgba(181,71,46,0.75)' : 'rgba(31,53,87,0.22)'
      g.beginPath(); g.arc(X(p.x), Y(p.y), p.hit ? 5 : 1.7, 0, 6.2832); g.fill()
    }
    g.fillStyle = '#B5472E'
    g.font = '600 12px "Plus Jakarta Sans", system-ui, sans-serif'
    g.textAlign = 'right'
    g.fillText('both wiped out', X(MAX - 10) - 6, Y(MAX) + 14)
    g.fillStyle = '#1F3557'
    g.textAlign = 'left'
    g.fillText('dashed line: pool pays US$80m', X(6), Y(MAX) + 14)
  }, [])

  // Expand cell counts into jittered dots (the loss states are discrete, so jitter keeps overlapping terms visible).
  useEffect(() => {
    if (!cells) return
    const out: { x: number; y: number; hit: number }[] = []
    for (const [a, b, n] of cells) {
      for (let k = 0; k < n; k++) {
        const hit = a >= MAX - 1e-9 && b >= MAX - 1e-9 ? 1 : 0
        out.push({ x: hit ? a : a + (a === 0 ? Math.random() * 3.4 : (Math.random() - 0.5) * 3.4), y: hit ? b : b + (b === 0 ? Math.random() * 3.4 : (Math.random() - 0.5) * 3.4), hit })
      }
    }
    for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [out[i], out[j]] = [out[j], out[i]] }
    dots.current = out
    const total = out.length
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    cancelAnimationFrame(raf.current)
    const t0 = performance.now()
    const step = (t: number) => {
      const f = reduce ? 1 : Math.min((t - t0) / 1800, 1)
      const n = Math.round(total * f * f * (3 - 2 * f))
      setShown(n)
      draw(n)
      if (f < 1) raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [cells, draw])

  const hits = (cs: [number, number, number][] | null) => (cs ?? []).reduce((s, [a, b, n]) => s + (a >= MAX - 1e-9 && b >= MAX - 1e-9 ? n : 0), 0)
  const hitsNow = hits(cells)
  const total = cells ? cells.reduce((s, c) => s + c[2], 0) : 0
  const done = shown >= total && total > 0

  const again = () => {
    if (!mc) return
    const cs = sampleCells(mc)
    setIsSeed(false)
    setRuns((r) => ({ total: r.total + 1, zero: r.zero + (hits(cs) === 0 ? 1 : 0) }))
    setCells(cs)
  }

  return (
    <section className="sec sec-white" id="problem">
      <div className="wrap">
        <Reveal>
          <p className="eyebrow">The problem</p>
          <h2 className="h2">The case that matters is too rare for simulation to see.</h2>
          <p className="lede">
            Each dot is one simulated three-year term for the two bonds. Almost all land near the bottom left. The corner where both countries use up their cover is the one a pool must be able to survive, and simulation hardly ever reaches it.
          </p>
        </Reveal>
        <Reveal className="sim" delay={80}>
          <div className="sim-plot">
            <canvas ref={canvas} style={{ width: '100%', maxWidth: SIZE, aspectRatio: '1' }} role="img"
              aria-label={`Scatter of ${total.toLocaleString('en-US')} simulated terms. ${hitsNow} landed in the corner where both bonds are wiped out.`} />
          </div>
          <div className="sim-side">
            <div className="sim-stat">
              <div className="k">Simulated terms</div>
              <div className="n">{shown.toLocaleString('en-US')}<span> of {total ? total.toLocaleString('en-US') : '…'}</span></div>
            </div>
            <div className="sim-stat">
              <div className="k">Landed where both are wiped out</div>
              <div className={`n ${done && hitsNow === 0 ? 'terra' : ''}`}>{done ? hitsNow : '…'}</div>
            </div>
            <div className="sim-exact">
              <div className="k">The exact answer, computed directly</div>
              <div className="n">{mc ? oneIn(mc.exact_joint_exhaustion) : '…'}</div>
              <p>in {v.EQT_EXACT_MS ? <Fancy s={v.EQT_EXACT_MS} /> : '…'} on one CPU core, no sampling error.</p>
            </div>
            <p className="sim-claim">Plain Monte Carlo returns zero in about <b>{v.MC_ZERO ?? '…'}</b> of runs at this budget.</p>
            <div className="sim-actions">
              <button type="button" className="btn-dark" onClick={again} disabled={!mc}>Simulate another {mc ? mc.n.toLocaleString('en-US') : ''}</button>
              {runs.total > 0 && <span className="tally" role="status">{runs.zero} of {runs.total} new runs saw nothing in the corner.</span>}
            </div>
            <p className="fineprint">
              {isSeed && mc
                ? <>Shown: one real run of the repository’s sampler (seed 0). Over {mc.repeats} such runs, {(100 * mc.zero_share).toFixed(0)}% found nothing.</>
                : <>New runs draw terms in your browser from the same exact model.</>}
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  )
}
