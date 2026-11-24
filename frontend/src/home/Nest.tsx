import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { useJSON } from './lib/data'
import { SvgTapHint, TapHint } from '../shared/TapHint'

/* "Three models, one inside the other": the small world inside RDPO inside the twin. Two views of the same thing,
   nested cards and the relation chain; picking a model shows what it does for its neighbour. Numbers come from
   data/nest.json, written by addons/export_web_values.py from the committed add-on results. */

type Nest = {
  rho2: { twisting: number; guide: number; guide_tables: number }
  equal_cpu_gain: number
  answer_over_truth: { guide: number; guide_tables: number }
  rdpo_z_max: number
  twin: Record<'exceedance' | 'joint_exhaustion', { fixed: number; uncertain: number; rel: number }>
}
type Id = 'sw' | 'rdpo' | 'twin'

const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹'
const sup = (n: number) => (n < 0 ? '⁻' : '') + String(Math.abs(n)).split('').map((c) => SUP[+c]).join('')
const sci = (x: number) => { const e = Math.floor(Math.log10(x)); return `${(x / 10 ** e).toFixed(1)}×10${sup(e)}` }
const pc = (x: number) => `${(x * 100).toFixed(2)}%`
const x1 = (x: number) => `${x < 10 ? x.toFixed(x < 1.01 ? 3 : 1) : Math.round(x)}×`

function Row({ label, value, pct, tone }: { label: ReactNode; value: string; pct: number; tone?: 'bad' | 'good' }) {
  return (
    <div className={`nb ${tone ?? ''}`}>
      <div className="nb-l"><span>{label}</span><b>{value}</b></div>
      <div className="nb-t"><i style={{ width: `${Math.max(2, Math.min(100, pct))}%` }} /></div>
    </div>
  )
}

function Panel({ id, n }: { id: Id; n: Nest }) {
  if (id === 'sw') {
    const lg = (v: number) => 6 + (94 * Math.log10(v)) / Math.log10(n.rho2.twisting)
    return (
      <>
        <span className="nk">Small world &rarr; RDPO</span>
        <h4>A guide table that keeps RDPO steady</h4>
        <p className="nsub">How much the answer jumps between runs (28-country ring, lower is better, log scale)</p>
        <div className="nbars">
          <Row label="Standard method (twisting)" value={n.rho2.twisting.toFixed(0)} pct={lg(n.rho2.twisting)} tone="bad" />
          <Row label="Small world guide" value={n.rho2.guide.toFixed(1)} pct={lg(n.rho2.guide)} />
          <Row label="Small world + cap tables" value={n.rho2.guide_tables.toFixed(2)} pct={lg(n.rho2.guide_tables)} tone="good" />
        </div>
        <p className="nfoot">No training needed. At the same CPU time it is <b>{n.equal_cpu_gain.toFixed(1)}&times;</b> as efficient as twisting.</p>
      </>
    )
  }
  if (id === 'rdpo') {
    const top = n.answer_over_truth.guide
    return (
      <>
        <span className="nk">RDPO &rarr; small world</span>
        <h4>RDPO fixes what the small world cannot see</h4>
        <p className="nsub">Each answer compared with a very long reference run (2 million paths) on the real, capped 28-country ring. 1&times; means they agree.</p>
        <div className="nbars">
          <Row label="Small world on its own" value={x1(top)} pct={100} tone="bad" />
          <Row label="With cap tables (from a short trial run, its cost counted)" value={x1(n.answer_over_truth.guide_tables)} pct={(100 * n.answer_over_truth.guide_tables) / top} />
        </div>
        <p className="nfoot">The small world has no payout caps, so it overstates the tail. RDPO samples the real pool and reweights every path, so nothing is biased: even with the plain table it lands within <b>{n.rdpo_z_max.toFixed(2)} standard errors</b> of that reference, well inside normal run-to-run noise.</p>
      </>
    )
  }
  const t = n.twin
  return (
    <>
      <span className="nk">Twin &rarr; RDPO</span>
      <h4>The twin adds what RDPO takes as known</h4>
      <p className="nsub">Two-country bond, inputs fixed at their calibration against inputs drawn from the data (bars compare within each pair)</p>
      <div className="npair">
        <div><h5>Pool pays more than US$80m</h5>
          <Row label="Inputs fixed" value={pc(t.exceedance.fixed)} pct={(100 * t.exceedance.fixed) / t.exceedance.uncertain} />
          <Row label="Inputs uncertain" value={pc(t.exceedance.uncertain)} pct={100} tone="bad" /></div>
        <div><h5>Both bonds exhausted</h5>
          <Row label="Inputs fixed" value={sci(t.joint_exhaustion.fixed)} pct={(100 * t.joint_exhaustion.fixed) / t.joint_exhaustion.uncertain} />
          <Row label="Inputs uncertain" value={sci(t.joint_exhaustion.uncertain)} pct={100} tone="bad" /></div>
      </div>
      <p className="nfoot">RDPO answers for one world. Counting how unsure the inputs are raises the pool tail by <b>{Math.round(t.exceedance.rel * 100)}%</b> and the joint exhaustion by <b>{Math.round(t.joint_exhaustion.rel * 100)}%</b>.</p>
    </>
  )
}

/* a short loop per model (frontend/animations, Manim); with reduced motion it stays on its poster with controls */
const ANIM: Record<Id, { src: string; label: string }> = {
  sw: { src: 'anim/smallworld', label: 'Animation: the 28-country pool with its payout caps removed is solved exactly in one pass; on its own its answer is 45 times a 2-million-path reference for the capped pool, with cap tables 1.002 times.' },
  rdpo: { src: 'anim/rdpo', label: 'Animation: plain Monte Carlo puts no paths in the corner where both bonds are exhausted; RDPO sends paths there and reweights them, and its estimate lands on the exact two-country value.' },
  twin: { src: 'anim/twin', label: 'Animation: with the inputs fixed the pool pays more than US$80m with probability 1.20%; averaged over many possible worlds it is 1.89%.' },
}
function Anim({ id }: { id: Id }) {
  const v = useRef<HTMLVideoElement>(null)
  // decided before the first render: with reduced motion nothing loads and the poster (the scene's result) stays on screen
  const [still] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  useEffect(() => { if (!still) v.current?.play().catch(() => undefined) }, [id, still])
  const a = ANIM[id]
  return (
    <video ref={v} className="nvid" key={id} muted loop playsInline autoPlay={!still} controls preload={still ? 'none' : 'metadata'} poster={`${a.src}.webp`} aria-label={a.label}>
      <source src={`${a.src}.webm`} type="video/webm" />
      <source src={`${a.src}.mp4`} type="video/mp4" />
    </video>
  )
}

const LAYERS: { id: Id; n: string; t: string; p: string; q: string; go: string }[] = [
  { id: 'twin', n: '03', t: 'Twin', p: 'The inputs are uncertain, so count many possible versions of them.', q: 'Which world are we in? Many possible worlds, updated with the data.', go: 'Adds to RDPO' },
  { id: 'rdpo', n: '02', t: 'RDPO', p: 'Count carefully where the rare loss happens, then correct for the shortcut.', q: 'What is the probability in this world? Sampling on the real pool.', go: 'Fixes the small world' },
  { id: 'sw', n: '01', t: 'Small world', p: 'A rough map first: where both countries can be hit at once.', q: 'Where should the paths go? The pool without payout caps, solved exactly in one pass.', go: 'Guides RDPO' },
]

function Nested({ sel, pick, hint }: { sel: Id; pick: (i: Id) => void; hint?: Id }) {
  const layer = (i: number): ReactNode => {
    const L = LAYERS[i]
    if (!L) return null
    return (
      <div className={`nl nl-${L.id} ${sel === L.id ? 'on' : ''}`}>
        <button type="button" className={hint === L.id ? 'nh th-glow' : 'nh'} aria-pressed={sel === L.id} onClick={() => pick(L.id)}>
          <span className="sn">{L.n}</span>
          <span className="nt"><b>{L.t}</b><span>{L.p}</span><small>{L.q}</small></span>
          <span className={hint === L.id ? 'go th-host' : 'go'}>{L.go}{hint === L.id && <TapHint />}</span>
        </button>
        {layer(i + 1)}
      </div>
    )
  }
  return <div className="nested">{layer(0)}</div>
}

/* chain: twin -> small world (world parameters), small world -> RDPO (guide table); dashed, two-country only:
   RDPO -> twin (crisis anatomy), twin -> RDPO (how precise is enough) */
const LINKS: Record<Id, string[]> = { sw: ['ts', 'sr'], rdpo: ['sr', 'rt', 'tr'], twin: ['ts', 'rt', 'tr'] }

// node boxes (top-left corners) in the wide and tall drawings, for the tapping hand
const NODE_AT: Record<'wide' | 'tall', Record<Id, [number, number]>> = {
  wide: { twin: [20, 152], sw: [430, 152], rdpo: [860, 152] },
  tall: { twin: [46, 16], sw: [46, 208], rdpo: [46, 404] },
}
// the dashed RDPO <-> twin loop gets its own dot going round it (drawn under the nodes, so it enters and leaves them),
// warm so it is not read as the teal main flow; the stylesheet's reduced-motion rule still stops it
const loopDot = (d: string) => <circle className="flow" r={6} style={{ offsetPath: `path('${d}')`, fill: 'var(--dk-sand)', filter: 'drop-shadow(0 0 6px var(--dk-sand))', animationDuration: '7s', animationDelay: '4.4s' }} />
const hand = (hint: Id | undefined, v: 'wide' | 'tall') => hint && <SvgTapHint x={NODE_AT[v][hint][0] + 170} y={NODE_AT[v][hint][1] + 66} size={v === 'wide' ? 58 : 44} />

function Chain({ sel, pick, hint }: { sel: Id; pick: (i: Id) => void; hint?: Id }) {
  const lit = (e: string) => (LINKS[sel].includes(e) ? 'lit' : 'dim')
  const node = (id: Id, x: number, y: number, w: number, t: string, s: string, d: number) => (
    <g style={{ '--d': `${d}s` } as CSSProperties} className={`cn cn-${id} ${sel === id ? 'on' : 'dim'}`} role="button" tabIndex={0} aria-pressed={sel === id} aria-label={t}
      onClick={() => pick(id)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(id) } }}>
      <rect x={x} y={y} width={w} height={96} rx={18} />
      <text className="ct" x={x + 20} y={y + 38}>{t}</text>
      <text className="cs" x={x + 20} y={y + 62}>{s}</text>
    </g>
  )
  const edge = (id: string, d: string, dashed: boolean, lines: string[], lx: number, ly: number, anchor: 'middle' | 'start' = 'middle', t = 0) => (
    <g style={{ '--d': `${t}s` } as CSSProperties} className={`ce ${lit(id)} ${dashed ? 'dsh' : 'sol'}`}>
      <path d={d} pathLength={dashed ? undefined : 1} strokeDasharray={dashed ? '6 6' : undefined} markerEnd={`url(#${lit(id) === 'lit' ? 'nah-l' : 'nah'})`} />
      {lines.map((s, i) => <text key={s} className={i ? '' : 'b'} x={lx} y={ly + 17 * i} textAnchor={anchor}>{s}</text>)}
    </g>
  )
  const defs = (
    <defs>
      <marker id="nah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" className="mk" /></marker>
      <marker id="nah-l" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" className="mk-l" /></marker>
    </defs>
  )
  return (
    <>
      <svg className="chain wide" viewBox="-140 0 1380 400" role="group" aria-label="Relation chain: earthquake data goes into the twin, which gives the small world the current inputs; the small world gives RDPO a guide table and RDPO gives the bond price. the small world gives RDPO a guide table; RDPO tells the twin what a crisis looks like, and the twin tells RDPO how precise is enough.">
        {defs}
        <g className="io" style={{ '--d': '0s' } as CSSProperties}><path d="M-128,200 L14,200" pathLength={1} markerEnd="url(#nah-l)" /><text x={-57} y={186} textAnchor="middle">Earthquake data</text></g>
        {edge('ts', 'M240,200 L426,200', false, ['Current inputs', 'updated in 1–2 s'], 333, 236, 'middle', 0.9)}
        {edge('sr', 'M650,200 L856,200', false, ['Where to look', 'a guide table'], 753, 236, 'middle', 1.9)}
        <g className="io" style={{ '--d': '2.9s' } as CSSProperties}><path d="M1084,200 L1226,200" pathLength={1} markerEnd="url(#nah-l)" /><text x={1155} y={186} textAnchor="middle">Bond price</text></g>
        {edge('rt', 'M970,152 C970,40 130,40 130,148', true, ['What a crisis looks like'], 550, 26, 'middle', 3.5)}
        {edge('tr', 'M130,248 C130,370 970,370 970,252', true, ['How precise is enough'], 550, 362, 'middle', 4)}
        {loopDot('M970,152 C970,40 130,40 130,148 L130,248 C130,370 970,370 970,252 Z')}
        {node('twin', 20, 152, 220, 'Twin', 'Which inputs are right?', 0.4)}
        {node('sw', 430, 152, 220, 'Small world', 'A rough map', 1.4)}
        {node('rdpo', 860, 152, 220, 'RDPO', 'The careful count', 2.4)}
        <circle className="flow" r={5} style={{ offsetPath: "path('M240,200 L426,200 M650,200 L856,200')" }} />
        {hand(hint, 'wide')}
      </svg>
      <svg className="chain tall" viewBox="-24 -60 364 640" role="group" aria-label="The same relation chain, top to bottom.">
        {defs}
        <g className="io" style={{ '--d': '0s' } as CSSProperties}><path d="M156,-46 L156,12" pathLength={1} markerEnd="url(#nah-l)" /><text x={168} y={-20}>Earthquake data</text></g>
        {edge('ts', 'M156,112 L156,208', false, ['Current inputs', 'updated in 1–2 s'], 168, 152, 'start', 0.9)}
        {edge('sr', 'M156,304 L156,400', false, ['Where to look', 'a guide table'], 168, 344, 'start', 1.9)}
        <g className="io" style={{ '--d': '2.9s' } as CSSProperties}><path d="M156,500 L156,560" pathLength={1} markerEnd="url(#nah-l)" /><text x={168} y={536}>Bond price</text></g>
        {edge('rt', 'M266,452 C334,400 334,110 270,66', true, [], 0, 0, 'middle', 3.5)}
        {edge('tr', 'M46,64 C-20,120 -20,400 42,452', true, [], 0, 0, 'middle', 4)}
        {loopDot('M266,452 C334,400 334,110 270,66 L46,64 C-20,120 -20,400 42,452 Z')}
        {node('twin', 46, 16, 220, 'Twin', 'Which inputs are right?', 0.4)}
        {node('sw', 46, 208, 220, 'Small world', 'A rough map', 1.4)}
        {node('rdpo', 46, 404, 220, 'RDPO', 'The careful count', 2.4)}
        {hand(hint, 'tall')}
      </svg>
      <ul className="ch-dash">
        <li><b>RDPO &rarr; twin</b> what a crisis looks like, how close it is, which input matters most</li>
        <li><b>Twin &rarr; RDPO</b> how precise is enough, from how much the answer moves with the uncertain inputs</li>
      </ul>
      <div className="ch-key"><span><i />Tested on pools up to 28 countries</span><span><i className="d" />Tested on the two-country bond</span></div>
    </>
  )
}

export default function NestBlock() {
  const n = useJSON<Nest>('nest.json')
  const [view, setView] = useState<'nested' | 'chain'>('nested')
  const [sel, setSel] = useState<Id>('sw')
  // a tapping hand walks the reader outward, small world -> RDPO -> twin, and leaves once each model has been opened
  const [opened, setOpened] = useState<Id[]>(['sw'])
  const pick = (i: Id) => { setSel(i); setOpened((v) => (v.includes(i) ? v : [...v, i])) }
  const hint = (['rdpo', 'twin'] as Id[]).find((i) => !opened.includes(i))
  return (
    <div className="nest">
      <div className="nest-head">
        <h3>Three models, <em>one inside the other</em></h3>
        <p>Each outer model fixes what the inner one cannot see. Each inner model makes the outer one cheaper. Pick a model to see what it does for its neighbour.</p>
        <div className="nest-seg" role="group" aria-label="View">
          <button type="button" className={view === 'nested' ? 'on' : ''} aria-pressed={view === 'nested'} onClick={() => setView('nested')}>Nested</button>
          <button type="button" className={view === 'chain' ? 'on' : ''} aria-pressed={view === 'chain'} onClick={() => setView('chain')}>Chain</button>
        </div>
      </div>
      <div className={`nest-body ${view}`}>
        <div className="nest-fig">{view === 'nested' ? <Nested sel={sel} pick={pick} hint={hint} /> : <Chain sel={sel} pick={pick} hint={hint} />}</div>
        <div className="ncomp" aria-live="polite">{hint && <p className="th-note">Click another model to play its animation ({3 - opened.length} left)</p>}{n ? <div key={sel}><Anim id={sel} /><Panel id={sel} n={n} /></div> : <div className="bskel" />}</div>
      </div>
      <p className="nest-foot">28-country ring pool and the two-country bond. Every number comes from a committed run.</p>
    </div>
  )
}
