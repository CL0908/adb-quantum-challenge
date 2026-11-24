import { Suspense, lazy, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { Pin, Tally } from './Hero3D'
import { useJSON, useStory, useValues, V, ValuesProvider } from './lib/data'
import NestBlock from './Nest'
import { webglAvailable } from './lib/dem'
import type { MapData } from './lib/map'
import Reveal, { useInView } from './Reveal'
import { Bar, Checks, Heat, Ring, Sources, TimeLine } from './Visuals'
import { HERO_ASPECT, PROBLEM_ASPECT } from './Hero3D.config'
import { SiteNav } from '../shared/SiteNav'
import '../shared/theme.css'
import './home.css'
import './visuals.css'

const Hero3D = lazy(() => import('./Hero3D'))

const PDFS = {
  proposal: 'RDPO_Proposal_Short.pdf',
  annex: 'RDPO_Proposal.pdf',
  deck: 'RDPO_Presentation.pdf',
}
const REPO = 'https://github.com/CL0908/adb-quantum-challenge'
const SOLUTION = '/solution/'
const DASHBOARD = '/dashboard/'

/* ---------- small pieces ---------- */

function Term({ name, children }: { name: string; children: ReactNode }) {
  return <span className="term" tabIndex={0}>{name}<span className="tip" role="tooltip"><b>{name}</b>{children}</span></span>
}

function Eyebrow({ children, dark = false }: { children: ReactNode; dark?: boolean }) {
  return <span className={`eyebrow ${dark ? 'on-dark' : ''}`}><i aria-hidden="true" />{children}</span>
}

function Card({ title, sub, children, className = '' }: { title: string; sub?: string; children: ReactNode; className?: string }) {
  const [ref, shown] = useInView<HTMLElement>()
  return (
    <article ref={ref} className={`bcard ${shown ? 'in' : ''} ${className}`}>
      <header className="bhead">
        <h3>{title}</h3>
        {sub && <p>{sub}</p>}
      </header>
      <div className="bvis">{children}</div>
    </article>
  )
}

function Arrow() {
  return <svg className="arr" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3 8h10M9 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
}

/* ---------- navigation ---------- */

/* ---------- hero ---------- */

function HeroFigure() {
  const map = useJSON<MapData>('map.json')
  const [gl, setGl] = useState(false)
  const [live, setLive] = useState(false)
  const [tally, setTally] = useState<Tally | null>(null)
  const pinEls = useRef<Record<string, HTMLElement | null>>({})
  const hold = useRef(false)
  const oneIn = useOneIn()
  useEffect(() => { setGl(webglAvailable()) }, [])
  const shared = map?.quakes.filter((q) => q[4] === 1).length
  // three places on the island that lead somewhere: the shared zone (why it matters), a country (try the model), the pool
  const pins = useMemo<Pin[]>(() => {
    if (!map) return []
    const z = map.zone[0]
    return [
      { id: 'zone', lon: z.reduce((a, p) => a + p[0], 0) / z.length, lat: z.reduce((a, p) => a + p[1], 0) / z.length },
      { id: 'model', lon: 74.6, lat: 41.4 },
      { id: 'pool', lon: 78.6, lat: 42.0 },
    ]
  }, [map])
  return (
    <figure className={`hero-fig ${live ? 'live' : ''}`} style={{ ['--ar' as string]: HERO_ASPECT }}>
      <div className="hero-shadow" aria-hidden="true" />
      <div className="hero-frame">
        <img className="hero-still" src="hero-terrain.webp" alt="The Tian Shan and Pamir cut from real elevation data as a floating terrain block, with the Kyrgyz Republic and Tajikistan, earthquake main shocks as thin columns and the shared border zone glowing." width={2400} height={Math.round(2400 / HERO_ASPECT)} fetchPriority="high" decoding="async" />
        {gl && map && (
          <Suspense fallback={null}>
            <Hero3D map={map} onReady={() => setLive(true)} onFail={() => setLive(false)} onYear={setTally} pins={pins} pinEls={pinEls} hold={hold} />
          </Suspense>
        )}
        {live && (
          <div className="pins" onPointerOver={() => { hold.current = true }} onPointerOut={() => { hold.current = false }} onFocus={() => { hold.current = true }} onBlur={() => { hold.current = false }}>
            <a className="hpin" href="#problem" ref={(e) => { pinEls.current.zone = e }} onClick={(e) => { e.preventDefault(); document.getElementById('problem')?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' }) }}><i aria-hidden="true" /><span><b className="l">Shared border zone</b><b className="s">Shared zone</b><small>Both bonds exhausted {oneIn ? <>1 in {oneIn.toLocaleString('en-US')}</> : ''} &darr;</small></span></a>
            <a className="hpin" href={SOLUTION} ref={(e) => { pinEls.current.model = e }}><i aria-hidden="true" /><span><b className="l">Try the two-country model</b><b className="s">Try the model</b><small>Dependence, cards, a live run &rarr;</small></span></a>
            <a className="hpin" href="/dashboard/" ref={(e) => { pinEls.current.pool = e }}><i aria-hidden="true" /><span><b className="l">Banker view</b><b className="s">Banker view</b><small>Scenarios and key figures &rarr;</small></span></a>
          </div>
        )}
        {live && map && (
          <div className="replay" aria-hidden="true">
            {tally ? <><b>{tally.year}</b><span>{tally.all} main shocks</span><span className="both">{tally.both} reached both</span></>
              : <><b>1965&ndash;2016</b><span>{map.quakes.length} main shocks</span><span className="both">{shared} reached both</span></>}
          </div>
        )}
      </div>
      <figcaption>
        <span><i className="d solo" />Earthquake main shocks{map ? <>, <b>{map.quakes.length}</b></> : null}</span>
        <span><i className="d both" />Reached both countries{shared !== undefined ? <>, <b>{shared}</b></> : null}</span>
        <span><i className="d zone" />Shared border zone</span>
        <span className="src">Real elevation, Tian Shan and Pamir</span>
      </figcaption>
    </figure>
  )
}

function Hero() {
  return (
    <section className="hero" id="top">
      <div className="hero-in">
        <h1>
          <span className="h-sans">When disaster cover is pooled,</span>
          <em className="h-serif">the risk is everyone losing at once.</em>
        </h1>
        <p className="lede">ADB&rsquo;s US$160m Disaster Relief Bond covers the Kyrgyz Republic and Tajikistan. We compute exactly how often both countries lose their cover together, and what that means for a regional pool.</p>
        <a className="glow-cta" href={SOLUTION}><span>Explore the solution</span><b aria-hidden="true"><Arrow /></b></a>
      </div>
      <HeroFigure />
    </section>
  )
}

/* ---------- problem ---------- */

function useOneIn() {
  const mc = useJSON<{ exact_joint_exhaustion: number }>('mc_sample.json')
  return mc ? Math.round(1 / mc.exact_joint_exhaustion / 100) * 100 : null
}

/* every repeat run of plain Monte Carlo as one dot (mc_sample.json): hollow = it saw no joint loss at all */
function RunsGrid() {
  const mc = useJSON<{ repeats: number; zero_runs: number; n: number }>('mc_sample.json')
  if (!mc) return null
  const cols = 40, cell = 7 // ten rows of forty: short enough to sit inside the card
  const rows = Math.ceil(mc.repeats / cols)
  return (
    <figure className="runs">
      <svg viewBox={`0 0 ${cols * cell} ${rows * cell}`} role="img" aria-label={`${mc.repeats} repeat runs of plain Monte Carlo at ${mc.n.toLocaleString('en-US')} paths; ${mc.zero_runs} of them saw no joint loss.`}>
        {Array.from({ length: mc.repeats }, (_, i) => {
          const empty = i < mc.zero_runs
          return <circle key={i} cx={(i % cols) * cell + cell / 2} cy={Math.floor(i / cols) * cell + cell / 2} r={cell * 0.32} className={empty ? 'e' : 'f'} />
        })}
      </svg>
      <figcaption><b>{mc.zero_runs}</b> of {mc.repeats} repeat runs ({(100 * mc.zero_runs / mc.repeats).toFixed(1)}%) saw no case of both bonds exhausted (hollow); one dot per run, grouped by result</figcaption>
    </figure>
  )
}

function Problem() {
  const oneIn = useOneIn()
  return (
    <section className="sec light" id="problem">
      <div className="wrap">
        <Reveal className="sec-head">
          <Eyebrow>Problem</Eyebrow>
          <h2 className="h2"><span className="sans">Joint losses are</span> <em>too rare to see</em></h2>
        </Reveal>
        <div className="prob">
          <Reveal className="prob-fig">
            <img src="problem-terrain.webp" alt="The same terrain block seen from higher up, with the shared border zone in terracotta." width={2400} height={Math.round(2400 / PROBLEM_ASPECT)} loading="lazy" decoding="async" />
          </Reveal>
          <Reveal className="stat s1" delay={0}>
            <b className="num">{oneIn ? <>1 in {oneIn.toLocaleString('en-US')}</> : <span className="skel" />}</b>
            <p>chance that both countries&rsquo; bonds are wiped out in one term, in the brief&rsquo;s base case.</p>
          </Reveal>
          <Reveal className="stat s2" delay={90}>
            <b className="num">~<V k="MC_ZERO" /></b>
            <p>of plain Monte Carlo runs at 16,384 paths never see both bonds exhausted, so the event goes unseen.</p>
            <RunsGrid />
          </Reveal>
          <Reveal className="stat s3" delay={180}>
            <b className="num">0&ndash;<V k="FIN_EXC_MAX" /></b>
            <p>range of the pool&rsquo;s tail, the chance it pays more than US$80m, across every dependence structure consistent with each country&rsquo;s own risk.</p>
          </Reveal>
        </div>
        <Reveal className="prob-foot">
          <p>A regional pool is only as safe as its joint tail. Plain sampling cannot resolve it, and the answer depends on how the countries&rsquo; hazards overlap. So we compute it exactly, then teach a sampler to match it.</p>
        </Reveal>
      </div>
    </section>
  )
}

/* ---------- how it works ---------- */

function How() {
  const s = useStory()
  const steps = [
    { n: '01', t: 'Source model', d: 'Five causes of loss per year, from public earthquake, flood and rain records, feed one two-country bond.' },
    { n: '02', t: <>Exact <Term name="tensor network">Adds up every combination of events exactly, by joining small tables one country at a time instead of listing every outcome.</Term></>, d: <>Exact joint answers for pools of up to {s ? s.scaling.ring[s.scaling.ring.length - 1][0] : 28} countries, <V k="TN_RING28_S" /> on one core.</> },
    { n: '03', t: <>Learned sampler (<Term name="RDPO">A sampler trained to send more paths into the rare joint loss, so fewer paths reach the same accuracy.</Term>)</>, d: <>A sampler trained on the rare event; it beats the best of four standard methods in <V k="EQT_A_PATH_ALL" /> test cases at equal paths, and in <V k="EQ0_TIME" /> of <V k="EQ0_N" /> at equal CPU time.</> },
    { n: '04', t: 'Quantum stage', d: 'Priced, not assumed: the hardware speed-up a quantum solver needs before it pays off.' },
  ]
  return (
    <section className="sec dark" id="how">
      <div className="wrap">
        <Reveal className="sec-head center">
          <Eyebrow dark>How it works</Eyebrow>
          <h2 className="h2"><span className="sans">Exact where possible,</span> <em>learned where not</em></h2>
        </Reveal>
        <div className="how">
          <div className="stack">
            <Card title="Source chain" sub="Five causes of loss, per year" className="c1">
              {s ? <Sources s={s} /> : <div className="bskel" />}
            </Card>
            <Card title="Exact solve time" sub="One CPU core" className="c2">
              {s ? (
                <>
                  <div className="exactrow">
                    <Ring n={s.scaling.ring[s.scaling.ring.length - 1][0]} />
                    <div className="bigstat"><b><V k="TN_RING28_S" /></b><span>28 countries, tensor width <V k="TN_RING28_BOND" /></span></div>
                  </div>
                  <TimeLine data={s.scaling.ring} />
                </>
              ) : <div className="bskel" />}
            </Card>
            <Card title="Learned sampler" sub="Accuracy against the best of four standard methods, same number of paths" className="c3">
              <p className="kpi"><b><V k="EQT_A_PATH_ALL" /></b><span>test cases won (5 scenarios &times; 5 questions)</span></p>
              <p className="kfoot">At equal CPU time: <b><V k="EQ0_TIME" /> of <V k="EQ0_N" /></b>. Largest gain on &ldquo;both bonds exhausted&rdquo;: <b><V k="E1_EXH_RANGE_SHORT" /></b>. <a href="#results">All 25 cases</a></p>
            </Card>
          </div>
          <ol className="steps">
            {steps.map((x, i) => (
              <Reveal key={x.n} delay={i * 80}>
                <li><span className="sn">{x.n}</span><div><h3>{x.t}</h3><p>{x.d}</p></div></li>
              </Reveal>
            ))}
            <Reveal delay={360}><a className="btn light" href={SOLUTION}>See it run <Arrow /></a></Reveal>
          </ol>
        </div>
        <Reveal><NestBlock /></Reveal>
      </div>
    </section>
  )
}

/* ---------- results ---------- */

function PoolLimit() {
  const v = useValues()
  const a = parseFloat(v.FIN_POOL_BRIEF), b = parseFloat(v.FIN_POOL_MED)
  const lo = parseFloat(v.FIN_POOL_LO), hi = parseFloat(v.FIN_POOL_HI)
  return (
    <div className="pool">
      <div className="prow">
        <div className="pbar"><i style={{ width: `${a || 0}%` }} /></div>
        <div className="pnum"><b><V k="FIN_POOL_BRIEF" /></b><span>under the brief&rsquo;s dependence</span></div>
      </div>
      <div className="prow">
        <div className="pbar data"><i style={{ width: `${b || 0}%` }} />{lo && hi ? <u style={{ left: `${lo}%`, width: `${hi - lo}%` }} /> : null}</div>
        <div className="pnum"><b><V k="FIN_POOL_MED" /></b><span>posterior median under the public records, 95% <V k="FIN_POOL_LO" /> to <V k="FIN_POOL_HI" /></span></div>
      </div>
    </div>
  )
}

function Results() {
  const s = useStory()
  const v = useValues()
  return (
    <section className="sec light" id="results">
      <div className="wrap wide">
        <Reveal className="sec-head">
          <Eyebrow>Results</Eyebrow>
          <h2 className="h2"><span className="sans">What the evidence</span> <em>actually shows</em></h2>
        </Reveal>
        <div className="res">
          <Card title="Per-query wins" sub="Accuracy gain at equal paths, by question and scenario." className="r1">
            <p className="kpi"><b><V k="EQT_A_PATH_ALL" /></b><span>test cases won against the best of four methods (5 scenarios &times; 5 questions)</span></p>
            {s ? <Heat s={s} /> : <div className="bskel" />}
            <p className="kfoot">On &ldquo;both bonds exhausted&rdquo; the gain is <b><V k="E1_EXH_RANGE_SHORT" /></b>.</p>
          </Card>
          <Card title="Equal-CPU honesty" sub="Test cases won with the same number of paths, and with the same CPU time." className="r2">
            <p className="kpi"><b>{v.EQ0_TIME} of {v.EQ0_N}</b><span>benchmark test cases still won at equal CPU time</span></p>
            <Checks />
          </Card>
          <Card title="A bar for quantum" sub="Hardware speed-up needed over the projected fault-tolerant baseline to break even." className="r3">
            {s ? <Bar s={s} /> : <div className="bskel" />}
          </Card>
          <Card title="One shared limit" sub="Share of expected payouts a single US$80m pooled limit keeps, against two separate covers." className="r4">
            <p className="kpi"><b><V k="FIN_POOL_BRIEF" /> <em>to</em> <V k="FIN_POOL_MED" /></b><span>from the brief&rsquo;s dependence to the posterior median under the public records</span></p>
            <PoolLimit />
          </Card>
        </div>
      </div>
    </section>
  )
}

/* ---------- team ---------- */

const PEOPLE = [
  {
    name: 'Tan Chun Loong', photo: '/team/tan-chun-loong.webp', role: 'Physics and Quantum Technology, NTU Singapore',
    lines: [
      'Incoming Phortify Student Research Fellow at ICFO Barcelona, in Prof. Antonio Acín’s Quantum Information Theory group.',
      'URECA Research Fellow with Prof. Mile Gu on relative entropy and belief revision in model-based reinforcement learning, the mathematics of RDPO.',
      'Quantum security intern, Fraunhofer Singapore. MIT iQuHACK 2026; SEA Quantathon 2025 finalist; Track Lead for Wildfire & Emergency Response, SEA Quantum Leaders Summit 2026.',
    ],
    does: ['Method', 'Exact engine', 'RDPO', 'Quantum resource model'],
  },
  {
    name: 'Xu Zi Yu', photo: '/team/xu-zi-yu.webp', role: 'Mechanical Engineering, Universiti Malaya',
    lines: [
      'SEA Quantathon 2026 Top 8: he benchmarked a first quantum design against a classical one, found it underperformed, and rebuilt the model around a classical core, the same evidence-first habit as this proposal.',
      'ADIA Lab Structural Break Challenge, peak #37 of 1,220.',
      'AI engineer building agent and RAG pipelines; Chief AI Officer of BunkerGuard AI, winner of SuperAI NEXT Hackathon 2026.',
    ],
    does: ['Small world and saturation modules', 'Quantum resource reductions', 'Data uncertainty'],
  },
]

function Team() {
  return (
    <section className="sec light tight" id="team">
      <div className="wrap">
        <Reveal className="sec-head">
          <Eyebrow>Team</Eyebrow>
          <h2 className="h2"><span className="sans">Two builders,</span> <em>one reproducible codebase</em></h2>
        </Reveal>
        <div className="people">
          {PEOPLE.map((p, i) => (
            <Reveal key={p.name} className="person" delay={i * 90}>
              <div className="pid">
                <img className="avatar" src={p.photo} alt={p.name} width={480} height={600} loading="lazy" decoding="async" />
                <div><h3>{p.name}</h3><div className="role">{p.role}</div></div>
              </div>
              {p.lines.map((l) => <p key={l}>{l}</p>)}
              <ul className="tags" aria-label="Contributions">{p.does.map((d) => <li key={d}>{d}</li>)}</ul>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  )
}

/* ---------- documents ---------- */

function Documents() {
  const v = useValues()
  const story = useStory()
  const docs: { key: 'proposal' | 'annex' | 'deck'; title: string; sub: string }[] = [
    { key: 'proposal', title: 'Main proposal', sub: 'The short version for judges' },
    { key: 'annex', title: 'Technical annex', sub: 'Derivations and experiments' },
    { key: 'deck', title: 'Presentation', sub: 'The slide deck' },
  ]
  return (
    <section className="sec dark tight" id="documents">
      <div className="wrap">
        <Reveal className="sec-head center">
          <Eyebrow dark>Documents</Eyebrow>
          <h2 className="h2"><span className="sans">Read it, run it,</span> <em>check it</em></h2>
        </Reveal>
        <div className="docs">
          {docs.map((d, i) => {
            const info = story?.documents[d.key]
            return (
              <Reveal key={d.key} className="docwrap" delay={i * 70}>
                <a className="doc" href={PDFS[d.key]}>
                  <span className={`thumb ${d.key === 'deck' ? 'wide' : ''}`}>{info && <img src={info.thumb} alt="" width={260} height={337} loading="lazy" />}</span>
                  <span className="dmeta"><b>{d.title}</b><span>{d.sub}</span><em>{info ? `${info.pages} pages · PDF` : 'PDF'}</em></span>
                </a>
              </Reveal>
            )
          })}
          <Reveal className="docwrap" delay={210}>
            <a className="doc repo" href={REPO}>
              <span className="gh" aria-hidden="true">
                <code>CL0908/<wbr />adb-quantum-challenge</code>
                <span className="ghstat"><b>{v.N_TESTS}</b> tests</span>
              </span>
              <span className="dmeta"><b>GitHub repository</b><span>Code, data and results</span><em>Open source</em></span>
            </a>
          </Reveal>
        </div>
      </div>
    </section>
  )
}

function Footer() {
  const dem = useJSON<{ source: string; licence: string }>('dem.json')
  return (
    <footer className="footer">
      <div className="wrap">
        <div className="frow">
          <b className="fbrand">Joint-Tail</b>
          <nav aria-label="Footer">
            <a href={SOLUTION}>Solution</a>
            <a href={DASHBOARD}>Dashboard</a>
            <a href={PDFS.proposal}>Proposal</a>
            <a href={PDFS.annex}>Technical annex</a>
            <a href={PDFS.deck}>Presentation</a>
            <a href={REPO}>GitHub</a>
          </nav>
        </div>
        <p>Tan Chun Loong · Xu Zi Yu · ADB Quantum Innovation Challenge, Phase 1. Independent entry; not affiliated with or endorsed by the organisations whose public data we use.</p>
        <p><b>AI use:</b> developed with Claude Code (Anthropic) as an AI pair-programmer and writing assistant, following the authors&rsquo; research plan. All numbers on this page are produced by the repository&rsquo;s scripts (<code>src/viz/export_web_data.py</code>, <code>src/viz/web_precompute.py</code>).</p>
        <p><b>Data:</b> earthquakes, USGS catalogue; loss records, NOAA; rainfall, GHCN-Daily stations; boundaries, public GeoJSON. <b>Terrain:</b>{' '}
          {dem ? <>{dem.source}. {dem.licence}. Surface imagery: NASA Blue Marble Next Generation (NASA Earth Observatory), public domain. Rock texture: Poly Haven, CC0.</> : 'elevation data unavailable; the page shows a still image instead.'}</p>
      </div>
    </footer>
  )
}

/** A link such as /#documents from another page lands before the sections above have filled in, so the browser's own
 *  jump misses. Land again once they settle; stop as soon as the reader scrolls. */
function useHashLanding() {
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1))
    if (!id) return
    const go = () => document.getElementById(id)?.scrollIntoView()
    // re-land whenever the page above the target grows, for up to 4 s
    const ro = new ResizeObserver(go)
    ro.observe(document.body)
    const timers = [80, 4000].map((ms, i) => window.setTimeout(i ? () => ro.disconnect() : go, ms))
    const stop = () => { timers.forEach((t) => window.clearTimeout(t)); ro.disconnect() }
    const opts = { once: true, passive: true } as const
    window.addEventListener('wheel', stop, opts)
    window.addEventListener('touchstart', stop, opts)
    window.addEventListener('keydown', stop, opts)
    return () => {
      stop()
      window.removeEventListener('wheel', stop)
      window.removeEventListener('touchstart', stop)
      window.removeEventListener('keydown', stop)
    }
  }, [])
}

export default function Home() {
  useHashLanding()
  return (
    <ValuesProvider>
      <SiteNav theme="light" current="home" />
      <main>
        <Hero />
        <Problem />
        <How />
        <Results />
        <Team />
        <Documents />
      </main>
      <Footer />
    </ValuesProvider>
  )
}
