import { Fragment } from 'react'
import { useJSON, V, useValues } from '../lib/data'
import Reveal from './Reveal'

export const PDFS = {
  proposal: 'RDPO_Proposal_Short.pdf',
  annex: 'RDPO_Proposal.pdf',
  deck: 'RDPO_Presentation.pdf',
}
export const REPO = 'https://github.com/CL0908/adb-quantum-challenge'

export function Method() {
  return (
    <section className="sec" id="method">
      <div className="wrap">
        <Reveal>
          <p className="eyebrow">Method</p>
          <h2 className="h2">Exact where possible. Learned where not. Quantum where it can be shown to pay.</h2>
        </Reveal>
        <div className="cards3">
          <Reveal className="mcard navy" delay={0}>
            <div className="mtag">1 · Exact</div>
            <div className="mbig"><V k="AP_N" /> countries</div>
            <p>A tensor network gives a pool’s whole joint law with zero sampling error. The <V k="AP_N" />-country Asia–Pacific pool takes <b><V k="AP_MPS_SEC" /></b> on one CPU core.</p>
          </Reveal>
          <Reveal className="mcard terra" delay={90}>
            <div className="mtag">2 · Learned</div>
            <div className="mbig"><V k="EQT_A_PATH_BRIEF" /></div>
            <p>benchmark cells where our learned sampler (RDPO) is more accurate than the best of the brief’s four methods on the same number of simulated paths, with honest 95% intervals.</p>
          </Reveal>
          <Reveal className="mcard teal" delay={180}>
            <div className="mtag">3 · Quantum</div>
            <div className="mbig"><V k="Q_SSTAR_28" />×</div>
            <p>the speed-up quantum hardware needs over today’s projections before it beats our classical sampler at 28 countries: a bar we can be held to.</p>
          </Reveal>
        </div>
      </div>
    </section>
  )
}

interface CarecMap {
  bbox: number[]
  carec: { countries: Record<string, { name: string; rings: number[][][]; c: number[]; p25: number | null }>; edges: [string, string][] }
}

export function Pools() {
  const map = useJSON<CarecMap>('map.json')
  const X0 = 40, X1 = 90, Y0 = 28, Y1 = 55
  const cos = Math.cos((41 * Math.PI) / 180), sx = 1000 / ((X1 - X0) * cos), h = (Y1 - Y0) * sx
  const px = (lo: number) => (lo - X0) * cos * sx
  const py = (la: number) => (Y1 - la) * sx
  const cs = map?.carec.countries
  return (
    <section className="sec" id="pools">
      <div className="wrap split">
        <Reveal>
          <p className="eyebrow">Regional pools</p>
          <h2 className="h2">The same exact answer for a pool of ten countries, and of twenty-eight.</h2>
          <p className="lede">
            Pooling is only worth it if you can say how likely the pool is to be stretched. For the ten-country CAREC pool, one CPU core produces the full joint law in <b><V k="CAREC_MPS_S" /></b>, down to tails of <V k="CAREC_MU_RARE" />. The <V k="AP_N" />-country Asia–Pacific pool takes <b><V k="AP_MPS_SEC" /></b>. Lines join countries that share earthquakes in the catalogue.
          </p>
        </Reveal>
        <Reveal className="poolmap" delay={90}>
          {cs ? (
            <svg viewBox={`0 0 1000 ${h.toFixed(0)}`} role="img" aria-label="Map of the ten CAREC countries, linked where earthquakes reached both">
              {Object.entries(cs).map(([iso, c]) => (
                <path key={iso} className={`pland ${iso === 'KG' || iso === 'TJ' ? 'hot' : ''}`} d={c.rings.map((r) => 'M' + r.map(([lo, la]) => `${px(lo).toFixed(1)},${py(la).toFixed(1)}`).join('L') + 'Z').join('')} />
              ))}
              {map!.carec.edges.map(([a, b]) => cs[a] && cs[b] && (
                <line key={a + b} className="pedge" x1={px(cs[a].c[0])} y1={py(cs[a].c[1])} x2={px(cs[b].c[0])} y2={py(cs[b].c[1])} />
              ))}
              {Object.entries(cs).map(([iso, c]) => (
                <Fragment key={iso}>
                  <circle className={`pnode ${iso === 'KG' || iso === 'TJ' ? 'hot' : ''}`} cx={px(c.c[0])} cy={py(c.c[1])} r={8} />
                  <text className="plabel" x={px(c.c[0])} y={py(c.c[1]) - 14} textAnchor="middle">{c.name.replace('Kyrgyzstan', 'Kyrgyz Rep.')}</text>
                </Fragment>
              ))}
            </svg>
          ) : <div className="map-loading">Loading map…</div>}
          <p className="figcap"><b>Figure 3.</b> The ten CAREC countries; the Kyrgyz Republic and Tajikistan in terracotta. A line joins two countries that share earthquakes in the catalogue.</p>
        </Reveal>
      </div>
    </section>
  )
}

export function Honest() {
  return (
    <section className="sec slim" id="accounting">
      <div className="wrap">
        <Reveal>
          <p className="eyebrow">Honest accounting</p>
          <h2 className="h2 mid">
            On the same number of paths our sampler wins <V k="EQT_A_PATH_BRIEF" /> benchmark cells; on the same CPU time it wins <V k="EQT_A_TIME_BRIEF" />, and keeps <V k="EQT_A_TIME_JE" /> on the rarest event.
          </h2>
          <p className="lede"><a className="textlink" href={PDFS.annex}>See the equal-CPU table in the technical annex (PDF) →</a></p>
        </Reveal>
      </div>
    </section>
  )
}

export function Quantum() {
  return (
    <section className="sec" id="quantum">
      <div className="wrap">
        <Reveal className="qpanel">
          <p className="eyebrow">Quantum stage</p>
          <h2 className="h2">A bar we can be held to.</h2>
          <div className="bignum"><V k="Q_SSTAR_28" /><span>×</span></div>
          <p className="lede">
            At <V k="AP_N" /> countries, quantum amplitude estimation breaks even with our best classical sampler only on hardware this many times faster than projected. At two countries the bar is <V k="Q_SSTAR_2" />×: it falls as pools grow, and anyone can test it.
          </p>
        </Reveal>
      </div>
    </section>
  )
}

export function Team() {
  return (
    <section className="sec" id="team">
      <div className="wrap">
        <Reveal>
          <p className="eyebrow">Team</p>
          <h2 className="h2">Quantum information, reinforcement learning and applied AI, in one reproducible codebase.</h2>
        </Reveal>
        <div className="cards2">
          <Reveal className="person" delay={0}>
            <h3>Tan Chun Loong</h3>
            <div className="role">Physics and Quantum Technology, NTU Singapore</div>
            <p>Incoming Phortify Student Research Fellow at ICFO Barcelona, in Prof. Antonio Acín’s Quantum Information Theory group. URECA Research Fellow with Prof. Mile Gu on relative entropy and belief revision in model-based reinforcement learning, the mathematics of RDPO. Quantum security intern, Fraunhofer Singapore.</p>
            <p>MIT iQuHACK 2026; SEA Quantathon 2025 finalist; Track Lead for Wildfire &amp; Emergency Response, SEA Quantum Leaders Summit 2026.</p>
            <p className="does">Method, exact engine, RDPO, quantum resource model.</p>
          </Reveal>
          <Reveal className="person" delay={90}>
            <h3>Xu Zi Yu</h3>
            <div className="role">Mechanical Engineering, Universiti Malaya</div>
            <p>SEA Quantathon 2026 Top 8: he benchmarked a first quantum design against a classical one, found it underperformed, and rebuilt the model around a classical core, the same evidence-first habit as this proposal. ADIA Lab Structural Break Challenge, peak #37 of 1,220.</p>
            <p>AI engineer building agent and RAG pipelines; Chief AI Officer of BunkerGuard AI, winner of SuperAI NEXT Hackathon 2026.</p>
            <p className="does">Quantum resource reductions, data-uncertainty and model-risk layers.</p>
          </Reveal>
        </div>
      </div>
    </section>
  )
}

export function Documents() {
  const v = useValues()
  const docs: [string, string, string][] = [
    [PDFS.proposal, 'Main proposal', 'The short version for judges (PDF)'],
    [PDFS.annex, 'Technical annex', 'Full derivations and experiments (PDF)'],
    [PDFS.deck, 'Presentation', 'The slide deck (PDF)'],
    [REPO, 'GitHub repository', `Code, data, results and ${v.N_TESTS ?? ''} tests`],
  ]
  return (
    <section className="sec" id="documents">
      <div className="wrap">
        <Reveal>
          <p className="eyebrow">Documents</p>
          <h2 className="h2">Read it, run it, check it.</h2>
        </Reveal>
        <div className="cards4">
          {docs.map(([href, t, s], i) => (
            <Reveal key={href} delay={i * 70}>
              <a className="doc" href={href}><b>{t}</b><span>{s}</span><i aria-hidden="true">→</i></a>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  )
}

export function Footer() {
  const dem = useJSON<{ source: string; licence: string }>('dem.json')
  return (
    <footer className="footer">
      <div className="wrap">
        <div className="frow">
          <b className="fbrand">Joint-Tail</b>
          <nav aria-label="Footer">
            <a href={PDFS.proposal}>Proposal</a>
            <a href={PDFS.annex}>Technical annex</a>
            <a href={PDFS.deck}>Presentation</a>
            <a href={REPO}>GitHub</a>
            <a href="#try">Estimator</a>
            <a href="#team">Team</a>
          </nav>
        </div>
        <p>Tan Chun Loong · Xu Zi Yu · ADB Quantum Innovation Challenge, Phase 1. Independent entry; not affiliated with or endorsed by the organisations whose public data we use.</p>
        <p><b>AI use:</b> developed with Claude Code (Anthropic) as an AI pair-programmer and writing assistant, following the authors’ research plan. All numbers on this page are produced by the repository’s scripts (<code>src/viz/export_web_data.py</code>, <code>src/viz/web_precompute.py</code>).</p>
        <p><b>Data:</b> earthquakes, USGS catalogue; loss records, NOAA; rainfall, GHCN-Daily stations; boundaries, public GeoJSON. <b>Terrain:</b>{' '}
          {dem ? <>{dem.source}. {dem.licence}.</> : 'elevation data unavailable; the page shows the flat map instead.'}</p>
      </div>
    </footer>
  )
}
