import { useJSON, useStory, useValues } from '../lib/data'
import Reveal from './Reveal'

export const PDFS = {
  proposal: 'RDPO_Proposal_Short.pdf',
  annex: 'RDPO_Proposal.pdf',
  deck: 'RDPO_Presentation.pdf',
}
export const REPO = 'https://github.com/CL0908/adb-quantum-challenge'

const PEOPLE = [
  {
    name: 'Tan Chun Loong', initials: 'TCL', role: 'Physics and Quantum Technology, NTU Singapore',
    lines: [
      'Incoming Phortify Student Research Fellow at ICFO Barcelona, in Prof. Antonio Acín’s Quantum Information Theory group.',
      'URECA Research Fellow with Prof. Mile Gu on relative entropy and belief revision in model-based reinforcement learning, the mathematics of RDPO.',
      'Quantum security intern, Fraunhofer Singapore. MIT iQuHACK 2026; SEA Quantathon 2025 finalist; Track Lead for Wildfire & Emergency Response, SEA Quantum Leaders Summit 2026.',
    ],
    does: ['Method', 'Exact engine', 'RDPO', 'Quantum resource model'],
  },
  {
    name: 'Xu Zi Yu', initials: 'XZY', role: 'Mechanical Engineering, Universiti Malaya',
    lines: [
      'SEA Quantathon 2026 Top 8: he benchmarked a first quantum design against a classical one, found it underperformed, and rebuilt the model around a classical core, the same evidence-first habit as this proposal.',
      'ADIA Lab Structural Break Challenge, peak #37 of 1,220.',
      'AI engineer building agent and RAG pipelines; Chief AI Officer of BunkerGuard AI, winner of SuperAI NEXT Hackathon 2026.',
    ],
    does: ['Quantum resource reductions', 'Data uncertainty', 'Model-risk layers'],
  },
]

export function Team() {
  return (
    <section className="sec" id="team">
      <div className="wrap wide">
        <Reveal>
          <h2 className="h2">Quantum information, reinforcement learning and applied AI, in one reproducible codebase.</h2>
        </Reveal>
        <div className="panel">
          <div className="people">
            {PEOPLE.map((p, i) => (
              <Reveal key={p.name} className="person" delay={i * 90}>
                <div className="pid">
                  <div className="avatar" aria-hidden="true">{p.initials}</div>
                  <div><h3>{p.name}</h3><div className="role">{p.role}</div></div>
                </div>
                {p.lines.map((l) => <p key={l}>{l}</p>)}
                <ul className="tags" aria-label="Contributions">{p.does.map((d) => <li key={d}>{d}</li>)}</ul>
              </Reveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

export function Documents() {
  const v = useValues()
  const story = useStory()
  const docs: { key: 'proposal' | 'annex' | 'deck'; title: string; sub: string }[] = [
    { key: 'proposal', title: 'Main proposal', sub: 'The short version for judges' },
    { key: 'annex', title: 'Technical annex', sub: 'Derivations and experiments' },
    { key: 'deck', title: 'Presentation', sub: 'The slide deck' },
  ]
  return (
    <section className="sec" id="documents">
      <div className="wrap wide">
        <Reveal>
          <h2 className="h2">Read it, run it, check it.</h2>
        </Reveal>
        <div className="panel">
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
