import { useEffect, useState } from 'react'
import { ridgePath } from '../lib/ridge'

const W = 1600
const H = 900
// far to near: [seed, base y, relief, fill, drift seconds]
const LAYERS: [number, number, number, string, number][] = [
  [11, 850, 285, 'snow', 150],
  [23, 872, 235, 'far', 120],
  [37, 888, 195, 'mid', 100],
  [53, 880, 170, 'near', 80],
  [71, 896, 120, 'front', 60],
]

export function Landscape({ id = 'hero', night = false }: { id?: string; night?: boolean }) {
  const paths = LAYERS.map(([seed, base, amp]) => ridgePath(seed, -160, W + 160, base, amp, H + 20, 0.62))
  const c = night
    ? { top: '#050d1f', mid: '#0d2347', low: '#1d3b66', glow: '#3d5a86', snow: ['#7d93b8', '#31496f'], far: ['#2a4269', '#16294a'], mid2: ['#182f55', '#0c1b36'], near: ['#0e2142', '#071328'], front: ['#071328', '#040b1a'] }
    : { top: '#17386b', mid: '#5f8ab5', low: '#f2b99a', glow: '#ffe2c4', snow: ['#f6e6e0', '#a9bdd6'], far: ['#8fa8c9', '#7d95b8'], mid2: ['#5f7aa3', '#4d6791'], near: ['#3c5683', '#2c426b'], front: ['#25395f', '#1a2a49'] }
  return (
    <svg className={`landscape ${night ? 'night' : ''}`} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMax slice" aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id={`${id}-sky`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={c.top} />
          <stop offset="0.45" stopColor={c.mid} />
          <stop offset="0.78" stopColor={c.low} />
          <stop offset="1" stopColor={c.glow} />
        </linearGradient>
        <radialGradient id={`${id}-sun`} cx="0.68" cy="0.78" r="0.5">
          <stop offset="0" stopColor="#fff3df" stopOpacity={night ? 0.25 : 0.95} />
          <stop offset="0.35" stopColor="#ffd7b0" stopOpacity={night ? 0.1 : 0.5} />
          <stop offset="1" stopColor="#ffd7b0" stopOpacity="0" />
        </radialGradient>
        <linearGradient id={`${id}-snow`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={c.snow[0]} /><stop offset="0.55" stopColor={c.snow[1]} /><stop offset="1" stopColor={c.far[0]} /></linearGradient>
        <linearGradient id={`${id}-far`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={c.far[0]} /><stop offset="1" stopColor={c.far[1]} /></linearGradient>
        <linearGradient id={`${id}-mid`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={c.mid2[0]} /><stop offset="1" stopColor={c.mid2[1]} /></linearGradient>
        <linearGradient id={`${id}-near`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={c.near[0]} /><stop offset="1" stopColor={c.near[1]} /></linearGradient>
        <linearGradient id={`${id}-front`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={c.front[0]} /><stop offset="1" stopColor={c.front[1]} /></linearGradient>
        <linearGradient id={`${id}-mist`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={night ? '#2b4672' : '#ffe9d6'} stopOpacity="0" />
          <stop offset="1" stopColor={night ? '#2b4672' : '#ffe9d6'} stopOpacity={night ? 0.35 : 0.55} />
        </linearGradient>
        <filter id={`${id}-blur`}><feGaussianBlur stdDeviation="14" /></filter>
      </defs>
      <rect width={W} height={H} fill={`url(#${id}-sky)`} />
      <rect width={W} height={H} fill={`url(#${id}-sun)`} />
      {!night && (
        <g filter={`url(#${id}-blur)`} className="clouds" opacity="0.55">
          <ellipse cx="320" cy="250" rx="260" ry="22" fill="#fff" opacity="0.5" />
          <ellipse cx="1180" cy="330" rx="330" ry="26" fill="#ffe9dc" opacity="0.55" />
          <ellipse cx="760" cy="440" rx="380" ry="20" fill="#fff6ee" opacity="0.5" />
          <ellipse cx="1420" cy="190" rx="200" ry="16" fill="#fff" opacity="0.4" />
        </g>
      )}
      {night &&
        Array.from({ length: 46 }, (_, i) => (
          <circle key={i} cx={(i * 197) % W} cy={((i * 83) % 420) + 10} r={i % 5 === 0 ? 1.6 : 0.9} fill="#fff" opacity={0.25 + (i % 4) * 0.12} />
        ))}
      {paths.map((d, i) => (
        <g key={i}>
          <g className="drift" style={{ animationDuration: `${LAYERS[i][4]}s`, animationDirection: i % 2 ? 'alternate' : 'alternate-reverse' }}>
            <path d={d} fill={`url(#${id}-${LAYERS[i][3]})`} />
          </g>
          {i < LAYERS.length - 1 && <rect x="0" y={LAYERS[i][1] - 40} width={W} height={H - LAYERS[i][1] + 40} fill={`url(#${id}-mist)`} opacity={0.5 - i * 0.07} />}
        </g>
      ))}
    </svg>
  )
}

export default function Hero({ proposalHref }: { proposalHref: string }) {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 40)
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])
  return (
    <header className="hero" id="top">
      <Landscape />
      <div className="hero-shade" />
      <nav className={`nav ${scrolled ? 'scrolled' : ''}`} aria-label="Main">
        <div className="nav-in">
          <a className="logo" href="#top">
            <span className="logo-mark" aria-hidden="true">
              <svg viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#fff" /><path d="M5 23 12 11l4 6 3-4 8 10z" fill="#1F3557" /></svg>
            </span>
            Joint-Tail
          </a>
          <div className="nav-links">
            <a href="#problem">Problem</a>
            <a href="#explore">Explore</a>
            <a href="#method">Method</a>
            <a href="#team">Team</a>
            <a href="#documents">Documents</a>
          </div>
          <a className="btn-white" href={proposalHref}>
            <span className="long">Read the proposal</span>
            <span className="short">Proposal</span>
          </a>
        </div>
      </nav>
      <div className="hero-copy">
        <h1>
          <span className="h1a">When disaster cover is pooled,</span>
          <span className="h1b">the risk is everyone losing at once.</span>
        </h1>
        <p className="hero-sub">
          ADB’s US$160m Disaster Relief Bond covers the Kyrgyz Republic and Tajikistan. The dangerous case is both countries using up their cover in the same three years: too rare to see in history or in ordinary simulation. We compute it exactly, and show how much money a regional pool needs because of it.
        </p>
        <a className="btn-white big" href="#explore">Explore the risk</a>
        <p className="hero-by">Tan Chun Loong · Xu Zi Yu — ADB Quantum Innovation Challenge, Phase 1</p>
      </div>
      <div className="strip" aria-label="Data and method sources">
        <span>USGS catalogue</span><i>·</i><span>NOAA loss records</span><i>·</i><span>GHCN-Daily stations</span><i>·</i><span>ADB challenge brief</span>
      </div>
    </header>
  )
}
