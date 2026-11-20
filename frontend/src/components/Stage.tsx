import { Component, Fragment, Suspense, lazy, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useJSON, useValues } from '../lib/data'
import { answers, oneIn, pct, type Grid } from '../lib/model'
import { loadDemInfo, type DemInfo } from '../lib/dem'
import type { MapData } from '../lib/map'

const W = 1000
const COS = Math.cos((40 * Math.PI) / 180)

function useProject(bbox: number[]) {
  return useMemo(() => {
    const [x0, y0, x1, y1] = bbox
    const sx = W / ((x1 - x0) * COS)
    const h = (y1 - y0) * sx
    const px = (lon: number) => (lon - x0) * COS * sx
    const py = (lat: number) => (y1 - lat) * sx
    return { h, px, py }
  }, [bbox])
}

const ring = (r: number[][], px: (v: number) => number, py: (v: number) => number) =>
  'M' + r.map(([lo, la]) => `${px(lo).toFixed(1)},${py(la).toFixed(1)}`).join('L') + 'Z'

function MapView({ map, eq }: { map: MapData; eq: number }) {
  const { h, px, py } = useProject(map.bbox)
  const zoneD = map.zone.map((r) => ring(r, px, py)).join('')
  const shared = map.quakes.filter((q) => q[4] === 1).length
  // the border zone darkens as the assumed shared share rises
  const zoneOpacity = 0.16 + 0.5 * Math.min(eq / 0.6, 1)
  return (
    <svg className="map-svg" viewBox={`0 0 ${W} ${h.toFixed(0)}`} role="img"
      aria-label={`Map of the Kyrgyz Republic and Tajikistan with ${map.quakes.length} earthquake main shocks, ${shared} of which reached both countries, and the shared border zone highlighted.`}>
      <defs>
        <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="7" height="7" fill="#C8553A" fillOpacity={zoneOpacity} />
          <line x1="0" y1="0" x2="0" y2="7" stroke="#C8553A" strokeOpacity="0.55" strokeWidth="1.6" />
        </pattern>
      </defs>
      {(['TJ', 'KG'] as const).map((c) => (
        <path key={c} className={`land land-${c}`} d={map.countries[c].map((r) => ring(r, px, py)).join('')} />
      ))}
      <path className="zone" d={zoneD} fill="url(#hatch)" fillRule="evenodd" />
      <path className="zone-line" d={zoneD} fillRule="evenodd" />
      {map.quakes.map((q, i) => (
        <circle key={i} cx={px(q[1])} cy={py(q[0])} r={2.2 + (q[2] - 5.5) * 3.2} className={q[4] ? 'quake shared' : 'quake'} />
      ))}
      <text className="country-name" x={px(74.6)} y={py(41.35)}>Kyrgyz Republic</text>
      <text className="country-name" x={px(70.4)} y={py(38.55)}>Tajikistan</text>
      <text className="zone-label" x={px(72.0)} y={py(40.35)}>shared border zone</text>
      <text className="map-credit" x={W - 8} y={h - 8} textAnchor="end">Earthquake main shocks M5.5+, 1965–2016 (USGS)</text>
    </svg>
  )
}

interface Marker { pos: number; label: string; sub: string }

function Slider({ id, label, min, max, step, value, onChange, display, markers, snap, hint }: {
  id: string; label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void
  display: string; markers: { at: number; label: string; sub: string }[]; snap: number; hint: string
}) {
  const ms: (Marker & { at: number })[] = markers.map((m) => ({ ...m, pos: (100 * (m.at - min)) / (max - min) }))
  return (
    <div className="slider">
      <div className="slider-head">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>{display}</output>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} aria-describedby={`${id}-hint`}
        onChange={(e) => {
          let v = +e.target.value
          for (const m of markers) if (Math.abs(v - m.at) <= snap) v = m.at
          onChange(v)
        }} />
      <div className="markers">
        {ms.map((m) => {
          const on = Math.abs(value - m.at) < 1e-9
          return (
            <Fragment key={m.label}>
              <i className={`tick ${on ? 'on' : ''}`} style={{ left: `${m.pos}%` }} aria-hidden="true" />
              <button type="button" className={`marker ${m.pos < 50 ? 'l' : 'r'} ${on ? 'on' : ''}`} onClick={() => onChange(m.at)}>
                <b>{m.label}</b>
                <span>{m.sub}</span>
              </button>
            </Fragment>
          )
        })}
      </div>
      <p id={`${id}-hint`} className="sr-only">{hint}</p>
    </div>
  )
}

function Readout({ label, big, small, tone }: { label: string; big: string; small: string; tone: string }) {
  return (
    <div className={`readout ${tone}`}>
      <div className="readout-label">{label}</div>
      <div className="readout-big" aria-live="polite">{big}</div>
      <div className="readout-small">{small}</div>
    </div>
  )
}

const Terrain3D = lazy(() => import('./Terrain3D'))

class ErrorCatch extends Component<{ onError: () => void; children: ReactNode }, { bad: boolean }> {
  state = { bad: false }
  static getDerivedStateFromError() { return { bad: true } }
  componentDidCatch() { this.props.onError() }
  render() { return this.state.bad ? null : this.props.children }
}

/** The 3D terrain model; the flat map when WebGL is missing, the elevation file is unavailable or the 3D chunk fails. Never fake terrain. */
function Model({ map, eq, children }: { map: MapData | null; eq: number; children: ReactNode }) {
  const [mode, setMode] = useState<'wait' | '3d' | '2d'>('wait')
  const [dem, setDem] = useState<DemInfo | null>(null)
  useEffect(() => {
    let live = true
    loadDemInfo().then((info) => live && (setDem(info), setMode('3d'))).catch(() => live && setMode('2d'))
    return () => { live = false }
  }, [])
  return (
    <figure className="model">
      <div className="model-stage">
        {!map || mode === 'wait' ? <div className="map-loading" role="status">Loading terrain</div>
          : mode === '2d' ? <MapView map={map} eq={eq} />
          : (
            <Suspense fallback={<div className="map-loading" role="status">Loading terrain</div>}>
              <ErrorCatch onError={() => setMode('2d')}><Terrain3D map={map} eq={eq} onFail={() => setMode('2d')} /></ErrorCatch>
            </Suspense>
          )}
        <ul className="legend" aria-label="Legend">
          <li><i className="lg quake" /> earthquake main shock</li>
          <li><i className="lg quake shared" /> reached both countries</li>
          <li><i className="lg zone" /> shared border zone</li>
        </ul>
        {children}
        {mode === '3d' && <p className="hint"><span className="hint-fine">Drag to rotate · scroll to zoom</span><span className="hint-touch">Two fingers to rotate</span></p>}
      </div>
      <figcaption>
        <b>Figure 1.</b>{' '}
        {mode === '2d' ? 'Map of the Kyrgyz Republic and Tajikistan' : `Relief of the Kyrgyz Republic and Tajikistan (vertical scale ×9${dem ? `, ${Math.round(dem.min_m)} to ${Math.round(dem.max_m).toLocaleString('en-US')} m` : ''})`}
        {' '}with the {map ? map.quakes.filter((x) => x[1] >= 66.4 && x[1] <= 81.2 && x[0] >= 35.9 && x[0] <= 44).length : ''} earthquake main shocks of the USGS catalogue (M5.5+, 1965–2016) at their true positions; {mode === '2d' ? 'dots' : 'columns'} grow with magnitude. The shared border zone lies within 50 km of both countries.
      </figcaption>
    </figure>
  )
}

export default function Stage({ proposalHref }: { proposalHref: string }) {
  const grid = useJSON<Grid>('grid.json')
  const map = useJSON<MapData>('map.json')
  const v = useValues()
  const [eq, setEq] = useState<number | null>(null)
  const [rain, setRain] = useState<number | null>(null)
  const cur = { eq: eq ?? grid?.brief.eq ?? 0.0745, rain: rain ?? grid?.brief.rain ?? 0.25 }
  const a = useMemo(() => (grid ? answers(grid, cur.eq, cur.rain) : null), [grid, cur.eq, cur.rain])
  const b = grid ? { jointExhaustion: grid.brief.joint_exhaustion, exceed: grid.brief.exceed, delivered: grid.brief.delivered } : null
  const at = (p: { eq: number; rain: number } | undefined) => !!p && Math.abs(cur.eq - p.eq) < 1e-9 && Math.abs(cur.rain - p.rain) < 1e-9
  const mult = (x: number, y: number) => (Math.abs(x / y - 1) < 0.005 ? 'same as brief' : `×${(x / y).toFixed(x / y >= 10 ? 0 : 1)} the brief’s`)
  const eqDisplay = grid && Math.abs(cur.eq - grid.brief.eq) < 1e-9 ? v.BRIEF_EQ_SHARE
    : grid && Math.abs(cur.eq - grid.data.eq) < 1e-9 ? v.DATA_EQ_SHARE : pct(cur.eq)
  const rainDisplay = grid && Math.abs(cur.rain - grid.brief.rain) < 1e-9 ? v.BRIEF_RAIN_SHARE
    : grid && Math.abs(cur.rain - grid.data.rain) < 1e-9 ? v.DATA_RAIN_SHARE : pct(cur.rain)

  return (
    <section className="first" id="top">
      <div className="first-copy">
        <p className="eyebrow">ADB Quantum Innovation Challenge · Phase 1</p>
        <h1>When disaster cover is pooled, the risk is everyone losing at once.</h1>
        <p className="first-lede">
          ADB’s US$160m Disaster Relief Bond covers the Kyrgyz Republic and Tajikistan. The dangerous case is both countries using up their cover in the same three years, too rare to see in history or in ordinary simulation: we compute it exactly, and show how much money a regional pool needs because of it.
        </p>
        <p className="authors"><b>Tan Chun Loong</b> · <b>Xu Zi Yu</b></p>
        <p className="first-links"><a href={proposalHref}>Read the proposal (PDF)</a><a href="#problem">Why simulation fails</a></p>
        <p className="first-note">Move the two sliders: the model and the three numbers change together. The numbers are exact, not simulated.</p>
        <p className="first-fine">Exact two-country model, solved at {grid ? grid.eq_axis.length * grid.rain_axis.length : 'many'} combinations of the two sliders and interpolated between them. Dependence evidence uses public proxies (loss years, station rain), not the bond’s triggers: a stress range, not a re-pricing.</p>
      </div>
      <div className="first-model">
        <Model map={map} eq={cur.eq}>
        <aside className="card ctl" aria-label="Controls and exact answers">
          <div className="presets" role="group" aria-label="Presets">
            <button type="button" className={at(grid?.brief) ? 'on' : ''} disabled={!grid} onClick={() => grid && (setEq(grid.brief.eq), setRain(grid.brief.rain))}>The brief’s assumptions</button>
            <button type="button" className={at(grid?.data) ? 'on' : ''} disabled={!grid} onClick={() => grid && (setEq(grid.data.eq), setRain(grid.data.rain))}>What public records show</button>
          </div>
          {grid && a && b ? (
            <>
              <div className="sliders">
                <Slider id="eq" label="One earthquake hits both countries" min={0} max={0.6} step={0.001} value={cur.eq}
                  onChange={setEq} display={eqDisplay} snap={0.006} hint="Share of one country's earthquakes that also hit the other, 0 to 60 percent."
                  markers={[{ at: grid.brief.eq, label: `Brief ${v.BRIEF_EQ_SHARE ?? ''}`, sub: 'taken as given' },
                            { at: grid.data.eq, label: `Records ~${v.DATA_EQ_SHARE ?? ''}`, sub: `range ${(v.DATA_EQ_CI ?? '').replace('%–', '–')}` }]} />
                <Slider id="rain" label="Extreme rain hits both" min={0} max={0.3} step={0.001} value={cur.rain}
                  onChange={setRain} display={rainDisplay} snap={0.004} hint="Share of extreme-rain events that hit both countries, 0 to 30 percent."
                  markers={[{ at: grid.data.rain, label: `Stations ~${(v.DATA_RAIN_SHARE ? Math.round(parseFloat(v.DATA_RAIN_SHARE)) + '%' : '')}`, sub: `range ${(v.DATA_RAIN_CI ?? '').replace('%–', '–')}` },
                            { at: grid.brief.rain, label: `Brief ${v.BRIEF_RAIN_SHARE ?? ''}`, sub: 'taken as given' }]} />
              </div>
              <div className="readouts">
                <Readout tone="terra" label="Both bonds wiped out in one 3-year term" big={oneIn(a.jointExhaustion)}
                  small={`${pct(a.jointExhaustion, 4)} · ${mult(a.jointExhaustion, b.jointExhaustion)}`} />
                <Readout tone="navy" label="Pool pays more than US$80m" big={pct(a.exceed)}
                  small={`${oneIn(a.exceed)} · ${mult(a.exceed, b.exceed)}`} />
                <Readout tone="teal" label="A shared US$80m pool still pays" big={pct(a.delivered)}
                  small="of expected claims, on half the capital" />
              </div>
            </>
          ) : (
            <p className="muted" role="status">Loading exact values</p>
          )}
        </aside>
        </Model>
      </div>
    </section>
  )
}
