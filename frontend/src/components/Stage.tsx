import { Fragment, useMemo, useState } from 'react'
import { useJSON, useValues } from '../lib/data'
import { answers, oneIn, pct, type Grid } from '../lib/model'
import Reveal from './Reveal'

interface MapData {
  bbox: number[]
  countries: Record<string, number[][][]>
  zone: number[][][]
  quakes: number[][]
}

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
          <rect width="7" height="7" fill="#B5472E" fillOpacity={zoneOpacity} />
          <line x1="0" y1="0" x2="0" y2="7" stroke="#B5472E" strokeOpacity="0.55" strokeWidth="1.6" />
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

export default function Stage() {
  const grid = useJSON<Grid>('grid.json')
  const map = useJSON<MapData>('map.json')
  const v = useValues()
  const [eq, setEq] = useState<number | null>(null)
  const [rain, setRain] = useState<number | null>(null)
  const ready = grid && map
  const cur = { eq: eq ?? grid?.brief.eq ?? 0.0745, rain: rain ?? grid?.brief.rain ?? 0.25 }
  const a = useMemo(() => (grid ? answers(grid, cur.eq, cur.rain) : null), [grid, cur.eq, cur.rain])
  const b = grid ? { jointExhaustion: grid.brief.joint_exhaustion, exceed: grid.brief.exceed, delivered: grid.brief.delivered } : null
  const at = (p: { eq: number; rain: number } | undefined) => !!p && Math.abs(cur.eq - p.eq) < 1e-9 && Math.abs(cur.rain - p.rain) < 1e-9
  const mult = (x: number, y: number) => (Math.abs(x / y - 1) < 0.005 ? 'same as brief' : `×${(x / y).toFixed(x / y >= 10 ? 0 : 1)} the brief’s`)
  const eqDisplay = at(grid?.brief) || (grid && Math.abs(cur.eq - grid.brief.eq) < 1e-9) ? v.BRIEF_EQ_SHARE ?? '7.5%'
    : grid && Math.abs(cur.eq - grid.data.eq) < 1e-9 ? v.DATA_EQ_SHARE ?? '48%' : pct(cur.eq)
  const rainDisplay = grid && Math.abs(cur.rain - grid.brief.rain) < 1e-9 ? v.BRIEF_RAIN_SHARE ?? '25%'
    : grid && Math.abs(cur.rain - grid.data.rain) < 1e-9 ? v.DATA_RAIN_SHARE ?? '2.2%' : pct(cur.rain)

  return (
    <section className="sec sec-blue" id="explore">
      <div className="wrap">
        <Reveal>
          <p className="eyebrow">Explore</p>
          <h2 className="h2">Drag the dependence. Watch the pool’s risk move.</h2>
          <p className="lede">
            Two things decide whether the two countries’ covers fail together: how often one earthquake hits both, and how often extreme rain does. The brief assumes one answer; public records suggest another. The answers are exact, not simulated.
          </p>
        </Reveal>
        <Reveal className="stage" delay={80}>
          <div className="stage-map">
            {ready ? <MapView map={map} eq={cur.eq} /> : <div className="map-loading" role="status">Loading map…</div>}
            {a && <p className="story" aria-live="polite">
              In <b>{oneIn(a.exceed).replace('1 in ', '1 of every ')}</b> three-year terms the pool would need more than US$80m; in <b>{oneIn(a.jointExhaustion).replace('1 in ', '1 of every ')}</b> both countries’ bonds are wiped out together.
            </p>}
            <ul className="legend">
              <li><i className="lg quake" /> earthquake main shock</li>
              <li><i className="lg quake shared" /> reached both countries</li>
              <li><i className="lg zone" /> shared border zone (within 50 km of both)</li>
            </ul>
          </div>
          <aside className="card" aria-label="Controls and exact answers">
            <div className="presets" role="group" aria-label="Presets">
              <button type="button" className={at(grid?.brief) ? 'on' : ''} disabled={!grid} onClick={() => grid && (setEq(grid.brief.eq), setRain(grid.brief.rain))}>The brief’s assumptions</button>
              <button type="button" className={at(grid?.data) ? 'on' : ''} disabled={!grid} onClick={() => grid && (setEq(grid.data.eq), setRain(grid.data.rain))}>What public records show</button>
            </div>
            {grid && a && b ? (
              <>
                <Slider id="eq" label="How often does one earthquake hit both countries?" min={0} max={0.6} step={0.001} value={cur.eq}
                  onChange={setEq} display={eqDisplay} snap={0.006} hint="Share of one country's earthquakes that also hit the other, 0 to 60 percent."
                  markers={[{ at: grid.brief.eq, label: `Brief’s assumption ${v.BRIEF_EQ_SHARE ?? ''}`, sub: 'taken as given' },
                            { at: grid.data.eq, label: `Loss records ~${v.DATA_EQ_SHARE ?? ''}`, sub: `range ${(v.DATA_EQ_CI ?? '').replace('%–', '–')}` }]} />
                <Slider id="rain" label="How often does extreme rain hit both?" min={0} max={0.3} step={0.001} value={cur.rain}
                  onChange={setRain} display={rainDisplay} snap={0.004} hint="Share of extreme-rain events that hit both countries, 0 to 30 percent."
                  markers={[{ at: grid.data.rain, label: `Station data ~${(v.DATA_RAIN_SHARE ? Math.round(parseFloat(v.DATA_RAIN_SHARE)) + '%' : '')}`, sub: `range ${(v.DATA_RAIN_CI ?? '').replace('%–', '–')}` },
                            { at: grid.brief.rain, label: `Brief’s ${v.BRIEF_RAIN_SHARE ?? ''}`, sub: 'taken as given' }]} />
                <div className="readouts">
                  <Readout tone="terra" label="Chance both bonds are wiped out in one 3-year term" big={oneIn(a.jointExhaustion)}
                    small={`${pct(a.jointExhaustion, 4)} · ${mult(a.jointExhaustion, b.jointExhaustion)}`} />
                  <Readout tone="navy" label="Chance the pool pays more than US$80m" big={pct(a.exceed)}
                    small={`${oneIn(a.exceed)} · ${mult(a.exceed, b.exceed)}`} />
                  <Readout tone="teal" label="A shared US$80m pool still pays" big={pct(a.delivered)}
                    small="of expected claims, on half the capital of two separate covers" />
                </div>
              </>
            ) : (
              <p className="muted" role="status">Loading exact values…</p>
            )}
          </aside>
        </Reveal>
        <Reveal>
          <p className="fineprint">
            Exact two-country model, solved for {grid ? grid.eq_axis.length * grid.rain_axis.length : '…'} combinations of the two sliders and interpolated between them. Dependence evidence uses public proxies (loss years, station rain), not the bond’s triggers: a stress range, not a re-pricing.
          </p>
        </Reveal>
      </div>
    </section>
  )
}
