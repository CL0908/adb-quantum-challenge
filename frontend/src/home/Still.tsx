import { useEffect, useState } from 'react'
import Hero3D, { type Pose } from './Hero3D'
import { HERO_ASPECT, PROBLEM_ASPECT } from './Hero3D.config'
import type { MapData } from './lib/map'

/** Build-time helper: renders the hero scene with a transparent background and no motion, so a PNG can be captured
 *  (frontend/scripts/render-stills.mjs) as the fallback for browsers without WebGL. Opened as /?still=hero or /?still=problem. */
const POSES: Record<string, Pose> = { hero: { az: 0, pol: 1.12 }, problem: { az: -0.5, pol: 0.86 } }
export default function Still({ name }: { name: string }) {
  const [map, setMap] = useState<MapData | null>(null)
  const [done, setDone] = useState(false)
  useEffect(() => { fetch('data/map.json').then((r) => r.json()).then(setMap) }, [])
  return (
    <div id="still" data-done={done ? '1' : '0'} style={{ position: 'relative', width: 2400, height: Math.round(2400 / (name === 'problem' ? PROBLEM_ASPECT : HERO_ASPECT)), background: 'transparent' }}>
      {map && <Hero3D map={map} still={POSES[name] ?? POSES.hero} onReady={() => setTimeout(() => setDone(true), 600)} />}
    </div>
  )
}
