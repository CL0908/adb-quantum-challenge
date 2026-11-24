import { useEffect, useRef } from 'react'
import {
  AdditiveBlending, AmbientLight, BufferAttribute, BufferGeometry, CanvasTexture, CylinderGeometry, DirectionalLight, DoubleSide,
  ACESFilmicToneMapping, Color, Group, HemisphereLight, PCFShadowMap, InstancedMesh, LineBasicMaterial, LineSegments, LinearFilter, Matrix4, Mesh, MeshBasicMaterial, MeshStandardMaterial,
  PerspectiveCamera, Quaternion, RepeatWrapping, TextureLoader, Vector2, Scene, SphereGeometry, SRGBColorSpace, Sprite, SpriteMaterial, Vector3, WebGLRenderer,
} from 'three'
import { loadDem } from './lib/dem'
import type { MapData } from './lib/map'
import { blur, buildField, colourCanvas, contours, fbm, fieldCanvas, hash, resample, smooth, vnoise } from './island'

const COS = Math.cos((40 * Math.PI) / 180)
const EXAG = 5.5 // vertical exaggeration
const K = EXAG / 111320 // metres -> world units (1 unit = 1 degree of latitude)

export interface Pose { az: number; pol: number }
import { HERO_ASPECT } from './Hero3D.config'
export { HERO_ASPECT }

function paintZone(map: MapData, lon0: number, lat0: number, lon1: number, lat1: number, W: number, H: number) {
  const tx = (lon: number) => ((lon - lon0) / (lon1 - lon0)) * W
  const ty = (lat: number) => ((lat1 - lat) / (lat1 - lat0)) * H
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const g = c.getContext('2d')!
  const path = () => {
    g.beginPath()
    for (const r of map.zone) {
      r.forEach(([lo, la], i) => (i ? g.lineTo(tx(lo), ty(la)) : g.moveTo(tx(lo), ty(la))))
      g.closePath()
    }
  }
  g.fillStyle = 'rgba(206,104,70,0.36)'
  g.shadowColor = 'rgba(236,130,80,0.7)'
  g.shadowBlur = W * 0.012
  path(); g.fill('evenodd')
  g.shadowBlur = 0
  g.strokeStyle = 'rgba(255,226,206,0.8)'
  g.lineWidth = Math.max(1.5, W * 0.0011)
  path(); g.stroke()
  return c
}

function halo() {
  const c = document.createElement('canvas')
  c.width = c.height = 128
  const g = c.getContext('2d')!
  const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64)
  gr.addColorStop(0, 'rgba(255,170,120,0.85)')
  gr.addColorStop(0.35, 'rgba(240,110,70,0.28)')
  gr.addColorStop(1, 'rgba(240,110,70,0)')
  g.fillStyle = gr
  g.fillRect(0, 0, 128, 128)
  const t = new CanvasTexture(c)
  t.colorSpace = SRGBColorSpace
  return t
}

/** A floating, sculpted island of real terrain: the Kyrgyz Republic and Tajikistan cut from the DEM with an eroded outline,
 *  layered rock below, thin earthquake columns on top and the shared border zone glowing. */
export type Pin = { id: string; lon: number; lat: number }
export type Tally = { year: number; all: number; both: number }
export default function Hero3D({ map, still, onReady, onFail, onYear, pins, pinEls, hold }: {
  map: MapData; still?: Pose; onReady?: () => void; onFail?: () => void
  onYear?: (t: Tally) => void; pins?: Pin[]; pinEls?: { current: Record<string, HTMLElement | null> }; hold?: { current: boolean }
}) {
  const host = useRef<HTMLDivElement>(null)
  const cb = useRef({ onReady, onFail, onYear, pins, pinEls, hold })
  cb.current = { onReady, onFail, onYear, pins, pinEls, hold }

  useEffect(() => {
    const el = host.current
    if (!el) return
    let dead = false
    let raf = 0
    const disposers: (() => void)[] = []
    ;(async () => {
      let renderer: WebGLRenderer
      try {
        renderer = new WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance', preserveDrawingBuffer: !!still })
      } catch {
        if (!dead) cb.current.onFail?.()
        return
      }
      try {
        const dem = await loadDem()
        if (dead) { renderer.dispose(); return }
        const [lon0, lat0, lon1, lat1] = dem.info.extent
        const nx = dem.info.width, ny = dem.info.height
        const sl = (lon1 - lon0) / (nx - 1), st = (lat1 - lat0) / (ny - 1)
        const wx = (lon: number) => (lon - lon0) * COS
        const wz = (lat: number) => -(lat - lat0)
        const small = window.innerWidth < 700

        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, still ? 2 : 1.75))
        renderer.setClearColor(0x000000, 0)
        renderer.toneMapping = ACESFilmicToneMapping
        renderer.toneMappingExposure = 1.08
        renderer.shadowMap.enabled = !small // ridges cast soft shadows into the valleys; off on phones to keep the frame rate
        renderer.shadowMap.type = PCFShadowMap
        renderer.domElement.className = 'hero-canvas'
        el.prepend(renderer.domElement)
        disposers.push(() => { renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove() })

        const scene = new Scene()
        const camera = new PerspectiveCamera(17, HERO_ASPECT, 0.5, 400)
        const pivot = new Group()
        const root = new Group()
        pivot.add(root)
        scene.add(pivot)

        /* ---- outline + top surface ---- */
        dem.h.set(blur(dem.h, nx, ny, 1, 2)) // soften the 2 km raster so the relief reads as sculpted, not spiky
        const F = buildField(dem, map)
        const loops = contours(F, dem)
        const areaOf = (p: [number, number][]) => Math.abs(p.reduce((s, q, i) => { const r = p[(i + 1) % p.length]; return s + q[0] * r[1] - r[0] * q[1] }, 0) / 2)
        loops.sort((a, b) => areaOf(b) - areaOf(a))
        if (!loops.length) throw new Error('no outline')
        const keep = loops.filter((l) => areaOf(l) > areaOf(loops[0]) * 0.04).slice(0, 3)

        const pos = new Float32Array(nx * ny * 3), uv = new Float32Array(nx * ny * 2)
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const k = j * nx + i
          pos[3 * k] = wx(lon0 + i * sl)
          pos[3 * k + 1] = Math.max(dem.h[k], 0) * K
          pos[3 * k + 2] = wz(lat1 - j * st)
          uv[2 * k] = i / (nx - 1)
          uv[2 * k + 1] = 1 - j / (ny - 1)
        }
        const idxs: number[] = []
        for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
          const a = j * nx + i, b = a + nx, c = a + 1, d = b + 1
          if (F.v[a] < F.T - 0.12 && F.v[b] < F.T - 0.12 && F.v[c] < F.T - 0.12 && F.v[d] < F.T - 0.12) continue
          idxs.push(a, b, c, c, b, d)
        }
        const geo = new BufferGeometry()
        geo.setAttribute('position', new BufferAttribute(pos, 3))
        geo.setAttribute('uv', new BufferAttribute(uv, 2))
        geo.setIndex(new BufferAttribute(new Uint32Array(idxs), 1))
        geo.computeVertexNormals()

        // colour at the DEM's resolution, enlarged smoothly and given a fine grain so it stays crisp on large screens
        const TW = small ? 1400 : 2048
        const TH = Math.round((TW * (lat1 - lat0)) / ((lon1 - lon0) * COS))
        // ground: NASA Blue Marble imagery on the DEM's own extent (data/ground.webp); the procedural colours stay as the fallback
        const gimg = await new Promise<HTMLImageElement | null>((ok) => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => ok(null); im.src = 'data/ground.webp' })
        const base = gimg ?? colourCanvas(dem, F)
        const big = document.createElement('canvas')
        big.width = TW
        big.height = TH
        const bg = big.getContext('2d')!
        bg.imageSmoothingEnabled = true
        bg.imageSmoothingQuality = 'high'
        bg.drawImage(base, 0, 0, TW, TH)
        const grainSize = 256
        const gc = document.createElement('canvas')
        gc.width = gc.height = grainSize
        const gg = gc.getContext('2d')!
        const gi = gg.createImageData(grainSize, grainSize)
        for (let y = 0; y < grainSize; y++) for (let x = 0; x < grainSize; x++) {
          const v = 128 + (fbm(x / 6, y / 6, 3, 0, 0, 40) - 0.5) * 120 + (vnoise(x * 0.9, y * 0.9, 0, 0, 41) - 0.5) * 70
          const k = 4 * (y * grainSize + x)
          gi.data[k] = gi.data[k + 1] = gi.data[k + 2] = v
          gi.data[k + 3] = 255
        }
        gg.putImageData(gi, 0, 0)
        bg.globalCompositeOperation = 'soft-light'
        bg.globalAlpha = gimg ? 0.16 : 0.55 // imagery has its own texture; the grain only keeps it crisp when enlarged
        bg.fillStyle = bg.createPattern(gc, 'repeat')!
        bg.fillRect(0, 0, TW, TH)
        const ground = new CanvasTexture(big)
        ground.colorSpace = SRGBColorSpace
        ground.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy())
        ground.minFilter = LinearFilter
        const alphaTex = new CanvasTexture(fieldCanvas(F))
        alphaTex.minFilter = LinearFilter
        const mat = new MeshStandardMaterial({ map: ground, alphaMap: alphaTex, alphaTest: 0.5, roughness: 0.96, metalness: 0 })
        const land = new Mesh(geo, mat)
        land.castShadow = land.receiveShadow = !small
        root.add(land)
        disposers.push(() => { geo.dispose(); mat.dispose(); ground.dispose(); alphaTex.dispose() })

        const zoneTex = new CanvasTexture(paintZone(map, lon0, lat0, lon1, lat1, TW, TH))
        zoneTex.colorSpace = SRGBColorSpace
        const zoneMat = new MeshBasicMaterial({ map: zoneTex, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, opacity: 0.9 })
        root.add(new Mesh(geo, zoneMat))
        disposers.push(() => { zoneMat.dispose(); zoneTex.dispose() })

        /* ---- the floating mountain beneath: an inverted height field on the same grid as the surface ----
           Depth grows from zero at the rim (where it meets the surface, so the island is closed) to a ragged bowl, with a
           few spires hanging lower. Same triangles and alpha cut as the surface, so the two edges coincide exactly. */
        const DEPTH = 2.5 // at the deepest point; kept modest so the surface, not the rock, carries the picture
        // fractured rock, not bedding: Poly Haven 'rock_face_03' (CC0) colour, normal and roughness maps
        const texLoader = new TextureLoader()
        // loaded before the first frame, so the still renders and the hero never shows untextured rock
        const rockTex = async (f: string, srgb: boolean) => {
          const t = await texLoader.loadAsync(`tex/${f}`)
          if (srgb) t.colorSpace = SRGBColorSpace
          t.wrapS = t.wrapT = RepeatWrapping
          t.anisotropy = 8
          t.channel = 1
          return t
        }
        const [rDiff, rNor, rRough] = await Promise.all([rockTex('rock_diff.webp', true), rockTex('rock_nor.webp', false), rockTex('rock_rough.webp', false)])
        const rockMat = new MeshStandardMaterial({ color: new Color(2.0, 1.9, 1.78), map: rDiff, normalMap: rNor, normalScale: new Vector2(1.2, 1.2), roughnessMap: rRough, roughness: 1, vertexColors: true, metalness: 0, side: DoubleSide, alphaMap: alphaTex, alphaTest: 0.5 })
        let cx = 0, cz = 0, minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9, minY = 0, maxY = 0
        const outline: Vector3[] = [] // points on the rim and the underside, for a tight camera fit
        keep.forEach((loop, li) => {
          const P = resample(loop, 360)
          const xs = P.map(([lo]) => wx(lo)), zs = P.map(([, la]) => wz(la))
          if (li === 0) { cx = xs.reduce((a, b) => a + b, 0) / xs.length; cz = zs.reduce((a, b) => a + b, 0) / zs.length }
          P.forEach(([lo, la], i) => {
            const x = xs[i], z = zs[i]
            minX = Math.min(minX, x); maxX = Math.max(maxX, x); minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z)
            const y = Math.max(dem.at(lo, la), 0) * K
            if (y > maxY) maxY = y
            if (i % 6 === 0) outline.push(new Vector3(x, y, z))
          })
        })
        // distance from the rim in grid cells (two-pass chamfer), normalised: 0 at the rim, 1 at the deepest point
        const dist = new Float32Array(nx * ny)
        for (let k = 0; k < nx * ny; k++) dist[k] = F.v[k] > F.T ? 1e9 : 0
        for (let j = 1; j < ny; j++) for (let i = 1; i < nx - 1; i++) { const k = j * nx + i; if (dist[k]) dist[k] = Math.min(dist[k], dist[k - 1] + 1, dist[k - nx] + 1, dist[k - nx - 1] + 1.414, dist[k - nx + 1] + 1.414) }
        for (let j = ny - 2; j >= 0; j--) for (let i = nx - 2; i > 0; i--) { const k = j * nx + i; if (dist[k]) dist[k] = Math.min(dist[k], dist[k + 1] + 1, dist[k + nx] + 1, dist[k + nx + 1] + 1.414, dist[k + nx - 1] + 1.414) }
        let dmax = 1
        for (let k = 0; k < nx * ny; k++) if (dist[k] < 1e8 && dist[k] > dmax) dmax = dist[k]
        // spires: interior cells picked by a seeded hash, each a gaussian of extra depth
        const inside = (k: number) => Math.min(1, dist[k] / dmax)
        const spires: { x: number; z: number; a: number; s: number }[] = []
        for (let tries = 0; spires.length < 6 && tries < 4000; tries++) {
          const k = Math.floor(hash(tries, 5, 51) * nx * ny)
          if (inside(k) < 0.25 || inside(k) > 0.8) continue
          const x = pos[3 * k], z = pos[3 * k + 2]
          if (spires.some((p) => Math.hypot(p.x - x, p.z - z) < 1.6)) continue
          spires.push({ x, z, a: 0.5 + 0.9 * hash(tries, 7, 53), s: 0.35 + 0.35 * hash(tries, 9, 57) })
        }
        const upos = new Float32Array(nx * ny * 3), uuv = new Float32Array(nx * ny * 2), ucol = new Float32Array(nx * ny * 3)
        for (let k = 0; k < nx * ny; k++) {
          const x = pos[3 * k], ytop = pos[3 * k + 1], z = pos[3 * k + 2]
          const q = inside(k)
          // a short cliff at the rim, then an inverted cone to the deepest point; ridged noise breaks it into crags
          const ridge = 1 - Math.abs(2 * vnoise(x * 1.1, z * 1.1, 0, 0, 61) - 1)
          let d = (0.22 * smooth(0, 0.04, q) + DEPTH * Math.pow(q, 0.8)) * (0.7 + 0.42 * ridge + 0.3 * (fbm(x * 0.35, z * 0.35, 3, 0, 0, 63) - 0.5))
          for (const p of spires) d += p.a * q * Math.exp(-((x - p.x) ** 2 + (z - p.z) ** 2) / (p.s * p.s))
          const y = ytop - d
          upos[3 * k] = x; upos[3 * k + 1] = y; upos[3 * k + 2] = z
          const t = Math.min(1, d / (DEPTH * 1.6))
          const dk = 1 - 0.28 * Math.pow(t, 1.1) // gentle darkening with depth: the rock stays light against the paper page
          const soil = 1 - smooth(0.0, 0.12, d) // dark topsoil right under the surface edge
          ucol[3 * k] = dk * (1 - 0.25 * soil); ucol[3 * k + 1] = dk * (1 - 0.28 * soil); ucol[3 * k + 2] = dk * (1 - 0.3 * soil)
          if (y < minY && F.v[k] > F.T) minY = y
          if (q > 0 && k % 97 === 0) outline.push(new Vector3(x, y, z))
        }
        const flip: number[] = []
        for (let i = 0; i < idxs.length; i += 3) flip.push(idxs[i], idxs[i + 2], idxs[i + 1]) // faces point down
        const ugeo = new BufferGeometry()
        ugeo.setAttribute('position', new BufferAttribute(upos, 3))
        ugeo.setAttribute('uv', new BufferAttribute(new Float32Array(uv), 2)) // surface uv: the alpha cut matches the surface exactly
        ugeo.setAttribute('uv1', new BufferAttribute(uuv, 2)) // planar uv for the rock beds (strata.channel = 1)
        ugeo.setAttribute('color', new BufferAttribute(ucol, 3))
        ugeo.setIndex(new BufferAttribute(new Uint32Array(flip), 1))
        ugeo.computeVertexNormals()
        // rock uv by the direction each point faces (a per-vertex box projection), so no wall stretches the texture
        const nrm = ugeo.getAttribute('normal').array as Float32Array
        const RS = 0.55
        for (let k = 0; k < nx * ny; k++) {
          const ax = Math.abs(nrm[3 * k]), ay = Math.abs(nrm[3 * k + 1]), az = Math.abs(nrm[3 * k + 2])
          const x = upos[3 * k], y = upos[3 * k + 1], z = upos[3 * k + 2]
          if (ay > ax && ay > az) { uuv[2 * k] = x * RS; uuv[2 * k + 1] = z * RS }
          else if (ax > az) { uuv[2 * k] = z * RS; uuv[2 * k + 1] = y * RS }
          else { uuv[2 * k] = x * RS; uuv[2 * k + 1] = y * RS }
        }
        ;(ugeo.getAttribute('uv1') as BufferAttribute).needsUpdate = true
        const under = new Mesh(ugeo, rockMat)
        under.castShadow = under.receiveShadow = !small
        root.add(under)
        disposers.push(() => { ugeo.dispose(); rockMat.dispose(); rDiff.dispose(); rNor.dispose(); rRough.dispose() })

        /* ---- country outlines draped on the terrain (very light) ---- */
        const lineMat = new LineBasicMaterial({ color: '#3B4350', transparent: true, opacity: 0.5 })
        for (const rings of Object.values(map.countries)) for (const r of rings) {
          const pts: number[] = []
          for (let s = 0; s < r.length; s++) {
            const [a, b] = [r[s], r[(s + 1) % r.length]]
            const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.04))
            for (let q = 0; q < n; q++) {
              const l0 = a[0] + ((b[0] - a[0]) * q) / n, a0 = a[1] + ((b[1] - a[1]) * q) / n
              const l1 = a[0] + ((b[0] - a[0]) * (q + 1)) / n, a1 = a[1] + ((b[1] - a[1]) * (q + 1)) / n
              if (F.at(l0, a0) < F.T + 0.03 || F.at(l1, a1) < F.T + 0.03) continue
              pts.push(wx(l0), Math.max(dem.at(l0, a0), 0) * K + 0.014, wz(a0), wx(l1), Math.max(dem.at(l1, a1), 0) * K + 0.014, wz(a1))
            }
          }
          const lg = new BufferGeometry()
          lg.setAttribute('position', new BufferAttribute(new Float32Array(pts), 3))
          root.add(new LineSegments(lg, lineMat))
          disposers.push(() => lg.dispose())
        }
        disposers.push(() => lineMat.dispose())

        /* ---- earthquake main shocks: thin columns capped by small beads, at their true locations ---- */
        const col = new CylinderGeometry(1, 1, 1, 10).translate(0, 0.5, 0)
        const ball = new SphereGeometry(1, 16, 12)
        const shown = map.quakes.filter((q) => F.at(q[1], q[0]) > F.T + 0.05)
        const solo = shown.filter((q) => !q[4]), shared = shown.filter((q) => q[4])
        const M4 = new Matrix4(), Q = new Quaternion(), P = new Vector3(), S = new Vector3()
        type Item = { x: number; y: number; z: number; h: number; rc: number; rb: number; year: number }
        const rises: { cols: InstancedMesh; balls: InstancedMesh; items: Item[] }[] = []
        const mk = (list: number[][], color: string, opacity: number, emissive: string, ei: number) => {
          const cm = new MeshStandardMaterial({ color, roughness: 0.55, transparent: opacity < 1, opacity, emissive, emissiveIntensity: ei })
          const bm = new MeshStandardMaterial({ color, roughness: 0.35, transparent: opacity < 1, opacity: Math.min(1, opacity + 0.25), emissive, emissiveIntensity: ei })
          const cols = new InstancedMesh(col, cm, list.length), balls = new InstancedMesh(ball, bm, list.length)
          cols.frustumCulled = balls.frustumCulled = false
          const items: Item[] = []
          list.forEach(([la, lo, mag, year], i) => {
            const m = mag - 5.5
            const h = 0.13 + m * 0.2
            const rc = 0.0055 + m * 0.0024, rb = 0.02 + m * 0.011
            const x = wx(lo), z = wz(la), y = Math.max(dem.at(lo, la), 0) * K
            P.set(x, y, z); S.set(rc, h, rc); M4.compose(P, Q, S); cols.setMatrixAt(i, M4)
            P.set(x, y + h, z); S.set(rb, rb, rb); M4.compose(P, Q, S); balls.setMatrixAt(i, M4)
            items.push({ x, y, z, h, rc, rb, year })
          })
          rises.push({ cols, balls, items })
          root.add(cols, balls)
          disposers.push(() => { cm.dispose(); bm.dispose() })
        }
        mk(solo, '#2A3342', 0.72, '#000000', 0)
        mk(shared, '#C8553A', 1, '#E0603A', 0.55)
        const haloTex = halo()
        const hm = new SpriteMaterial({ map: haloTex, blending: AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.8 })
        const sprites: { s: Sprite; year: number; base: number }[] = []
        shared.forEach(([la, lo, mag, year]) => {
          const m = mag - 5.5, h = 0.13 + m * 0.2
          const s = new Sprite(hm)
          s.position.set(wx(lo), Math.max(dem.at(lo, la), 0) * K + h, wz(la))
          s.scale.setScalar(0.34 + m * 0.12)
          root.add(s)
          sprites.push({ s, year, base: 0.34 + m * 0.12 })
        })
        /* ---- replay of the catalogue, 1965 to 2016: each main shock rises in its year; the shared zone brightens only when a
           shock that reached both countries rises. Every value comes from map.quakes; nothing here is simulated. ---- */
        const Y0 = 1965, Y1 = 2016, PLAY = 15, HOLD = 3
        let lastYear = -1
        const setYear = (yr: number) => {
          const rise = (y0: number) => Math.min(1, Math.max(0, (yr - y0) / 0.8))
          for (const g of rises) {
            g.items.forEach((it, i) => {
              const p = rise(it.year), e = 1 - (1 - p) ** 3
              P.set(it.x, it.y, it.z); S.set(it.rc, Math.max(1e-4, it.h * e), it.rc); M4.compose(P, Q, S); g.cols.setMatrixAt(i, M4)
              const b = p > 0 ? it.rb : 1e-4
              P.set(it.x, it.y + it.h * e, it.z); S.set(b, b, b); M4.compose(P, Q, S); g.balls.setMatrixAt(i, M4)
            })
            g.cols.instanceMatrix.needsUpdate = g.balls.instanceMatrix.needsUpdate = true
          }
          let flash = 0
          for (const sp of sprites) {
            const p = rise(sp.year)
            sp.s.visible = p > 0
            sp.s.scale.setScalar(sp.base * (0.4 + 0.6 * p))
            if (yr >= sp.year) flash = Math.max(flash, Math.exp(-(yr - sp.year) / 0.6))
          }
          zoneMat.opacity = 0.72 + 0.28 * flash
          const whole = Math.floor(yr)
          if (whole !== lastYear) {
            lastYear = whole
            const upTo = map.quakes.filter((q) => q[3] <= whole)
            cb.current.onYear?.({ year: Math.min(whole, Y1), all: upTo.length, both: upTo.filter((q) => q[4]).length })
          }
        }
        disposers.push(() => { col.dispose(); ball.dispose(); hm.dispose(); haloTex.dispose() })

        root.position.set(-cx, 0.3, -cz)

        /* ---- studio light: warm key, cool fill, rim, soft ambient ---- */
        scene.add(new HemisphereLight(0xf7f4ee, 0xbfb6a4, 0.95))
        scene.add(new AmbientLight(0xffffff, 0.12))
        // low warm key from the north-west, as the hillshade convention, so the relief reads at a glance
        const key = new DirectionalLight(0xffeedd, 3.1)
        key.position.set(-8, 7, -3)
        key.castShadow = !small
        key.shadow.mapSize.set(2048, 2048)
        key.shadow.bias = -0.0006
        key.shadow.normalBias = 0.02
        key.shadow.radius = 4
        scene.add(key)
        const rimL = new DirectionalLight(0xdfe9ff, 0.9)
        rimL.position.set(6, 3, -7)
        scene.add(rimL)
        const fill = new DirectionalLight(0xffffff, 0.35)
        fill.position.set(5, 2, 7)
        scene.add(fill)

        // brand-coloured overlays (shared zone, columns, halos, borders) keep their exact colours: only the lit terrain is tone-mapped
        scene.traverse((o) => {
          const m = (o as Mesh).material as { isMeshBasicMaterial?: boolean; isSpriteMaterial?: boolean; isLineBasicMaterial?: boolean; toneMapped?: boolean } | undefined
          if (m && (m.isMeshBasicMaterial || m.isSpriteMaterial || m.isLineBasicMaterial)) m.toneMapped = false
        })

        /* ---- camera ---- */
        const bw = maxX - minX, bd = maxZ - minZ
        { const sc = key.shadow.camera, R = Math.max(bw, bd) * 0.7; sc.left = -R; sc.right = R; sc.top = R; sc.bottom = -R; sc.near = 0.5; sc.far = 40; sc.updateProjectionMatrix() }
        const pol = still?.pol ?? 1.04
        const yawMax = still ? 0 : 0.36
        const yaw = Math.max(yawMax, Math.abs(still?.az ?? 0))
        // frame the whole floating mountain, hanging spires included: the box corners (at the extremes of the slow yaw)
        // must project inside the frame; the distance is found by bisection for the current aspect
        const box: Vector3[] = []
        for (const q of outline) {
          const x = q.x - cx, y = q.y + 0.3 + 0.12, z = q.z - cz // +0.12: room for the earthquake columns above the rim
          for (const a of [-yaw, 0, yaw]) box.push(new Vector3(x * Math.cos(a) + z * Math.sin(a), y, -x * Math.sin(a) + z * Math.cos(a)))
        }
        const aim = new Vector3(0, (minY + maxY) / 2 + 0.3 + (maxY - minY) * 0.06, 0.1)
        const put = (d: number) => {
          camera.position.set(aim.x, aim.y + Math.cos(pol) * d, aim.z + Math.sin(pol) * d)
          camera.lookAt(aim)
          camera.updateMatrixWorld()
          camera.updateProjectionMatrix()
        }
        const fitDist = () => {
          let lo = 2, hi = 150 // the far plane is 400, so every distance tried here is fully inside the frustum
          for (let i = 0; i < 34; i++) {
            const mid = (lo + hi) / 2
            put(mid)
            const ok = box.every((c) => { const p = c.clone().project(camera); return Math.abs(p.x) < 0.96 && Math.abs(p.y) < 0.93 })
            if (ok) hi = mid; else lo = mid
          }
          put(hi)
        }
        const PV = new Vector3()
        const placePins = () => {
          const els = cb.current.pinEls?.current, list = cb.current.pins
          if (!els || !list) return
          scene.updateMatrixWorld()
          const w = el.clientWidth, h = el.clientHeight
          for (const p of list) {
            const e = els[p.id]
            if (!e) continue
            PV.set(wx(p.lon), Math.max(dem.at(p.lon, p.lat), 0) * K + 0.05, wz(p.lat))
            root.localToWorld(PV)
            PV.project(camera)
            const px = ((PV.x + 1) / 2) * w, py = ((1 - PV.y) / 2) * h
            // near the right edge the label flips to the left of its dot, so it never leaves the frame
            const left = px + e.offsetWidth > w - 8
            e.dataset.side = left ? 'l' : 'r'
            e.style.transform = `translate(${left ? px - e.offsetWidth + 14 : px}px, ${py}px)`
          }
        }
        const fit = () => {
          const w = el.clientWidth, h = el.clientHeight
          if (!w || !h) return
          renderer.setSize(w, h, false)
          renderer.domElement.style.width = w + 'px'
          renderer.domElement.style.height = h + 'px'
          camera.aspect = w / h
          fitDist()
          renderer.render(scene, camera)
          placePins()
        }
        const ro = new ResizeObserver(fit)
        ro.observe(el)
        disposers.push(() => ro.disconnect())
        pivot.rotation.y = still?.az ?? 0
        fit()

        const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
        let visible = true
        const io = new IntersectionObserver((es) => { visible = es[0].isIntersecting })
        io.observe(el)
        disposers.push(() => io.disconnect())
        const t0 = performance.now()
        let yawT = 0, lastT = 0
        const frame = el.parentElement
        const loop = (now: number) => {
          raf = requestAnimationFrame(loop)
          if (!visible || document.hidden) return
          const t = (now - t0) / 1000
          // the slow turn stops while a pointer rests on a hotspot, so a moving target is never what someone has to click
          if (!cb.current.hold?.current) yawT += Math.min(0.1, t - lastT)
          lastT = t
          const bob = 0 // no bob or pulse: motion that is not data reads as changing risk
          pivot.rotation.y = Math.sin((yawT * Math.PI * 2) / 46) * yawMax
          pivot.position.y = bob * 0.07
          hm.opacity = 0.75
          const tr = t % (PLAY + HOLD)
          setYear(Y0 + Math.min(1, tr / PLAY) * (Y1 - Y0 + 1))
          frame?.style.setProperty('--bob', bob.toFixed(3))
          renderer.render(scene, camera)
          placePins()
        }
        if (!still && !reduce) raf = requestAnimationFrame(loop)
        cb.current.onReady?.()
      } catch (e) {
        console.error('hero terrain failed', e)
        if (!dead) cb.current.onFail?.()
      }
    })()
    return () => {
      dead = true
      cancelAnimationFrame(raf)
      disposers.forEach((d) => { try { d() } catch { /* ignore */ } })
    }
  }, [map, still])

  return <div className="hero-gl" ref={host} />
}

