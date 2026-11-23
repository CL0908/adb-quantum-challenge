import { useEffect, useRef } from 'react'
import {
  AdditiveBlending, BufferAttribute, BufferGeometry, CanvasTexture, CylinderGeometry, InstancedMesh, LineBasicMaterial, LineLoop,
  LinearFilter, Matrix4, Mesh, MeshBasicMaterial, PerspectiveCamera, Quaternion, Scene, SphereGeometry, SRGBColorSpace, Sprite,
  SpriteMaterial, TOUCH, Vector3, WebGLRenderer,
} from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { dataUrl } from './data'
import type { MapData } from '../lib/map'
import { DARK, rgb } from '../shared/palette'

const COS = Math.cos((40 * Math.PI) / 180)
const EXAG = 7
const K = EXAG / 111320 // metres to world units (1 unit = one degree of latitude)

interface DemInfo { extent: [number, number, number, number]; width: number; height: number; offset_m: number }
interface Dem { info: DemInfo; h: Float32Array; at: (lon: number, lat: number) => number }

async function loadDem(): Promise<Dem> {
  const info: DemInfo = await fetch(dataUrl('dem.json')).then((r) => (r.ok ? r.json() : Promise.reject(new Error('dem.json'))))
  const blob = await fetch(dataUrl('dem.png')).then((r) => (r.ok ? r.blob() : Promise.reject(new Error('dem.png'))))
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  const c = document.createElement('canvas')
  c.width = bmp.width
  c.height = bmp.height
  const g = c.getContext('2d', { willReadFrequently: true })!
  g.drawImage(bmp, 0, 0)
  const px = g.getImageData(0, 0, c.width, c.height).data
  if (c.width !== info.width || c.height !== info.height) throw new Error('dem size mismatch')
  const h = new Float32Array(c.width * c.height)
  for (let i = 0; i < h.length; i++) h[i] = px[4 * i] * 256 + px[4 * i + 1] - info.offset_m
  const [lon0, lat0, lon1, lat1] = info.extent
  const at = (lon: number, lat: number) => {
    const fx = ((lon - lon0) / (lon1 - lon0)) * (info.width - 1)
    const fy = ((lat1 - lat) / (lat1 - lat0)) * (info.height - 1)
    const x = Math.min(Math.max(fx, 0), info.width - 1.001)
    const y = Math.min(Math.max(fy, 0), info.height - 1.001)
    const i = Math.floor(x), j = Math.floor(y), u = x - i, w = y - j
    const k = j * info.width + i
    return (1 - u) * (1 - w) * h[k] + u * (1 - w) * h[k + 1] + (1 - u) * w * h[k + info.width] + u * w * h[k + info.width + 1]
  }
  return { info, h, at }
}

type RGB = [number, number, number]
const RAMP: [number, RGB][] = [
  [0, [10, 17, 27]], [1200, [18, 30, 46]], [2500, [30, 50, 78]], [3600, [74, 88, 108]], [4800, [168, 164, 148]], [6200, [236, 228, 208]],
]
function ramp(m: number): RGB {
  if (m <= 0) return RAMP[0][1]
  for (let i = 1; i < RAMP.length; i++) {
    if (m <= RAMP[i][0]) {
      const [a, ca] = RAMP[i - 1], [b, cb] = RAMP[i]
      const t = (m - a) / (b - a)
      return [ca[0] + (cb[0] - ca[0]) * t, ca[1] + (cb[1] - ca[1]) * t, ca[2] + (cb[2] - ca[2]) * t]
    }
  }
  return RAMP[RAMP.length - 1][1]
}
const smooth = (a: number, b: number, x: number) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t) }

interface Frame { lon0: number; lat0: number; lon1: number; lat1: number; lonC: number; latC: number }
const wx = (f: Frame, lon: number) => (lon - f.lonC) * COS
const wz = (f: Frame, lat: number) => -(lat - f.latC)

/** Dark relief: hypsometric deep teal, baked hillshade from the north-west, thin contour lines, countries lit, the rest dimmed. */
function paintGround(dem: Dem, map: MapData, f: Frame, W: number, H: number) {
  const tx = (lon: number) => ((lon - f.lon0) / (f.lon1 - f.lon0)) * W
  const ty = (lat: number) => ((f.lat1 - lat) / (f.lat1 - f.lat0)) * H
  const m = document.createElement('canvas')
  m.width = W
  m.height = H
  const mg = m.getContext('2d', { willReadFrequently: true })!
  mg.fillStyle = 'rgb(0,0,0)'
  mg.fillRect(0, 0, W, H)
  mg.globalCompositeOperation = 'lighter'
  for (const [iso, col] of [['KG', 'rgb(255,0,0)'], ['TJ', 'rgb(0,255,0)']] as const) {
    mg.fillStyle = col
    mg.beginPath()
    for (const r of map.countries[iso] ?? []) {
      r.forEach(([lo, la], i) => (i ? mg.lineTo(tx(lo), ty(la)) : mg.moveTo(tx(lo), ty(la))))
      mg.closePath()
    }
    mg.fill()
  }
  const mask = mg.getImageData(0, 0, W, H).data
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const g = c.getContext('2d')!
  const img = g.createImageData(W, H)
  const dLon = (f.lon1 - f.lon0) / W, dLat = (f.lat1 - f.lat0) / H
  const mx = dLon * 111320 * COS, my = dLat * 111320
  const az = (315 * Math.PI) / 180, alt = (38 * Math.PI) / 180
  const lx = Math.sin(az) * Math.cos(alt), ly = Math.cos(az) * Math.cos(alt), lz = Math.sin(alt)
  const Z = 2.4
  const BAND = 450
  for (let y = 0; y < H; y++) {
    const lat = f.lat1 - ((y + 0.5) / H) * (f.lat1 - f.lat0)
    for (let x = 0; x < W; x++) {
      const lon = f.lon0 + ((x + 0.5) / W) * (f.lon1 - f.lon0)
      const h0 = dem.at(lon, lat), hE = dem.at(lon + dLon, lat), hW = dem.at(lon - dLon, lat), hN = dem.at(lon, lat + dLat), hS = dem.at(lon, lat - dLat)
      const dzx = ((hE - hW) / (2 * mx)) * Z, dzy = ((hN - hS) / (2 * my)) * Z
      const nl = Math.hypot(dzx, dzy, 1)
      const lam = (-dzx * lx - dzy * ly + lz) / nl
      const shade = Math.min(1.25, Math.max(0, 0.42 + (lam - lz) * 1.7 + 0.35))
      const base = ramp(h0)
      const k = 0.3 + 0.95 * Math.pow(shade, 1.15)
      let r = base[0] * k, gg = base[1] * k, b = base[2] * k
      // contour lines every BAND metres
      const band = Math.floor(h0 / BAND)
      const edge = band !== Math.floor(hE / BAND) || band !== Math.floor(hS / BAND)
      const i = 4 * (y * W + x)
      const kg = mask[i] > 127, tj = mask[i + 1] > 127
      const inside = kg || tj
      if (edge && h0 > 200) { const a = inside ? 0.16 : 0.06; r += (190 - r) * a; gg += (210 - gg) * a; b += (236 - b) * a }
      if (inside) {
        if (kg) { r += 0; gg += 12; b += 12 } else { r += 4; gg += 9; b += 20 }
      } else {
        const l = (r + gg + b) / 3
        r = r * 0.35 + l * 0.12; gg = gg * 0.4 + l * 0.12; b = b * 0.42 + l * 0.14
      }
      const u = (x + 0.5) / W, v = (y + 0.5) / H
      const fade = smooth(0, 0.07, Math.min(u, 1 - u)) * smooth(0, 0.1, Math.min(v, 1 - v))
      img.data[i] = Math.min(255, r); img.data[i + 1] = Math.min(255, gg); img.data[i + 2] = Math.min(255, b)
      img.data[i + 3] = Math.round(255 * fade)
    }
  }
  g.putImageData(img, 0, 0)
  return c
}

/** Glowing country outlines and the shared border zone (additive overlay). */
function paintOverlay(map: MapData, f: Frame, W: number, H: number) {
  const tx = (lon: number) => ((lon - f.lon0) / (f.lon1 - f.lon0)) * W
  const ty = (lat: number) => ((f.lat1 - lat) / (f.lat1 - f.lat0)) * H
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const g = c.getContext('2d')!
  const path = (rings: number[][][]) => {
    g.beginPath()
    for (const r of rings) {
      r.forEach(([lo, la], i) => (i ? g.lineTo(tx(lo), ty(la)) : g.moveTo(tx(lo), ty(la))))
      g.closePath()
    }
  }
  for (const [iso, col] of [['KG', rgb(DARK.teal)], ['TJ', rgb(DARK.navyLight)]] as const) {
    path(map.countries[iso] ?? [])
    g.shadowColor = `rgba(${col},0.9)`
    g.shadowBlur = W * 0.006
    g.strokeStyle = `rgba(${col},0.75)`
    g.lineWidth = Math.max(1.5, W * 0.0011)
    g.stroke()
    g.shadowBlur = 0
  }
  path(map.zone)
  g.shadowColor = `rgba(${rgb(DARK.terra)},1)`
  g.shadowBlur = W * 0.02
  g.fillStyle = `rgba(${rgb(DARK.terra)},0.55)`
  g.fill('evenodd')
  g.shadowBlur = 0
  g.strokeStyle = `rgba(${rgb(DARK.sand)},0.95)`
  g.lineWidth = Math.max(1.5, W * 0.0012)
  g.stroke()
  return c
}

function label(text: string, color: string, size: number) {
  const c = document.createElement('canvas')
  c.width = 900
  c.height = 130
  const g = c.getContext('2d')!
  g.font = `600 ${size}px Inter, sans-serif`
  g.textAlign = 'center'
  g.textBaseline = 'middle'
  try { (g as unknown as { letterSpacing: string }).letterSpacing = `${Math.round(size * 0.14)}px` } catch { /* older engines */ }
  g.shadowColor = `rgba(${rgb(DARK.bg)},0.9)`
  g.shadowBlur = 14
  g.fillStyle = color
  g.fillText(text.toUpperCase(), 450, 65)
  const t = new CanvasTexture(c)
  t.colorSpace = SRGBColorSpace
  const s = new Sprite(new SpriteMaterial({ map: t, depthTest: false, transparent: true }))
  s.scale.set(4.2, 4.2 * (130 / 900), 1)
  s.renderOrder = 10
  return s
}

export interface AnchorDef { id: string; lon: number; lat: number }
export interface Insets { l: number; r: number; t: number; b: number }
export interface TerrainApi { setWeight: (w: number) => void; setFocus: (stage: number) => void; reset: () => void }

interface Props {
  map: MapData
  eq: number
  stage: number
  anchors: AnchorDef[]
  anchorEls: { current: Record<string, HTMLElement | null> }
  getInsets: () => Insets
  onFail: () => void
  onReady?: () => void
}

export default function Terrain({ map, eq, stage, anchors, anchorEls, getInsets, onFail, onReady }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const api = useRef<TerrainApi | null>(null)
  const eqRef = useRef(eq)
  const stageRef = useRef(stage)
  eqRef.current = eq
  stageRef.current = stage
  const cb = useRef({ onFail, onReady, getInsets, anchors })
  cb.current = { onFail, onReady, getInsets, anchors }

  useEffect(() => {
    const el = host.current
    if (!el) return
    let dead = false
    let raf = 0
    const disposers: (() => void)[] = []
    ;(async () => {
      let renderer: WebGLRenderer
      try {
        renderer = new WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' })
      } catch {
        if (!dead) cb.current.onFail()
        return
      }
      let dem: Dem
      try {
        dem = await loadDem()
        await Promise.race([document.fonts.load('300 40px "Barlow Condensed"'), new Promise((r) => setTimeout(r, 1200))])
      } catch {
        renderer.dispose()
        if (!dead) cb.current.onFail()
        return
      }
      if (dead) { renderer.dispose(); return }
      try {
        const [lon0, lat0, lon1, lat1] = dem.info.extent
        const f: Frame = { lon0, lat0, lon1, lat1, lonC: (lon0 + lon1) / 2, latC: (lat0 + lat1) / 2 }
        const nx = dem.info.width, ny = dem.info.height
        const stepLon = (lon1 - lon0) / (nx - 1), stepLat = (lat1 - lat0) / (ny - 1)
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
        renderer.setClearColor(0x000000, 0)
        renderer.domElement.className = 'terrain-canvas'
        el.prepend(renderer.domElement)
        disposers.push(() => { renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove() })
        const scene = new Scene()
        const camera = new PerspectiveCamera(28, 1, 0.5, 120)

        const pos = new Float32Array(nx * ny * 3), uv = new Float32Array(nx * ny * 2)
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const k = j * nx + i
          pos[3 * k] = wx(f, lon0 + i * stepLon)
          pos[3 * k + 1] = Math.max(dem.h[k], 0) * K
          pos[3 * k + 2] = wz(f, lat1 - j * stepLat)
          uv[2 * k] = i / (nx - 1)
          uv[2 * k + 1] = 1 - j / (ny - 1)
        }
        const idx = new Uint32Array((nx - 1) * (ny - 1) * 6)
        let q = 0
        for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
          const a = j * nx + i, b = a + nx, c = a + 1, d = b + 1
          idx[q++] = a; idx[q++] = b; idx[q++] = c; idx[q++] = c; idx[q++] = b; idx[q++] = d
        }
        const geo = new BufferGeometry()
        geo.setAttribute('position', new BufferAttribute(pos, 3))
        geo.setAttribute('uv', new BufferAttribute(uv, 2))
        geo.setIndex(new BufferAttribute(idx, 1))
        const TW = window.innerWidth > 1100 ? 2048 : 1280
        const TH = Math.round((TW * (lat1 - lat0)) / ((lon1 - lon0) * COS))
        await new Promise((r) => setTimeout(r, 0))
        const ground = new CanvasTexture(paintGround(dem, map, f, TW, TH))
        ground.colorSpace = SRGBColorSpace
        ground.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy())
        ground.minFilter = LinearFilter
        const mat = new MeshBasicMaterial({ map: ground, transparent: true, depthWrite: true })
        scene.add(new Mesh(geo, mat))
        disposers.push(() => { geo.dispose(); mat.dispose(); ground.dispose() })

        const ovTex = new CanvasTexture(paintOverlay(map, f, TW, TH))
        ovTex.colorSpace = SRGBColorSpace
        const ovMat = new MeshBasicMaterial({ map: ovTex, transparent: true, depthWrite: false, blending: AdditiveBlending, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 })
        scene.add(new Mesh(geo, ovMat))
        disposers.push(() => { ovMat.dispose(); ovTex.dispose() })

        // crisp outline lines lifted above the relief
        const lineMats: LineBasicMaterial[] = []
        for (const [iso, color] of [['KG', DARK.teal], ['TJ', DARK.navyLight]] as const) {
          const lm = new LineBasicMaterial({ color, transparent: true, opacity: 0.9 })
          lineMats.push(lm)
          for (const r of map.countries[iso] ?? []) {
            const pts: number[] = []
            for (let s = 0; s < r.length; s++) {
              const [a, b] = [r[s], r[(s + 1) % r.length]]
              const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.04))
              for (let t = 0; t < n; t++) {
                const lo = a[0] + ((b[0] - a[0]) * t) / n, la = a[1] + ((b[1] - a[1]) * t) / n
                pts.push(wx(f, lo), Math.max(dem.at(lo, la), 0) * K + 0.012, wz(f, la))
              }
            }
            const lg = new BufferGeometry()
            lg.setAttribute('position', new BufferAttribute(new Float32Array(pts), 3))
            scene.add(new LineLoop(lg, lm))
            disposers.push(() => lg.dispose())
          }
        }
        disposers.push(() => lineMats.forEach((m2) => m2.dispose()))

        // earthquake columns at their true locations
        const col = new CylinderGeometry(1, 1, 1, 6).translate(0, 0.5, 0)
        const ball = new SphereGeometry(1, 12, 8)
        const shown = map.quakes.filter((x) => x[1] >= lon0 && x[1] <= lon1 && x[0] >= lat0 && x[0] <= lat1)
        const mk = (list: number[][], color: string, opacity: number) => {
          const cm = new MeshBasicMaterial({ color, transparent: true, opacity: opacity * 0.8, blending: AdditiveBlending, depthWrite: false })
          const bm = new MeshBasicMaterial({ color, transparent: true, opacity })
          const cols = new InstancedMesh(col, cm, list.length), balls = new InstancedMesh(ball, bm, list.length)
          cols.frustumCulled = balls.frustumCulled = false
          scene.add(cols, balls)
          disposers.push(() => { cm.dispose(); bm.dispose() })
          return { list, cols, balls, cm, bm, o: opacity }
        }
        const gS = mk(shown.filter((x) => !x[4]), DARK.teal, 0.55)
        const gB = mk(shown.filter((x) => x[4]), DARK.terra, 1)
        disposers.push(() => { col.dispose(); ball.dispose() })
        const M4 = new Matrix4(), Q = new Quaternion(), P = new Vector3(), S = new Vector3()
        const place = (g: ReturnType<typeof mk>, w: number, isShared: boolean, boost: number) => {
          g.list.forEach((qk, i) => {
            const [la, lo, mag] = qk
            const m = mag - 5.5
            const wt = (isShared ? 0.55 + 1.0 * w : 0.9) * boost
            const h = (0.12 + m * 0.2) * wt
            const rc = (0.006 + m * 0.0025) * (isShared ? 1 + 0.5 * w : 1)
            const rb = (0.022 + m * 0.011) * (isShared ? 0.8 + 0.5 * w : 0.9) * Math.sqrt(boost)
            const x = wx(f, lo), z = wz(f, la), y = Math.max(dem.at(lo, la), 0) * K
            P.set(x, y, z); S.set(rc, h, rc)
            M4.compose(P, Q, S)
            g.cols.setMatrixAt(i, M4)
            P.set(x, y + h, z); S.set(rb, rb, rb)
            M4.compose(P, Q, S)
            g.balls.setMatrixAt(i, M4)
          })
          g.cols.instanceMatrix.needsUpdate = g.balls.instanceMatrix.needsUpdate = true
        }

        const kgL = label('Kyrgyz Republic', DARK.teal, 52)
        const tjL = label('Tajikistan', DARK.navyLight, 52)
        kgL.position.set(wx(f, 73.4), 0.5, wz(f, 41.7))
        tjL.position.set(wx(f, 70.6), 0.55, wz(f, 38.25))
        scene.add(kgL, tjL)
        disposers.push(() => { kgL.material.map?.dispose(); kgL.material.dispose(); tjL.material.map?.dispose(); tjL.material.dispose() })

        let dirty = true
        let weight = 0, focus = 0
        const apply = () => {
          const boost = focus === 1 ? 1.35 : 1
          place(gS, weight, false, boost)
          place(gB, weight, true, boost)
          gS.bm.opacity = gS.o * (focus === 3 ? 0.35 : focus === 1 ? 1.3 : 1)
          gS.cm.opacity = gS.bm.opacity * 0.8
          ovMat.opacity = focus === 2 || focus === 3 ? 1 : 0.8 + 0.2 * Math.min(weight / 0.5, 1)
          lineMats.forEach((m2) => { m2.opacity = focus === 2 ? 1 : 0.8 })
          dirty = true
        }
        const setWeight = (e: number) => { weight = Math.min(Math.max(e / 0.5, 0), 1); apply() }
        const setFocus = (s: number) => { focus = s; apply() }
        weight = Math.min(Math.max(eqRef.current / 0.5, 0), 1)
        focus = stageRef.current
        apply()

        const controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.dampingFactor = 0.08
        controls.enablePan = false
        controls.minDistance = 7
        controls.maxDistance = 34
        controls.minPolarAngle = 0.3
        controls.maxPolarAngle = 1.3
        controls.target.set(0, 0.1, 0.1)
        controls.rotateSpeed = 0.55
        if (window.matchMedia('(pointer: coarse)').matches) {
          controls.touches = { ONE: -1 as unknown as TOUCH, TWO: TOUCH.DOLLY_ROTATE }
          renderer.domElement.style.touchAction = 'pan-y'
        }
        const rest = { az: 0.0, pol: 0.8, dist: 18 }
        let interacted = false, intro = false, t0 = 0
        const narrow = () => window.innerWidth <= 1100
        const fit = () => {
          const w = el.clientWidth, h = el.clientHeight
          if (!w || !h) return
          renderer.setSize(w, h, false)
          renderer.domElement.style.width = w + 'px'
          renderer.domElement.style.height = h + 'px'
          camera.aspect = w / h
          const ins = narrow() ? { l: 0, r: 0, t: 0, b: 0 } : cb.current.getInsets()
          const freeW = Math.max(w - 0.8 * (ins.l + ins.r), w * 0.4), freeH = Math.max(h - 0.8 * (ins.t + ins.b), h * 0.5)
          const tanV = Math.tan((camera.fov * Math.PI) / 360)
          const dW = (narrow() ? 12.6 : 11.4) / (2 * tanV * camera.aspect * (freeW / w))
          const dH = (narrow() ? 7.2 : 5.6) / (2 * tanV * (freeH / h))
          rest.dist = Math.max(dW, dH)
          const shiftX = (ins.l - ins.r) / 2, shiftY = (ins.t - ins.b) / 2
          camera.setViewOffset(w, h, -shiftX, -shiftY, w, h)
          camera.updateProjectionMatrix()
          if (!interacted && !intro) place0()
          dirty = true
        }
        const place0 = () => {
          const s = Math.sin(rest.pol) * rest.dist
          camera.position.set(controls.target.x + s * Math.sin(rest.az), controls.target.y + Math.cos(rest.pol) * rest.dist, controls.target.z + s * Math.cos(rest.az))
          camera.lookAt(controls.target)
        }
        controls.addEventListener('start', () => { interacted = true; intro = false })
        controls.addEventListener('change', () => { dirty = true })
        const ro = new ResizeObserver(fit)
        ro.observe(el)
        disposers.push(() => ro.disconnect())
        const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
        fit()
        // arriving from another tab of the site: show the settled view at once, the fly-in is for a first visit
        const fromSite = (() => { try { return new URL(document.referrer).origin === location.origin } catch { return false } })()
        if (!reduce && !fromSite) { intro = true; t0 = performance.now() } else place0()
        controls.update()
        api.current = {
          setWeight, setFocus,
          reset: () => { intro = true; interacted = false; t0 = performance.now(); controls.target.set(0, 0.1, 0.1) },
        }

        let visible = true
        const io = new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) dirty = true })
        io.observe(el)
        disposers.push(() => io.disconnect())

        const v3 = new Vector3()
        const project = () => {
          const w = el.clientWidth, h = el.clientHeight
          const off = narrow()
          for (const a of cb.current.anchors) {
            const node = anchorEls.current[a.id]
            if (!node) continue
            if (off) { node.style.transform = ''; continue }
            v3.set(wx(f, a.lon), Math.max(dem.at(a.lon, a.lat), 0) * K, wz(f, a.lat)).project(camera)
            node.style.transform = `translate3d(${((v3.x + 1) / 2) * w}px, ${((1 - v3.y) / 2) * h}px, 0)`
            node.style.opacity = v3.z < 1 ? '1' : '0'
          }
        }
        const ease = (t: number) => 1 - Math.pow(1 - t, 3)
        const loop = (now: number) => {
          raf = requestAnimationFrame(loop)
          if (!visible || document.hidden) return
          if (intro) {
            const t = Math.min((now - t0) / 2400, 1), e = ease(t)
            const cur = { az: rest.az - 0.45 * (1 - e), pol: rest.pol - 0.35 * (1 - e), dist: rest.dist * (1 + 0.2 * (1 - e)) }
            const s = Math.sin(cur.pol) * cur.dist
            camera.position.set(controls.target.x + s * Math.sin(cur.az), controls.target.y + Math.cos(cur.pol) * cur.dist, controls.target.z + s * Math.cos(cur.az))
            camera.lookAt(controls.target)
            dirty = true
            if (t >= 1) { intro = false; controls.update() }
          } else if (controls.update()) dirty = true
          if (dirty) { renderer.render(scene, camera); project(); dirty = false }
        }
        raf = requestAnimationFrame(loop)
        cb.current.onReady?.()
      } catch (e) {
        console.error('terrain failed', e)
        if (!dead) cb.current.onFail()
      }
    })()
    return () => {
      dead = true
      cancelAnimationFrame(raf)
      api.current = null
      disposers.forEach((d) => { try { d() } catch { /* ignore */ } })
    }
  }, [map, anchorEls])

  useEffect(() => { api.current?.setWeight(eq) }, [eq])
  useEffect(() => { api.current?.setFocus(stage) }, [stage])

  return (
    <div className="terrain" ref={host}>
      <button type="button" className="reset-view" onClick={() => api.current?.reset()}>Reset view</button>
    </div>
  )
}
