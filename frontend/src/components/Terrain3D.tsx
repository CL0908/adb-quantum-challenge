import { useEffect, useRef } from 'react'
import {
  AmbientLight, BufferAttribute, BufferGeometry, CanvasTexture, Color, CylinderGeometry, DirectionalLight, DoubleSide,
  HemisphereLight, InstancedMesh, LineBasicMaterial, LineLoop, LinearFilter, Matrix4, Mesh, Quaternion, MeshBasicMaterial,
  MeshStandardMaterial, PerspectiveCamera, Scene, SphereGeometry, SRGBColorSpace, Sprite, SpriteMaterial, TOUCH,
  Vector3, WebGLRenderer,
} from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { loadDem, type Dem } from '../lib/dem'
import type { MapData } from '../lib/map'

const COS = Math.cos((40 * Math.PI) / 180)
const EXAG = 9 // vertical exaggeration
const K = EXAG / 111320 // metres -> world units (1 unit = 1 degree of latitude)
const BASE = -0.14

type RGB = [number, number, number]
const hex = (s: string): RGB => [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)]
// hypsometric tint in the paper palette: warm sand lowlands to cool grey and snow
const RAMP: [number, RGB][] = [
  [0, hex('#D8D2B6')], [900, hex('#E2DAC0')], [1900, hex('#D3C6A5')], [2900, hex('#B9AC92')],
  [3800, hex('#A2A3A5')], [4800, hex('#C5CCD5')], [6000, hex('#F1F4F7')],
]
function ramp(m: number): RGB {
  if (m <= RAMP[0][0]) return RAMP[0][1]
  for (let i = 1; i < RAMP.length; i++) {
    if (m <= RAMP[i][0]) {
      const [a, ca] = RAMP[i - 1], [b, cb] = RAMP[i]
      const t = (m - a) / (b - a)
      return [ca[0] + (cb[0] - ca[0]) * t, ca[1] + (cb[1] - ca[1]) * t, ca[2] + (cb[2] - ca[2]) * t]
    }
  }
  return RAMP[RAMP.length - 1][1]
}
const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]

interface Frame { lon0: number; lat0: number; lon1: number; lat1: number; lonC: number; latC: number }
const wx = (f: Frame, lon: number) => (lon - f.lonC) * COS
const wz = (f: Frame, lat: number) => -(lat - f.latC)

function paintGround(dem: Dem, map: MapData, f: Frame, W: number, H: number) {
  const tx = (lon: number) => ((lon - f.lon0) / (f.lon1 - f.lon0)) * W
  const ty = (lat: number) => ((f.lat1 - lat) / (f.lat1 - f.lat0)) * H
  const m = document.createElement('canvas')
  m.width = W
  m.height = H
  const mg = m.getContext('2d', { willReadFrequently: true })!
  // red channel = Kyrgyz Republic, green = Tajikistan
  const fillCountry = (iso: string, col: string) => {
    mg.fillStyle = col
    mg.beginPath()
    for (const r of map.countries[iso] ?? []) {
      r.forEach(([lo, la], i) => (i ? mg.lineTo(tx(lo), ty(la)) : mg.moveTo(tx(lo), ty(la))))
      mg.closePath()
    }
    mg.fill()
  }
  mg.fillStyle = '#000'
  mg.fillRect(0, 0, W, H)
  mg.globalCompositeOperation = 'lighter'
  fillCountry('KG', '#ff0000')
  fillCountry('TJ', '#00ff00')
  const mask = mg.getImageData(0, 0, W, H).data
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const g = c.getContext('2d')!
  const img = g.createImageData(W, H)
  const paper = hex('#F1EADB'), kg = hex('#2F6AB0'), tj = hex('#AD7A00')
  for (let y = 0; y < H; y++) {
    const lat = f.lat1 - ((y + 0.5) / H) * (f.lat1 - f.lat0)
    for (let x = 0; x < W; x++) {
      const lon = f.lon0 + ((x + 0.5) / W) * (f.lon1 - f.lon0)
      let col = ramp(dem.at(lon, lat))
      const i = 4 * (y * W + x)
      const inKG = mask[i] > 127, inTJ = mask[i + 1] > 127
      if (inKG) col = mix(col, kg, 0.1)
      else if (inTJ) col = mix(col, tj, 0.13)
      else col = mix(mix(col, paper, 0.62), [200, 200, 200], 0.0)
      img.data[i] = col[0]; img.data[i + 1] = col[1]; img.data[i + 2] = col[2]; img.data[i + 3] = 255
    }
  }
  g.putImageData(img, 0, 0)
  return c
}

function paintZone(map: MapData, f: Frame, W: number, H: number) {
  const tx = (lon: number) => ((lon - f.lon0) / (f.lon1 - f.lon0)) * W
  const ty = (lat: number) => ((f.lat1 - lat) / (f.lat1 - f.lat0)) * H
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
  g.fillStyle = 'rgba(200,85,58,0.9)'
  g.shadowColor = 'rgba(224,110,70,1)'
  g.shadowBlur = W * 0.016
  path(); g.fill('evenodd'); path(); g.fill('evenodd')
  g.shadowBlur = 0
  g.strokeStyle = 'rgba(255,214,190,0.95)'
  g.lineWidth = Math.max(2, W * 0.0016)
  path(); g.stroke()
  return c
}

function label(text: string) {
  const c = document.createElement('canvas')
  c.width = 640
  c.height = 110
  const g = c.getContext('2d')!
  g.font = 'italic 600 54px "Source Serif 4", Georgia, serif'
  g.textAlign = 'center'
  g.textBaseline = 'middle'
  g.lineJoin = 'round'
  g.lineWidth = 12
  g.strokeStyle = 'rgba(246,241,231,0.92)'
  g.strokeText(text, 320, 55)
  g.fillStyle = '#1F3557'
  g.fillText(text, 320, 55)
  const t = new CanvasTexture(c)
  t.colorSpace = SRGBColorSpace
  const s = new Sprite(new SpriteMaterial({ map: t, depthTest: false, transparent: true }))
  s.scale.set(2.6, 2.6 * (110 / 640), 1)
  s.renderOrder = 10
  return s
}

export default function Terrain3D({ map, eq, onFail, onReady }: { map: MapData; eq: number; onFail: () => void; onReady?: () => void }) {
  const host = useRef<HTMLDivElement>(null)
  const eqRef = useRef(eq)
  const api = useRef<{ setWeight: (w: number) => void; reset: () => void } | null>(null)
  eqRef.current = eq
  const cb = useRef({ onFail, onReady })
  cb.current = { onFail, onReady }

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

        // ---- terrain mesh
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
        geo.computeVertexNormals()
        const TW = Math.min(2048, Math.max(1024, Math.round(window.innerWidth > 700 ? 2048 : 1400)))
        const TH = Math.round((TW * (lat1 - lat0)) / ((lon1 - lon0) * COS))
        const ground = new CanvasTexture(paintGround(dem, map, f, TW, TH))
        ground.colorSpace = SRGBColorSpace
        ground.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy())
        ground.minFilter = LinearFilter
        const mat = new MeshStandardMaterial({ map: ground, roughness: 1, metalness: 0 })
        const terrain = new Mesh(geo, mat)
        scene.add(terrain)
        disposers.push(() => { geo.dispose(); mat.dispose(); ground.dispose() })

        // ---- shared border zone, draped on the same geometry
        const zoneTex = new CanvasTexture(paintZone(map, f, TW, TH))
        zoneTex.colorSpace = SRGBColorSpace
        const zoneMat = new MeshBasicMaterial({ map: zoneTex, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 })
        scene.add(new Mesh(geo, zoneMat))
        disposers.push(() => { zoneMat.dispose(); zoneTex.dispose() })

        // ---- skirt (the model's side walls)
        const wall: number[] = []
        const edge = (ids: number[]) => {
          for (let s = 0; s < ids.length - 1; s++) {
            const a = ids[s], b = ids[s + 1]
            const A = [pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]], B = [pos[3 * b], pos[3 * b + 1], pos[3 * b + 2]]
            wall.push(...A, A[0], BASE, A[2], ...B, ...B, A[0], BASE, A[2], B[0], BASE, B[2])
          }
        }
        edge(Array.from({ length: nx }, (_, i) => i))
        edge(Array.from({ length: nx }, (_, i) => (ny - 1) * nx + i))
        edge(Array.from({ length: ny }, (_, j) => j * nx))
        edge(Array.from({ length: ny }, (_, j) => j * nx + nx - 1))
        const wg = new BufferGeometry()
        wg.setAttribute('position', new BufferAttribute(new Float32Array(wall), 3))
        wg.computeVertexNormals()
        const wm = new MeshStandardMaterial({ color: new Color('#B9AE98'), roughness: 1, side: DoubleSide, flatShading: true })
        scene.add(new Mesh(wg, wm))
        const baseG = new BufferGeometry()
        baseG.setAttribute('position', new BufferAttribute(new Float32Array([
          wx(f, lon0), BASE, wz(f, lat1), wx(f, lon1), BASE, wz(f, lat1), wx(f, lon1), BASE, wz(f, lat0), wx(f, lon0), BASE, wz(f, lat0)]), 3))
        const baseLine = new LineLoop(baseG, new LineBasicMaterial({ color: '#1F3557', transparent: true, opacity: 0.55 }))
        scene.add(baseLine)
        disposers.push(() => { wg.dispose(); wm.dispose(); baseG.dispose() })

        // ---- country outlines draped on the terrain
        const lineMat = new LineBasicMaterial({ color: '#1F3557', transparent: true, opacity: 0.95 })
        for (const rings of Object.values(map.countries)) for (const r of rings) {
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
          scene.add(new LineLoop(lg, lineMat))
          disposers.push(() => lg.dispose())
        }
        disposers.push(() => lineMat.dispose())

        // ---- earthquake main shocks as columns topped by spheres, at their true locations
        const col = new CylinderGeometry(1, 1, 1, 8).translate(0, 0.5, 0)
        const ball = new SphereGeometry(1, 16, 12)
        const shown = map.quakes.filter((x) => x[1] >= lon0 && x[1] <= lon1 && x[0] >= lat0 && x[0] <= lat1)
        const single = shown.filter((x) => !x[4]), shared = shown.filter((x) => x[4])
        const mk = (list: number[][], color: string, opacity: number, emissive: string) => {
          const cm = new MeshStandardMaterial({ color, roughness: 0.6, transparent: opacity < 1, opacity, emissive, emissiveIntensity: 0.0 })
          const bm = new MeshStandardMaterial({ color, roughness: 0.45, transparent: opacity < 1, opacity: Math.min(1, opacity + 0.25), emissive, emissiveIntensity: 0.0 })
          const cols = new InstancedMesh(col, cm, list.length), balls = new InstancedMesh(ball, bm, list.length)
          cols.frustumCulled = balls.frustumCulled = false
          scene.add(cols, balls)
          disposers.push(() => { cm.dispose(); bm.dispose() })
          return { list, cols, balls, bm, cm }
        }
        const gS = mk(single, '#1F3557', 0.5, '#000000')
        const gB = mk(shared, '#C8553A', 1, '#E0603A')
        disposers.push(() => { col.dispose(); ball.dispose() })
        const M4 = new Matrix4(), Q = new Quaternion(), P = new Vector3(), S = new Vector3()
        const place = (g: ReturnType<typeof mk>, w: number, isShared: boolean) => {
          g.list.forEach((qk, i) => {
            const [la, lo, mag] = qk
            const m = mag - 5.5
            const wt = isShared ? 0.5 + 1.1 * w : 1
            const h = (0.15 + m * 0.2) * wt
            const rc = 0.011 + m * 0.005
            const rb = (0.04 + m * 0.02) * (isShared ? 0.75 + 0.65 * w : 1)
            const x = wx(f, lo), z = wz(f, la), y = Math.max(dem.at(lo, la), 0) * K
            P.set(x, y, z); S.set(rc * (isShared ? 0.8 + 0.5 * w : 1), h, rc * (isShared ? 0.8 + 0.5 * w : 1))
            M4.compose(P, Q, S)
            g.cols.setMatrixAt(i, M4)
            P.set(x, y + h, z); S.set(rb, rb, rb)
            M4.compose(P, Q, S)
            g.balls.setMatrixAt(i, M4)
          })
          g.cols.instanceMatrix.needsUpdate = g.balls.instanceMatrix.needsUpdate = true
        }

        // ---- labels
        const kgL = label('Kyrgyz Republic'), tjL = label('Tajikistan')
        kgL.position.set(wx(f, 75.6), 0.9, wz(f, 42.35))
        tjL.position.set(wx(f, 71.3), 0.95, wz(f, 38.55))
        scene.add(kgL, tjL)

        // ---- light
        scene.add(new HemisphereLight(0xffffff, 0xd8cdb6, 1.25))
        scene.add(new AmbientLight(0xffffff, 0.25))
        const sun = new DirectionalLight(0xfff1dc, 2.6)
        sun.position.set(-6, 5, 5)
        scene.add(sun)

        // ---- weight driven by the earthquake slider
        let dirty = true
        const setWeight = (e: number) => {
          const w = Math.min(Math.max(e / 0.5, 0), 1)
          zoneMat.opacity = 0.2 + 0.8 * Math.pow(w, 0.7)
          place(gS, w, false)
          place(gB, w, true)
          gB.bm.emissiveIntensity = 0.15 + 0.85 * w
          gB.cm.emissiveIntensity = 0.1 + 0.6 * w
          dirty = true
        }
        setWeight(eqRef.current)

        // ---- camera + controls
        const controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.dampingFactor = 0.08
        controls.enablePan = false
        controls.minDistance = 7
        controls.maxDistance = 30
        controls.minPolarAngle = 0.25
        controls.maxPolarAngle = 1.32
        controls.target.set(0, 0.15, 0.1)
        controls.rotateSpeed = 0.6
        const coarse = window.matchMedia('(pointer: coarse)').matches
        if (coarse) {
          // one finger scrolls the page; two fingers orbit/zoom the model
          controls.touches = { ONE: -1 as unknown as TOUCH, TWO: TOUCH.DOLLY_ROTATE }
          renderer.domElement.style.touchAction = 'pan-y'
          disposers.push(() => undefined)
        }
        const rest = { az: 0, pol: 0.98, dist: 18 }
        let inset = 0
        const fit = () => {
          const w = el.clientWidth, h = el.clientHeight
          if (!w || !h) return
          renderer.setSize(w, h, false)
          renderer.domElement.style.width = w + 'px'
          renderer.domElement.style.height = h + 'px'
          camera.aspect = w / h
          const ctl = el.closest('.first-model')?.querySelector('.ctl')
          inset = ctl ? Math.max(0, el.getBoundingClientRect().bottom - ctl.getBoundingClientRect().top) : 0
          const usableH = Math.max(h - inset, h * 0.5)
          const tanV = Math.tan((camera.fov * Math.PI) / 360)
          const wide = (5.68 * (camera.aspect < 1.3 ? 1.0 : 0.86)) / (tanV * camera.aspect)
          const tall = (4.3 * 0.95) / (tanV * (usableH / h))
          rest.dist = Math.max(wide, tall)
          camera.setViewOffset(w, h, 0, inset / 2, w, h)
          camera.updateProjectionMatrix()
          if (!interacted && !intro) place0()
          dirty = true
        }
        const place0 = () => {
          const s = Math.sin(rest.pol) * rest.dist
          camera.position.set(controls.target.x + s * Math.sin(rest.az), controls.target.y + Math.cos(rest.pol) * rest.dist, controls.target.z + s * Math.cos(rest.az))
          camera.lookAt(controls.target)
        }
        let interacted = false
        let intro = false
        let t0 = 0
        controls.addEventListener('start', () => { interacted = true; intro = false })
        controls.addEventListener('change', () => { dirty = true })
        const ro = new ResizeObserver(fit)
        ro.observe(el)
        const ctlEl = el.closest('.first-model')?.querySelector('.ctl')
        if (ctlEl) ro.observe(ctlEl)
        disposers.push(() => ro.disconnect())
        const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
        fit()
        if (!reduce) { intro = true; t0 = performance.now() } else place0()
        controls.update()

        api.current = {
          setWeight,
          reset: () => { intro = true; interacted = false; t0 = performance.now(); controls.target.set(0, 0.15, 0.1) },
        }

        let visible = true
        const io = new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) dirty = true })
        io.observe(el)
        disposers.push(() => io.disconnect())

        const ease = (t: number) => 1 - Math.pow(1 - t, 3)
        const loop = (now: number) => {
          raf = requestAnimationFrame(loop)
          if (!visible || document.hidden) return
          if (intro) {
            const t = Math.min((now - t0) / 2600, 1), e = ease(t)
            const cur = { az: rest.az - 0.5 * (1 - e), pol: rest.pol - 0.38 * (1 - e), dist: rest.dist * (1 + 0.22 * (1 - e)) }
            const s = Math.sin(cur.pol) * cur.dist
            camera.position.set(controls.target.x + s * Math.sin(cur.az), controls.target.y + Math.cos(cur.pol) * cur.dist, controls.target.z + s * Math.cos(cur.az))
            camera.lookAt(controls.target)
            dirty = true
            if (t >= 1) { intro = false; controls.update() }
          } else if (controls.update()) dirty = true
          if (dirty) { renderer.render(scene, camera); dirty = false }
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
  }, [map])

  useEffect(() => { api.current?.setWeight(eq) }, [eq])

  return (
    <div className="terrain" ref={host}>
      <button type="button" className="reset-view" onClick={() => api.current?.reset()}>Reset view</button>
    </div>
  )
}
