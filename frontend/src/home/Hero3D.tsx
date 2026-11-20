import { useEffect, useRef } from 'react'
import {
  AdditiveBlending, AmbientLight, BufferAttribute, BufferGeometry, CanvasTexture, CylinderGeometry, DirectionalLight, DoubleSide,
  Group, HemisphereLight, InstancedMesh, LineBasicMaterial, LineSegments, LinearFilter, Matrix4, Mesh, MeshBasicMaterial, MeshStandardMaterial,
  PerspectiveCamera, Quaternion, RepeatWrapping, Scene, SphereGeometry, SRGBColorSpace, Sprite, SpriteMaterial, Vector3, WebGLRenderer,
} from 'three'
import { loadDem } from './lib/dem'
import type { MapData } from './lib/map'
import { blur, buildField, colourCanvas, contours, fbm, fieldCanvas, resample, strataCanvas, vnoise } from './island'

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
export default function Hero3D({ map, still, onReady, onFail }: { map: MapData; still?: Pose; onReady?: () => void; onFail?: () => void }) {
  const host = useRef<HTMLDivElement>(null)
  const cb = useRef({ onReady, onFail })
  cb.current = { onReady, onFail }

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
        renderer.domElement.className = 'hero-canvas'
        el.prepend(renderer.domElement)
        disposers.push(() => { renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove() })

        const scene = new Scene()
        const camera = new PerspectiveCamera(17, HERO_ASPECT, 0.5, 200)
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
        const base = colourCanvas(dem, F)
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
        bg.globalAlpha = 0.55
        bg.fillStyle = bg.createPattern(gc, 'repeat')!
        bg.fillRect(0, 0, TW, TH)
        const ground = new CanvasTexture(big)
        ground.colorSpace = SRGBColorSpace
        ground.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy())
        ground.minFilter = LinearFilter
        const alphaTex = new CanvasTexture(fieldCanvas(F))
        alphaTex.minFilter = LinearFilter
        const mat = new MeshStandardMaterial({ map: ground, alphaMap: alphaTex, alphaTest: 0.5, roughness: 0.96, metalness: 0 })
        root.add(new Mesh(geo, mat))
        disposers.push(() => { geo.dispose(); mat.dispose(); ground.dispose(); alphaTex.dispose() })

        const zoneTex = new CanvasTexture(paintZone(map, lon0, lat0, lon1, lat1, TW, TH))
        zoneTex.colorSpace = SRGBColorSpace
        const zoneMat = new MeshBasicMaterial({ map: zoneTex, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, opacity: 0.9 })
        root.add(new Mesh(geo, zoneMat))
        disposers.push(() => { zoneMat.dispose(); zoneTex.dispose() })

        /* ---- the carved underside: rings that taper inward, with layered rock texture ---- */
        const N = 720, R = 20, DEPTH = 2.5
        const strata = new CanvasTexture(strataCanvas(1024, 1024))
        strata.colorSpace = SRGBColorSpace
        strata.wrapS = strata.wrapT = RepeatWrapping
        strata.anisotropy = 8
        const rockMat = new MeshStandardMaterial({ map: strata, vertexColors: true, roughness: 0.92, metalness: 0, side: DoubleSide })
        let cx = 0, cz = 0, minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9
        const U_REP = 6, V_SCALE = 0.62
        keep.forEach((loop, li) => {
          const P = resample(loop, N)
          const xs = P.map(([lo]) => wx(lo)), zs = P.map(([, la]) => wz(la))
          const mx = xs.reduce((a, b) => a + b, 0) / N, mz = zs.reduce((a, b) => a + b, 0) / N
          if (li === 0) { cx = mx; cz = mz }
          xs.forEach((x) => { minX = Math.min(minX, x); maxX = Math.max(maxX, x) })
          zs.forEach((z) => { minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z) })
          const top = P.map(([lo, la]) => Math.max(dem.at(lo, la), 0) * K + 0.012)
          const dl = P.map((_, i) => {
            const th = (i / N) * Math.PI * 2
            return 0.7 + 0.6 * fbm(Math.cos(th) * 1.3 + 3 + li * 7, Math.sin(th) * 1.3 + 3, 3)
          })
          const dMean = DEPTH * (li === 0 ? 1 : 0.55)
          const cols = N + 1
          const p3 = new Float32Array((R + 1) * cols * 3), uv2 = new Float32Array((R + 1) * cols * 2), vc = new Float32Array((R + 1) * cols * 3)
          for (let r = 0; r <= R; r++) {
            const t = r / R
            const sc = Math.pow(Math.max(0, 1 - Math.pow(t, 2.1)), 0.5)
            for (let i = 0; i <= cols - 1; i++) {
              const ii = i % N
              const th = (ii / N) * Math.PI * 2
              const j = 1 + 0.07 * (vnoise(Math.cos(th) * 2.4 + r * 0.7, Math.sin(th) * 2.4 + 5, 0, 0, 11 + li) - 0.5) * 2 * Math.sin(Math.PI * Math.min(t * 1.15, 1))
              const dd = dMean * (1 + (dl[ii] - 1) * 1.1 * (1 - t))
              const x = mx + (xs[ii] - mx) * sc * j, z = mz + (zs[ii] - mz) * sc * j
              const y = top[ii] + (-dd - top[ii]) * (t < 0.04 ? t * 0.5 : t)
              const o = r * cols + i
              p3[3 * o] = x; p3[3 * o + 1] = y; p3[3 * o + 2] = z
              uv2[2 * o] = (i / N) * U_REP
              uv2[2 * o + 1] = y * V_SCALE
              const dk = 1 - 0.42 * Math.pow(t, 1.25)
              vc[3 * o] = dk; vc[3 * o + 1] = dk * 0.99; vc[3 * o + 2] = dk * 0.97
            }
          }
          const ix: number[] = []
          for (let r = 0; r < R; r++) for (let i = 0; i < N; i++) {
            const a = r * cols + i, b = a + 1, c = a + cols, d = c + 1
            ix.push(a, c, b, b, c, d)
          }
          const wg = new BufferGeometry()
          wg.setAttribute('position', new BufferAttribute(p3, 3))
          wg.setAttribute('uv', new BufferAttribute(uv2, 2))
          wg.setAttribute('color', new BufferAttribute(vc, 3))
          wg.setIndex(ix)
          wg.computeVertexNormals()
          root.add(new Mesh(wg, rockMat))
          disposers.push(() => wg.dispose())
        })
        disposers.push(() => { rockMat.dispose(); strata.dispose() })

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
        const mk = (list: number[][], color: string, opacity: number, emissive: string, ei: number) => {
          const cm = new MeshStandardMaterial({ color, roughness: 0.55, transparent: opacity < 1, opacity, emissive, emissiveIntensity: ei })
          const bm = new MeshStandardMaterial({ color, roughness: 0.35, transparent: opacity < 1, opacity: Math.min(1, opacity + 0.25), emissive, emissiveIntensity: ei })
          const cols = new InstancedMesh(col, cm, list.length), balls = new InstancedMesh(ball, bm, list.length)
          cols.frustumCulled = balls.frustumCulled = false
          list.forEach(([la, lo, mag], i) => {
            const m = mag - 5.5
            const h = 0.13 + m * 0.2
            const rc = 0.0055 + m * 0.0024, rb = 0.02 + m * 0.011
            const x = wx(lo), z = wz(la), y = Math.max(dem.at(lo, la), 0) * K
            P.set(x, y, z); S.set(rc, h, rc); M4.compose(P, Q, S); cols.setMatrixAt(i, M4)
            P.set(x, y + h, z); S.set(rb, rb, rb); M4.compose(P, Q, S); balls.setMatrixAt(i, M4)
          })
          root.add(cols, balls)
          disposers.push(() => { cm.dispose(); bm.dispose() })
        }
        mk(solo, '#2A3342', 0.72, '#000000', 0)
        mk(shared, '#C8553A', 1, '#E0603A', 0.55)
        const haloTex = halo()
        const hm = new SpriteMaterial({ map: haloTex, blending: AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.8 })
        shared.forEach(([la, lo, mag]) => {
          const m = mag - 5.5, h = 0.13 + m * 0.2
          const s = new Sprite(hm)
          s.position.set(wx(lo), Math.max(dem.at(lo, la), 0) * K + h, wz(la))
          s.scale.setScalar(0.34 + m * 0.12)
          root.add(s)
        })
        disposers.push(() => { col.dispose(); ball.dispose(); hm.dispose(); haloTex.dispose() })

        root.position.set(-cx, 0.3, -cz)

        /* ---- studio light: warm key, cool fill, rim, soft ambient ---- */
        scene.add(new HemisphereLight(0xfaf7f2, 0xcfc6b4, 1.05))
        scene.add(new AmbientLight(0xffffff, 0.2))
        const key = new DirectionalLight(0xfff0dd, 2.5)
        key.position.set(-7, 9, 6)
        scene.add(key)
        const rimL = new DirectionalLight(0xdfe9ff, 0.9)
        rimL.position.set(6, 3, -7)
        scene.add(rimL)
        const fill = new DirectionalLight(0xffffff, 0.35)
        fill.position.set(5, 2, 7)
        scene.add(fill)

        /* ---- camera ---- */
        const bw = maxX - minX, bd = maxZ - minZ
        const pol = still?.pol ?? 1.1
        const yawMax = still ? 0 : 0.36
        const yaw = Math.max(yawMax, Math.abs(still?.az ?? 0))
        const need = bw * Math.cos(yaw) + bd * Math.sin(yaw) * 0.8
        const tanH = Math.tan((camera.fov * Math.PI) / 360) * HERO_ASPECT
        const dist = (need / 2 / tanH) * 1.16
        const aim = new Vector3(0, -0.3, 0.1)
        const place = () => {
          const s = Math.sin(pol) * dist
          camera.position.set(aim.x, aim.y + Math.cos(pol) * dist, aim.z + s)
          camera.lookAt(aim)
        }
        place()
        const fit = () => {
          const w = el.clientWidth, h = el.clientHeight
          if (!w || !h) return
          renderer.setSize(w, h, false)
          renderer.domElement.style.width = w + 'px'
          renderer.domElement.style.height = h + 'px'
          camera.aspect = w / h
          const tH = Math.tan((camera.fov * Math.PI) / 360) * camera.aspect
          const d = (need / 2 / tH) * 1.16
          const s = Math.sin(pol) * d
          camera.position.set(aim.x, aim.y + Math.cos(pol) * d, aim.z + s)
          camera.lookAt(aim)
          camera.updateProjectionMatrix()
          renderer.render(scene, camera)
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
        const frame = el.parentElement
        const loop = (now: number) => {
          raf = requestAnimationFrame(loop)
          if (!visible || document.hidden) return
          const t = (now - t0) / 1000
          const bob = Math.sin((t * Math.PI * 2) / 7.5)
          pivot.rotation.y = Math.sin((t * Math.PI * 2) / 46) * yawMax
          pivot.position.y = bob * 0.07
          zoneMat.opacity = 0.78 + 0.22 * Math.sin((t * Math.PI * 2) / 5)
          hm.opacity = 0.65 + 0.2 * Math.sin((t * Math.PI * 2) / 5 + 0.6)
          frame?.style.setProperty('--bob', bob.toFixed(3))
          renderer.render(scene, camera)
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

