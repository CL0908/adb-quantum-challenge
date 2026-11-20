/* Procedural helpers for the hero centrepiece: an organic outline around the two countries, a marching-squares contour of it,
   an elevation/slope/cavity colour ramp for the top, and a layered-rock texture for the sides. Everything is derived from the real DEM
   (public/data/dem.png) and the country polygons (public/data/map.json); nothing here is data, only shaping. */
import type { Dem } from './lib/dem'
import type { MapData } from './lib/map'

export type RGB = [number, number, number]
export const hex = (s: string): RGB => [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)]
export const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x))
const smooth = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a))
  return t * t * (3 - 2 * t)
}

/* ---------- value noise (optionally periodic) ---------- */
function hash(i: number, j: number, s = 0) {
  let h = (Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(s, 1013904223)) | 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295
}
const wrap = (a: number, p: number) => (p ? ((a % p) + p) % p : a)
export function vnoise(x: number, y: number, px = 0, py = 0, s = 0) {
  const xi = Math.floor(x), yi = Math.floor(y)
  const fx = x - xi, fy = y - yi
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy)
  const a = hash(wrap(xi, px), wrap(yi, py), s), b = hash(wrap(xi + 1, px), wrap(yi, py), s)
  const c = hash(wrap(xi, px), wrap(yi + 1, py), s), d = hash(wrap(xi + 1, px), wrap(yi + 1, py), s)
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}
export function fbm(x: number, y: number, oct = 4, px = 0, py = 0, s = 0) {
  let sum = 0, amp = 0.5, f = 1, norm = 0
  for (let o = 0; o < oct; o++) {
    sum += amp * vnoise(x * f, y * f, px * f, py * f, s + o)
    norm += amp
    amp *= 0.5
    f *= 2
  }
  return sum / norm
}

/* ---------- box blur on a float grid (3 passes ~ gaussian) ---------- */
export function blur(src: Float32Array, w: number, h: number, r: number, passes = 3) {
  let a = Float32Array.from(src)
  const b = new Float32Array(src.length)
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < h; y++) {
      let sum = 0
      const o = y * w
      for (let x = -r; x <= r; x++) sum += a[o + clamp(x, 0, w - 1)]
      for (let x = 0; x < w; x++) {
        b[o + x] = sum / (2 * r + 1)
        sum += a[o + Math.min(w - 1, x + r + 1)] - a[o + Math.max(0, x - r)]
      }
    }
    for (let x = 0; x < w; x++) {
      let sum = 0
      for (let y = -r; y <= r; y++) sum += b[clamp(y, 0, h - 1) * w + x]
      for (let y = 0; y < h; y++) {
        a[y * w + x] = sum / (2 * r + 1)
        sum += b[Math.min(h - 1, y + r + 1) * w + x] - b[Math.max(0, y - r) * w + x]
      }
    }
  }
  return a
}

/* ---------- organic outline field ---------- */
export interface Field {
  nx: number
  ny: number
  v: Float32Array
  T: number
  at: (lon: number, lat: number) => number
}
export const FIELD_T = 0.24

export function buildField(dem: Dem, map: MapData): Field {
  const nx = dem.info.width, ny = dem.info.height
  const [lon0, lat0, lon1, lat1] = dem.info.extent
  const sl = (lon1 - lon0) / (nx - 1), st = (lat1 - lat0) / (ny - 1)
  const c = document.createElement('canvas')
  c.width = nx
  c.height = ny
  const g = c.getContext('2d', { willReadFrequently: true })!
  g.fillStyle = '#000'
  g.fillRect(0, 0, nx, ny)
  g.fillStyle = '#fff'
  for (const iso of ['KG', 'TJ']) {
    g.beginPath()
    for (const r of map.countries[iso] ?? []) {
      r.forEach(([lo, la], i) => { const x = (lo - lon0) / sl, y = (lat1 - la) / st; i ? g.lineTo(x, y) : g.moveTo(x, y) })
      g.closePath()
    }
    g.fill()
  }
  const px = g.getImageData(0, 0, nx, ny).data
  const m = new Float32Array(nx * ny)
  for (let i = 0; i < m.length; i++) m[i] = px[4 * i] / 255
  const b = blur(m, nx, ny, 12, 3)
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const lon = lon0 + i * sl, lat = lat1 - j * st
    const n = (fbm(lon * 0.62, lat * 0.62, 4, 0, 0, 3) - 0.5) * 0.52 + (fbm(lon * 3.1, lat * 3.1, 3, 0, 0, 9) - 0.5) * 0.2
    let val = b[j * nx + i] + n
    const edge = Math.min(i, j, nx - 1 - i, ny - 1 - j)
    if (edge < 4) val = Math.min(val, -1)
    b[j * nx + i] = val
  }
  const at = (lon: number, lat: number) => {
    const fx = clamp((lon - lon0) / sl, 0, nx - 1.001), fy = clamp((lat1 - lat) / st, 0, ny - 1.001)
    const i = Math.floor(fx), j = Math.floor(fy), u = fx - i, w = fy - j, k = j * nx + i
    return (1 - u) * (1 - w) * b[k] + u * (1 - w) * b[k + 1] + (1 - u) * w * b[k + nx] + u * w * b[k + nx + 1]
  }
  return { nx, ny, v: b, T: FIELD_T, at }
}

/** Field as a grey-scale canvas (used as alpha map: the top surface is cut where it falls below the threshold). */
export function fieldCanvas(F: Field) {
  const c = document.createElement('canvas')
  c.width = F.nx
  c.height = F.ny
  const g = c.getContext('2d')!
  const img = g.createImageData(F.nx, F.ny)
  for (let i = 0; i < F.v.length; i++) {
    const a = Math.round(255 * clamp((F.v[i] - F.T) * 9 + 0.5))
    img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = a
    img.data[4 * i + 3] = 255
  }
  g.putImageData(img, 0, 0)
  return c
}

/* ---------- marching squares: closed loops of the field at the threshold, in (lon, lat) ---------- */
export function contours(F: Field, dem: Dem): [number, number][][] {
  const { nx, ny, v, T } = F
  const [lon0, lat0, lon1, lat1] = dem.info.extent
  const sl = (lon1 - lon0) / (nx - 1), st = (lat1 - lat0) / (ny - 1)
  const pts = new Map<number, [number, number]>()
  const adj = new Map<number, number[]>()
  const H = (i: number, j: number) => j * nx + i // edge between (i,j)-(i+1,j)
  const Vv = (i: number, j: number) => nx * ny + j * nx + i // edge between (i,j)-(i,j+1)
  const cross = (id: number, i0: number, j0: number, i1: number, j1: number) => {
    if (!pts.has(id)) {
      const a = v[j0 * nx + i0], b = v[j1 * nx + i1]
      const t = (T - a) / (b - a)
      pts.set(id, [lon0 + (i0 + (i1 - i0) * t) * sl, lat1 - (j0 + (j1 - j0) * t) * st])
    }
  }
  const link = (a: number, b: number) => {
    ;(adj.get(a) ?? adj.set(a, []).get(a)!).push(b)
    ;(adj.get(b) ?? adj.set(b, []).get(b)!).push(a)
  }
  for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const a = v[j * nx + i] >= T, b = v[j * nx + i + 1] >= T, c = v[(j + 1) * nx + i + 1] >= T, d = v[(j + 1) * nx + i] >= T
    if (a === b && b === c && c === d) continue
    const es: number[] = []
    if (a !== b) { cross(H(i, j), i, j, i + 1, j); es.push(H(i, j)) } // top
    if (b !== c) { cross(Vv(i + 1, j), i + 1, j, i + 1, j + 1); es.push(Vv(i + 1, j)) } // right
    if (c !== d) { cross(H(i, j + 1), i, j + 1, i + 1, j + 1); es.push(H(i, j + 1)) } // bottom
    if (d !== a) { cross(Vv(i, j), i, j, i, j + 1); es.push(Vv(i, j)) } // left
    if (es.length === 2) link(es[0], es[1])
    else if (es.length === 4) { link(es[0], es[1]); link(es[2], es[3]) }
  }
  const seen = new Set<number>()
  const loops: [number, number][][] = []
  for (const start of adj.keys()) {
    if (seen.has(start)) continue
    const loop: [number, number][] = []
    let prev = -1, cur = start
    for (let guard = 0; guard < 200000; guard++) {
      seen.add(cur)
      loop.push(pts.get(cur)!)
      const nb = adj.get(cur)!
      const next = nb.find((n) => n !== prev && !seen.has(n)) ?? nb.find((n) => n !== prev)
      if (next === undefined || next === start || seen.has(next)) break
      prev = cur
      cur = next
    }
    if (loop.length > 30) loops.push(loop)
  }
  return loops
}

const area = (p: [number, number][]) => {
  let s = 0
  for (let i = 0; i < p.length; i++) { const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length]; s += x0 * y1 - x1 * y0 }
  return s / 2
}

/** Smooths a closed loop and resamples it to n evenly spaced points (counter-clockwise in lon/lat). */
export function resample(loop: [number, number][], n: number): [number, number][] {
  let p = loop
  if (area(p) < 0) p = [...p].reverse()
  for (let pass = 0; pass < 4; pass++) {
    p = p.map((q, i) => { const a = p[(i + p.length - 1) % p.length], b = p[(i + 1) % p.length]; return [(a[0] + 2 * q[0] + b[0]) / 4, (a[1] + 2 * q[1] + b[1]) / 4] as [number, number] })
  }
  const cum = [0]
  for (let i = 0; i < p.length; i++) { const a = p[i], b = p[(i + 1) % p.length]; cum.push(cum[i] + Math.hypot(a[0] - b[0], a[1] - b[1])) }
  const total = cum[cum.length - 1]
  const out: [number, number][] = []
  let k = 0
  for (let s = 0; s < n; s++) {
    const d = (s / n) * total
    while (cum[k + 1] < d) k++
    const t = (d - cum[k]) / (cum[k + 1] - cum[k] || 1)
    const a = p[k], b = p[(k + 1) % p.length]
    out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
  }
  return out
}

/* ---------- natural colouring of the top ---------- */
const STOPS: [number, RGB][] = [
  [0, hex('#93A064')], [1300, hex('#A3A872')], [2100, hex('#B7AF7F')], [3000, hex('#BCA97F')],
  [3800, hex('#A99F8F')], [4600, hex('#8E8A86')], [6000, hex('#7F7C79')],
]
function ramp(m: number): RGB {
  if (m <= STOPS[0][0]) return STOPS[0][1]
  for (let i = 1; i < STOPS.length; i++) {
    if (m <= STOPS[i][0]) {
      const [a, ca] = STOPS[i - 1], [b, cb] = STOPS[i]
      return mix(ca, cb, (m - a) / (b - a))
    }
  }
  return STOPS[STOPS.length - 1][1]
}
const ROCK = hex('#8D877E'), SNOW = hex('#F6F7F8')

/** Colour canvas at the DEM's own resolution: hypsometric ramp, steep slopes turn to rock, snow on the highest and flattest, cavity darkening. */
export function colourCanvas(dem: Dem, F: Field) {
  const { width: nx, height: ny } = dem.info
  const [lon0, lat0, lon1, lat1] = dem.info.extent
  const sl = (lon1 - lon0) / (nx - 1), st = (lat1 - lat0) / (ny - 1)
  const mx = 111320 * Math.cos((((lat0 + lat1) / 2) * Math.PI) / 180) * sl, my = 111320 * st
  const hb = blur(dem.h, nx, ny, 7, 3)
  const c = document.createElement('canvas')
  c.width = nx
  c.height = ny
  const g = c.getContext('2d')!
  const img = g.createImageData(nx, ny)
  const H = (i: number, j: number) => dem.h[clamp(j, 0, ny - 1) * nx + clamp(i, 0, nx - 1)]
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const k = j * nx + i
    const h = Math.max(dem.h[k], 0)
    const gx = (H(i + 1, j) - H(i - 1, j)) / (2 * mx), gy = (H(i, j + 1) - H(i, j - 1)) / (2 * my)
    const slope = Math.hypot(gx, gy)
    const lon = lon0 + i * sl, lat = lat1 - j * st
    const n = fbm(lon * 9, lat * 9, 3, 0, 0, 21) - 0.5
    let col = ramp(h + n * 260)
    col = mix(col, ROCK, 0.72 * smooth(0.16, 0.5, slope + n * 0.08))
    const snow = smooth(4400, 5300, h + n * 420) * (1 - 0.6 * smooth(0.3, 0.8, slope))
    col = mix(col, SNOW, snow)
    // soft key-light shading baked from the slope direction (north-west light) gives the relief a sculpted feel even before real lighting
    const shade = clamp(1 + (-gx * 0.55 + gy * 0.45) * 0.9, 0.78, 1.18)
    const cav = clamp(1 + (h - hb[k]) / 1400, 0.8, 1.05)
    const rim = 0.9 + 0.1 * smooth(F.T, F.T + 0.12, F.v[k])
    const f = shade * cav * rim * (1 + n * 0.1)
    img.data[4 * k] = clamp(col[0] * f, 0, 255)
    img.data[4 * k + 1] = clamp(col[1] * f, 0, 255)
    img.data[4 * k + 2] = clamp(col[2] * f, 0, 255)
    img.data[4 * k + 3] = 255
  }
  g.putImageData(img, 0, 0)
  return c
}

/* ---------- layered rock for the sides ---------- */
const STRATA = ['#D6CBB4', '#BDB19B', '#E0D5BC', '#A89D8B', '#C9BC9F', '#938A7A', '#C2B6A3', '#DDD1B6', '#B3A58C'].map(hex)
/** Horizontal rock layers of uneven thickness with wavy contacts, dark seams and grain. Periodic in u so it wraps around the island. */
export function strataCanvas(W = 1024, Hh = 1024) {
  const c = document.createElement('canvas')
  c.width = W
  c.height = Hh
  const g = c.getContext('2d')!
  const img = g.createImageData(W, Hh)
  const PU = 6
  for (let y = 0; y < Hh; y++) {
    for (let x = 0; x < W; x++) {
      const u = x / W
      const warp = (fbm(u * PU, y / 90, 3, PU, 0, 5) - 0.5) * 22 + Math.sin(u * Math.PI * 2 * 3 + y * 0.01) * 3
      const yy = y + warp
      // layer index from cumulative uneven thickness
      const L = yy / 15
      const li = Math.floor(L)
      const th = 0.55 + hash(li, 3, 1) * 1.1
      const band = Math.floor(L / th + hash(li, 7, 2) * 0.3)
      const base = STRATA[Math.abs(Math.floor(band * 1.7 + hash(band, 1, 3) * 5)) % STRATA.length]
      const within = L - Math.floor(L)
      const seam = 1 - 0.2 * smooth(0.82, 1, within) * (hash(li, 5, 4) > 0.45 ? 1 : 0.4)
      const grain = (hash(x, y, 8) - 0.5) * 0.07 + (vnoise(u * 90, y / 3.5, 90, 0, 6) - 0.5) * 0.1
      const f = seam * (1 + grain)
      const k = 4 * (y * W + x)
      img.data[k] = clamp(base[0] * f, 0, 255)
      img.data[k + 1] = clamp(base[1] * f, 0, 255)
      img.data[k + 2] = clamp(base[2] * f, 0, 255)
      img.data[k + 3] = 255
    }
  }
  g.putImageData(img, 0, 0)
  return c
}
