/** Real elevation for the 3D model: public/data/dem.png (16-bit metres in R,G; see src/viz/web_precompute.py) + dem.json. */
export interface DemInfo {
  extent: [number, number, number, number] // lon0, lat0, lon1, lat1
  width: number
  height: number
  offset_m: number
  min_m: number
  max_m: number
  source: string
  licence: string
}
export interface Dem {
  info: DemInfo
  /** metres, row 0 = north edge */
  h: Float32Array
  /** metres at (lon, lat), bilinear */
  at: (lon: number, lat: number) => number
}

export async function loadDem(): Promise<Dem> {
  const info: DemInfo = await fetch('data/dem.json').then((r) => (r.ok ? r.json() : Promise.reject(new Error('dem.json ' + r.status))))
  const blob = await fetch('data/dem.png').then((r) => (r.ok ? r.blob() : Promise.reject(new Error('dem.png ' + r.status))))
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
  if (!(Math.max(...h.subarray(0, 5000)) < 9000)) throw new Error('dem decode check failed')
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

/** True when a WebGL context can be created (checked before the three.js chunk is downloaded). */
export function webglAvailable(): boolean {
  try {
    const c = document.createElement('canvas')
    const gl = c.getContext('webgl2') || c.getContext('webgl')
    if (!gl) return false
    ;(gl as WebGLRenderingContext).getExtension('WEBGL_lose_context')?.loseContext()
    return true
  } catch {
    return false
  }
}

/** Just dem.json (a few hundred bytes): tells the page whether terrain exists before anything heavy loads. */
export async function loadDemInfo(): Promise<DemInfo> {
  if (!webglAvailable()) throw new Error('no WebGL')
  const info: DemInfo = await fetch('data/dem.json').then((r) => (r.ok ? r.json() : Promise.reject(new Error('dem.json ' + r.status))))
  const png = await fetch('data/dem.png', { method: 'HEAD' })
  if (!png.ok) throw new Error('dem.png ' + png.status)
  return info
}
