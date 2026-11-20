export interface Point { eq: number; rain: number; m: number; joint_exhaustion: number; exceed: number; delivered: number }
export interface Grid {
  eq_axis: number[]
  rain_axis: number[]
  values: { joint_exhaustion: number[][]; exceed: number[][]; delivered: number[][] }
  brief: Point
  data: Point
}
export interface Answers { jointExhaustion: number; exceed: number; delivered: number }

function bilinear(g: number[][], ea: number[], ra: number[], eq: number, rain: number, log: boolean): number {
  const f = (x: number) => (log ? Math.log(x) : x)
  const loc = (a: number[], x: number) => {
    const t = Math.min(Math.max(x, a[0]), a[a.length - 1])
    let i = Math.min(Math.floor((t - a[0]) / (a[1] - a[0]) + 1e-9), a.length - 2)
    if (i < 0) i = 0
    return [i, (t - a[i]) / (a[i + 1] - a[i])] as const
  }
  const [i, u] = loc(ea, eq)
  const [j, w] = loc(ra, rain)
  const v = (1 - u) * (1 - w) * f(g[i][j]) + u * (1 - w) * f(g[i + 1][j]) + (1 - u) * w * f(g[i][j + 1]) + u * w * f(g[i + 1][j + 1])
  return log ? Math.exp(v) : v
}

/** Exact answers at (eq, rain): the stored exact point when the sliders sit on the brief or the data, else the
 *  exact 61 x 31 grid, interpolated (bilinear; log scale for the two probabilities). */
export function answers(g: Grid, eq: number, rain: number): Answers {
  for (const p of [g.brief, g.data]) {
    if (Math.abs(eq - p.eq) < 1e-9 && Math.abs(rain - p.rain) < 1e-9)
      return { jointExhaustion: p.joint_exhaustion, exceed: p.exceed, delivered: p.delivered }
  }
  const { eq_axis: ea, rain_axis: ra, values: v } = g
  return {
    jointExhaustion: bilinear(v.joint_exhaustion, ea, ra, eq, rain, true),
    exceed: bilinear(v.exceed, ea, ra, eq, rain, true),
    delivered: bilinear(v.delivered, ea, ra, eq, rain, false),
  }
}

export const oneIn = (p: number): string => {
  const n = 1 / p
  const mag = Math.pow(10, Math.floor(Math.log10(n)) - 2) // three significant figures
  return `1 in ${(Math.round(n / mag) * mag).toLocaleString('en-US')}`
}
export const pct = (x: number, d = 1) => `${(100 * x).toFixed(d)}%`
