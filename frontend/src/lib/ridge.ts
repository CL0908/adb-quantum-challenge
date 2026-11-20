/** Deterministic ridge silhouettes for the landscape drawings (Tian Shan / Pamir style, no photograph). */
function rng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

function noise1(seed: number, freq: number) {
  const r = rng(seed)
  const n = 512
  const v = Array.from({ length: n }, () => r())
  const sm = (t: number) => t * t * (3 - 2 * t)
  return (x: number) => {
    const t = x * freq
    const i = Math.floor(t)
    const f = sm(t - i)
    const a = v[((i % n) + n) % n]
    const b = v[(((i + 1) % n) + n) % n]
    return a + (b - a) * f
  }
}

/** SVG path of a mountain ridge between x0 and x1; `base` is the mean skyline y, `amp` the relief. */
export function ridgePath(seed: number, x0: number, x1: number, base: number, amp: number, bottom: number, sharp = 0.6): string {
  const n1 = noise1(seed, 0.0042)
  const n2 = noise1(seed + 17, 0.0115)
  const n3 = noise1(seed + 41, 0.034)
  const n4 = noise1(seed + 77, 0.09)
  const ridged = (n: number) => 1 - Math.abs(2 * n - 1)
  const pts: string[] = []
  for (let x = x0; x <= x1; x += 4) {
    const big = Math.pow(ridged(n1(x)), 1.7 + sharp) // tall pointed massifs
    const y = base - amp * (0.58 * big + 0.24 * Math.pow(ridged(n2(x)), 1.4) + 0.1 * ridged(n3(x)) + 0.04 * n4(x) + 0.04 * n1(x + 500))
    pts.push(`${x},${y.toFixed(1)}`)
  }
  return `M${x0},${bottom} L${pts.join(' L')} L${x1},${bottom} Z`
}
