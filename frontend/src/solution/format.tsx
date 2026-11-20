import type { ReactNode } from 'react'

export const sup = (e: number) => <sup>{String(e).replace('-', '−')}</sup>

/** 3.16×10⁻⁵ as real superscripts; plain percentages above 0.1%. */
export function prob(x: number | null | undefined, digits = 3): ReactNode {
  if (x === null || x === undefined || !Number.isFinite(x)) return '–'
  if (x === 0) return '0'
  if (x >= 1e-3) return (x * 100).toPrecision(digits) + '%'
  const e = Math.floor(Math.log10(x))
  return <>{(x / Math.pow(10, e)).toPrecision(digits)}{'×'}10{sup(e)}</>
}
export function sci(x: number, digits = 2): ReactNode {
  if (!Number.isFinite(x) || x <= 0) return '–'
  const e = Math.floor(Math.log10(x))
  if (e >= -1 && e < 3) return x.toPrecision(digits + 1)
  return <>{(x / Math.pow(10, e)).toPrecision(digits)}{'×'}10{sup(e)}</>
}
export function money(x: number): string {
  return 'US$' + (x >= 100 ? x.toFixed(1) : x.toFixed(2)) + 'm'
}
export function seconds(s: number): string {
  if (!Number.isFinite(s)) return '–'
  if (s < 1) return (s * 1e3).toPrecision(2) + ' ms'
  if (s < 60) return s.toPrecision(2) + ' s'
  if (s < 3600) return (s / 60).toFixed(1) + ' min'
  if (s < 86400) return (s / 3600).toFixed(1) + ' h'
  return (s / 86400).toFixed(1) + ' days'
}
export const pctText = (x: number, d = 1) => (100 * x).toFixed(d) + '%'
