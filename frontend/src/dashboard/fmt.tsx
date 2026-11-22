import type { ReactNode } from 'react'

/** 3.2 x 10^-5 as a node (real superscript). */
export function sci(x: number, digits = 2): ReactNode {
  if (!Number.isFinite(x)) return 'n/a'
  if (x === 0) return '0'
  const [m, e] = x.toExponential(digits - 1).split('e')
  return (
    <>
      {m}×10<sup>{e.replace('+', '').replace('-', '−')}</sup>
    </>
  )
}
/** Same as sci() but for use inside <svg><text>, where <sup> is not allowed. */
export function sciSvg(x: number, digits = 2): ReactNode {
  if (!Number.isFinite(x)) return 'n/a'
  if (x === 0) return '0'
  const [m, e] = x.toExponential(digits - 1).split('e')
  return (
    <>
      {m}×10<tspan dy={-4} fontSize="10">{e.replace('+', '').replace('-', '−')}</tspan>
    </>
  )
}
export function sciText(x: number, digits = 2): string {
  const [m, e] = x.toExponential(digits - 1).split('e')
  return `${m}×10^${e.replace('+', '').replace('-', '−')}`
}

/** A probability for a banker: percent when it is at least 1e-3, scientific below. */
export function prob(x: number): ReactNode {
  if (!Number.isFinite(x)) return 'n/a'
  if (x >= 0.1) return `${(x * 100).toFixed(1)}%`
  if (x >= 0.001) return `${(x * 100).toFixed(2)}%`
  return sci(x)
}
export function probText(x: number): string {
  if (x >= 0.1) return `${(x * 100).toFixed(1)}%`
  if (x >= 0.001) return `${(x * 100).toFixed(2)}%`
  return sciText(x)
}

export function usd(x: number): string {
  if (!Number.isFinite(x)) return 'n/a'
  const a = Math.abs(x)
  if (a >= 1e6) return `US$${(x / 1e6).toFixed(a >= 1e8 ? 0 : 2)}m`
  if (a >= 1e3) return `US$${(x / 1e3).toFixed(a >= 1e5 ? 0 : 1)}k`
  return `US$${x.toFixed(2)}`
}

export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`
export function relErr(x: number | null): string {
  if (x === null || !Number.isFinite(x)) return 'no run'
  const v = x * 100
  return `${v < 1 ? v.toFixed(2) : v.toFixed(1)}% RMSE` // relative root-mean-square error over repeated runs, not an interval
}
export function delta(a: number, b: number): { text: string; sign: number } | null {
  if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return null
  const d = (a / b - 1) * 100
  if (Math.abs(d) < 0.05) return { text: 'same as brief', sign: 0 }
  return { text: `${d > 0 ? '+' : '−'}${Math.abs(d).toFixed(Math.abs(d) < 10 ? 1 : 0)}%`, sign: d > 0 ? 1 : -1 }
}
export const TARGET_LABEL: Record<string, string> = {
  expected_loss: 'Expected loss per term',
  joint_attachment: 'Joint attachment',
  exceedance: 'Pr(pool loss > threshold)',
  stop_loss: 'Stop-loss above threshold',
  joint_exhaustion: 'Joint exhaustion',
}
export const TARGET_HINT: Record<string, string> = {
  expected_loss: 'mean principal loss over the 3-year term',
  joint_attachment: 'both bonds take a loss in the same term',
  exceedance: 'pool loss exceeds the shared limit',
  stop_loss: 'expected claims above the shared limit',
  joint_exhaustion: 'both bonds fully exhausted',
}
export function moduleCmd(script: string): string {
  return `python -m ${script.replace(/\.py$/, '').split('/').join('.')}`
}
