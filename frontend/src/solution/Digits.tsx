import { useState } from 'react'
import { sup } from './format'

// Blur-swap digits (motion library: unl-digit-blur-swap, ported to CSS): only the characters that changed animate,
// entering from below when the digit grew and from above when it shrank, so a slider drag reads as cause and effect.
type Slot = { ch: string; old: string | null; dir: 1 | -1; v: number }

function diff(prev: Slot[], next: string): Slot[] {
  const sameShape = prev.length === next.length
  return [...next].map((ch, i) => {
    const p = prev[i]
    if (p && p.ch === ch) return p
    const a = Number(p?.ch), b = Number(ch)
    const dir: 1 | -1 = Number.isFinite(a) && Number.isFinite(b) && b < a ? -1 : 1
    return { ch, old: sameShape && p ? p.ch : null, dir, v: (p?.v ?? 0) + 1 }
  })
}

export function Digits({ text }: { text: string }) {
  const [cur, setCur] = useState(text)
  const [slots, setSlots] = useState<Slot[]>(() => [...text].map((ch) => ({ ch, old: null, dir: 1, v: 0 })))
  if (text !== cur) { setCur(text); setSlots(diff(slots, text)) } // derive from the previous render, StrictMode safe
  return (
    <span className="dg" aria-label={text}>
      {slots.map((s, i) => (
        <span className="dg-c" key={i} aria-hidden>
          {s.v > 0 && s.old !== null && <span key={`o${s.v}`} className={`dg-out ${s.dir > 0 ? 'up' : 'dn'}`}>{s.old}</span>}
          <span key={`n${s.v}`} className={s.v > 0 ? `dg-in ${s.dir > 0 ? 'up' : 'dn'}` : undefined}>{s.ch}</span>
        </span>
      ))}
    </span>
  )
}

/** Same output as prob(), with the mantissa as animated digits. */
export function ProbDigits({ x, digits = 3 }: { x: number | null | undefined; digits?: number }) {
  if (x === null || x === undefined || !Number.isFinite(x)) return <>–</>
  if (x === 0) return <>0</>
  if (x >= 1e-3) return <Digits text={(x * 100).toPrecision(digits) + '%'} />
  const e = Math.floor(Math.log10(x))
  return <><Digits text={(x / Math.pow(10, e)).toPrecision(digits)} />{'×'}10{sup(e)}</>
}
