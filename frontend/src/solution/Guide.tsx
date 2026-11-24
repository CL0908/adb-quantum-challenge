import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { TapHint } from '../shared/TapHint'

// Three-step guide for the Solution page. Motion from the library, ported to CSS:
//   status-dot-pulse  -> the pulsing ring around the control the current step asks for, plus a tapping hand on it
//   bar-fill          -> the step bar: the link fills first, then the next dot lights (only when moving forward)
//   step-slide-wait   -> step text enters from the right on Next, from the left on Back
export type GuideStep = {
  target: string // CSS selector of the control to press or drag
  point?: (el: HTMLElement, r: DOMRect) => { x: number; y: number } // where the fingertip goes; centre by default
  title: string; body: ReactNode; done: boolean; doneText: string
  action?: { label: string; run: () => void }
}

const KEY = 'sol-guide-seen'
export const guideSeen = () => { try { return localStorage.getItem(KEY) === '1' } catch { return true } }
const markSeen = () => { try { localStorage.setItem(KEY, '1') } catch { /* private mode: the guide just shows again */ } }

type Box = { x: number; y: number; w: number; h: number; px: number; py: number }

export default function Guide({ steps, step, setStep, onClose }: { steps: GuideStep[]; step: number; setStep: (n: number) => void; onClose: () => void }) {
  const [dir, setDir] = useState<1 | -1>(1)
  const [box, setBox] = useState<Box | null>(null)
  const s = steps[step]
  const last = step === steps.length - 1
  const go = (n: number) => { setDir(n > step ? 1 : -1); setStep(n) }
  const close = () => { markSeen(); onClose() }

  // bring the step's control into view when it changes (the right panel scrolls on laptops)
  useEffect(() => {
    document.querySelector<HTMLElement>(s.target)?.scrollIntoView({ block: 'center', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' })
  }, [s.target])

  // follow the target every frame: panels scroll, stages expand, sliders move, the window resizes
  const pointRef = useRef(s.point)
  pointRef.current = s.point
  useLayoutEffect(() => {
    let prev = '', raf = 0
    const tick = () => {
      const el = document.querySelector<HTMLElement>(s.target)
      const r = el?.getBoundingClientRect()
      let next: Box | null = null
      if (el && r && r.width > 0) {
        const p = pointRef.current?.(el, r) ?? { x: r.left + r.width / 2, y: r.top + r.height / 2 }
        next = { x: r.left, y: r.top, w: r.width, h: r.height, px: p.x, py: p.y }
      }
      const k = next ? Object.values(next).map(Math.round).join(',') : ''
      if (k !== prev) { prev = k; setBox(next) }
      raf = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(raf)
  }, [s.target])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  return (
    <>
      {box && (
        <>
          <div className={`gd-ring${s.done ? ' ok' : ''}`} aria-hidden style={{ left: box.x - 6, top: box.y - 6, width: box.w + 12, height: box.h + 12 }} />
          {!s.done && <div className="gd-hand" style={{ left: box.px, top: box.py }}><TapHint /></div>}
        </>
      )}
      <section className="jt-panel solid gd-card" role="dialog" aria-modal="false" aria-label="Guide to this page">
        <div className="gd-head">
          <ol className="gd-steps" aria-label={`Step ${step + 1} of ${steps.length}`}>
            {steps.map((x, i) => (
              <li key={i} className={`${i < step ? 'past' : ''}${i === step ? ' now' : ''}${dir > 0 ? ' adv' : ''}`}>
                {i > 0 && <i className="gd-link" />}
                <button type="button" onClick={() => go(i)} aria-current={i === step ? 'step' : undefined} aria-label={`Step ${i + 1}: ${x.title}`}>{x.done && i !== step ? '✓' : i + 1}</button>
              </li>
            ))}
          </ol>
          <button type="button" className="gd-x" onClick={close}>Skip guide</button>
        </div>
        <div className="gd-view">
          <div key={step} className={`gd-body ${dir > 0 ? 'fwd' : 'back'}`}>
            <div className="lbl teal">Step {step + 1} of {steps.length}</div>
            <h2>{s.title}</h2>
            <p>{s.body}</p>
            <div className={`gd-state${s.done ? ' ok' : ''}`} aria-live="polite">{s.done ? <>✓ {s.doneText}</> : <>Follow the hand on the page</>}</div>
          </div>
        </div>
        <div className="gd-foot">
          <button type="button" className="jt-btn ghost" onClick={() => go(step - 1)} disabled={step === 0}>Back</button>
          {s.action && !s.done && <button type="button" className="jt-btn ghost" onClick={s.action.run}>{s.action.label}</button>}
          <button type="button" className={`jt-btn solid${s.done ? ' gd-ready' : ''}`} onClick={() => (last ? close() : go(step + 1))}>{last ? 'Finish' : 'Next'}</button>
        </div>
      </section>
    </>
  )
}
