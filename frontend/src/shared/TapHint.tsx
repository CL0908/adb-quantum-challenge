import './taphint.css'

// pointer glyph after lucide "pointer" (ISC); drawn twice: a wide white fill-and-stroke under the ink outline
const HAND = 'M14 10V9a2 2 0 0 0-2-2a2 2 0 0 0-2 2v1M18 11v-1a2 2 0 0 0-2-2a2 2 0 0 0-2 2M10 9.5V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v10M18 11a2 2 0 1 1 4 0v3a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15'

/** A hand that taps the spot it is placed on (the wrapper's 0,0 is the fingertip), with a ripple at each press.
 *  Decorative: the control it points at keeps its own label. With reduced motion the hand stays still. */
export function TapHint({ className = '' }: { className?: string }) {
  return (
    <span className={`tap-hint ${className}`} aria-hidden>
      <i className="th-ripple" />
      <svg className="th-hand" viewBox="0 0 24 24" width="34" height="34">
        <path className="th-bg" d={HAND} />
        <path d={HAND} />
      </svg>
    </span>
  )
}

/** The same hand inside an SVG drawing: (x, y) is the fingertip in the drawing's units, size is the hand's height there. */
export function SvgTapHint({ x, y, size }: { x: number; y: number; size: number }) {
  return (
    <g className="tap-hint-svg" transform={`translate(${x} ${y})`} aria-hidden>
      <circle className="th-ripple-svg" r={size * 0.4} />
      <g className="th-hand-g">
        <svg x={(-size * 8) / 24} y={(-size * 2) / 24} width={size} height={size} viewBox="0 0 24 24" overflow="visible">
          <path className="th-bg" d={HAND} />
          <path d={HAND} />
        </svg>
      </g>
    </g>
  )
}
