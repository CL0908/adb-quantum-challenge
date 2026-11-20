import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

export type Values = Record<string, string>

/** Fetches a JSON file from data/ once; null until it arrives (or if it fails). */
export function useJSON<T>(file: string): T | null {
  const [d, setD] = useState<T | null>(null)
  useEffect(() => {
    let live = true
    fetch(`data/${file}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((j: T) => live && setD(j))
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [file])
  return d
}

const SUP: Record<string, string> = { '⁻': '−', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', 'ⁿ': 'n' }

/** Turns unicode superscripts in a value string (3.2×10⁻⁵) into real <sup> elements. */
export function Fancy({ s }: { s: string }) {
  const parts = s.split(/([⁻⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]+)/)
  return (
    <>
      {parts.map((p, i) =>
        i % 2 ? <sup key={i}>{[...p].map((c) => SUP[c] ?? c).join('')}</sup> : <span key={i}>{p}</span>,
      )}
    </>
  )
}

const Ctx = createContext<Values>({})
export function ValuesProvider({ children }: { children: ReactNode }) {
  const v = useJSON<Values>('values.json')
  return <Ctx.Provider value={v ?? {}}>{children}</Ctx.Provider>
}
export const useValues = () => useContext(Ctx)

/** A number from values.json (never typed by hand). */
export function V({ k }: { k: string }) {
  const v = useValues()
  return v[k] ? <Fancy s={v[k]} /> : <span className="pending">…</span>
}
