import { useState } from 'react'
import './theme.css'

export type SitePage = 'home' | 'solution' | 'dashboard'

const LINKS: { id: SitePage | 'documents'; label: string; href: string }[] = [
  { id: 'home', label: 'Home', href: '/' },
  { id: 'solution', label: 'Solution', href: '/solution/' },
  { id: 'dashboard', label: 'Dashboard', href: '/dashboard/' },
  { id: 'documents', label: 'Documents', href: '/#documents' },
]
export const PROPOSAL_PDF = '/RDPO_Proposal_Short.pdf'

/** The one top navigation for every page. Sets data-theme on <html> so tokens in theme.css apply. */
export function SiteNav({ theme, current }: { theme: 'light' | 'dark'; current: SitePage }) {
  const [open, setOpen] = useState(false)
  if (typeof document !== 'undefined') document.documentElement.dataset.theme = theme
  const cur = (id: string) => (id === current ? ({ 'aria-current': 'page' } as const) : {})
  return (
    <header className="jt-nav-wrap">
      <nav className="jt-nav" aria-label="Main">
        <a className="jt-brand jt-pillbox" href="/" aria-label="Joint-Tail, home">
          <svg viewBox="0 0 28 20" width="28" height="20" aria-hidden="true">
            <circle cx="9" cy="10" r="8" fill="var(--navy)" stroke="var(--c3)" strokeWidth={theme === 'dark' ? 1 : 0} opacity="0.95" />
            <circle cx="19" cy="10" r="8" fill="var(--terra)" opacity="0.92" />
            <path d="M14 3.6a8 8 0 0 1 0 12.8 8 8 0 0 1 0-12.8z" fill="var(--ink)" />
          </svg>
          Joint-Tail
        </a>
        <div className="jt-links jt-pillbox">
          {LINKS.map((l) => <a key={l.id} href={l.href} {...cur(l.id)}>{l.label}</a>)}
        </div>
        <div className="jt-right">
          <a className="jt-pdf" href={PROPOSAL_PDF}>Proposal (PDF)</a>
          <button type="button" className="jt-menu-btn" aria-expanded={open} aria-controls="jt-mnav" onClick={() => setOpen(!open)}>
            <span className="jt-sr">Menu</span><i /><i />
          </button>
        </div>
      </nav>
      {open && (
        <div className="jt-mnav" id="jt-mnav">
          {LINKS.map((l) => <a key={l.id} href={l.href} {...cur(l.id)} onClick={() => setOpen(false)}>{l.label}</a>)}
          <a href={PROPOSAL_PDF}>Proposal (PDF)</a>
        </div>
      )}
    </header>
  )
}
