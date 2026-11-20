export default function TopBar({ proposalHref }: { proposalHref: string }) {
  return (
    <header className="topbar">
      <nav className="topbar-in" aria-label="Main">
        <a className="brand" href="#top">Joint-Tail<span> · pooled disaster cover, exactly</span></a>
        <div className="toplinks">
          <a href="#top">Model</a>
          <a href="#problem">Simulation</a>
          <a href="#method">Method</a>
          <a href="#pools">Pools</a>
          <a href="#try">Try it</a>
          <a href="#team">Team</a>
          <a href="#documents">Documents</a>
          <a className="top-proposal" href={proposalHref}>Proposal (PDF)</a>
        </div>
      </nav>
    </header>
  )
}
