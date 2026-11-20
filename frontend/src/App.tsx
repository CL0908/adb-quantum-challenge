import { ValuesProvider } from './lib/data'
import TopBar from './components/TopBar'
import Stage from './components/Stage'
import SimPanel from './components/SimPanel'
import TryIt from './components/TryIt'
import { Method, Pools, Honest, Quantum, Team, Documents, Footer, PDFS } from './components/Sections'

export default function App() {
  return (
    <ValuesProvider>
      <TopBar proposalHref={PDFS.proposal} />
      <main>
        <Stage proposalHref={PDFS.proposal} />
        <SimPanel />
        <Method />
        <Pools />
        <Honest />
        <Quantum />
        <TryIt />
        <Team />
        <Documents />
      </main>
      <Footer />
    </ValuesProvider>
  )
}
