import { ValuesProvider } from './lib/data'
import Hero from './components/Hero'
import Stage from './components/Stage'
import SimPanel from './components/SimPanel'
import TryIt from './components/TryIt'
import { Method, Pools, Honest, Quantum, Team, Documents, Footer, PDFS } from './components/Sections'

export default function App() {
  return (
    <ValuesProvider>
      <Hero proposalHref={PDFS.proposal} />
      <main>
        <Stage />
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
