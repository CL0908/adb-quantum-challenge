"""One-command builder for the challenge upload package.

Usage (from the repository root):
    PYTHONPATH=. python -m src.viz.package_submission [--cv path/to/cv.pdf] [--skip-build]

Writes submission/upload/ (gitignored):
    01_Technical_Proposal.pdf   main proposal followed by the technical annex
    02_Team_Profile_CVs.pdf     CVs given with --cv (repeatable), never committed
    03_Source_Code.pdf, 04_Source_Code.zip   only with --code-pdf (the portal takes a GitHub URL instead)
    README_upload.txt                     what goes where, sizes, page counts, deliverable checklist

The challenge accepts PDF/PPT/MP3/MPEG/MOV/MP4 only, so the source code is delivered as a PDF; the zip is optional.
Only committed files are listed: uncommitted work never reaches the package. The CV is never written outside
submission/.
"""
import html
import importlib.metadata as md
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename, TextLexer

ROOT = Path(".").resolve()
OUT = ROOT / "submission/upload"
PAPER = ROOT / "docs/paper"
AUTHOR = "Tan Chun Loong"
SIZE_LIMIT_MB = 25

CODE_ROOTS = ("src/", "tests/", "api/")
ROOT_TEXT = ["requirements.txt", "README.md", "HANDOFF.md"]


def sh(*cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def git(*args):
    return sh("git", *args)


def pages(pdf):
    out = sh("pdfinfo", str(pdf))
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else 0


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


# ---------------------------------------------------------------- step 1: papers
def build_papers(cv):
    cmd = [sys.executable, "-m", "src.viz.build_paper"] + (["--cv", cv] if cv else [])
    print("building papers:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env={**__import__("os").environ, "PYTHONPATH": "."})
    if r.returncode != 0:
        sys.exit("FAIL: build_paper exited with status %d" % r.returncode)
    for f in ("proposal.filled.md", "short.filled.md"):
        txt = (PAPER / f).read_text()
        if "{{" in txt:
            left = sorted(set(re.findall(r"\{\{(\w+)\}\}", txt)))
            sys.exit(f"FAIL: placeholder survives in {f}: {left or 'stray {{'}")


# ---------------------------------------------------------------- step 2: code listing
def tracked():
    """(path, size) of every file at HEAD."""
    rows = []
    for line in git("ls-tree", "-r", "-l", "-z", "HEAD").split("\0"):
        if not line:
            continue
        meta, path = line.split("\t", 1)
        rows.append((path, int(meta.split()[3])))
    return sorted(rows)


def show(path):
    return sh("git", "show", f"HEAD:{path}")


def ai_use_statement():
    txt = (PAPER / "short.md").read_text().splitlines()
    for i, ln in enumerate(txt):
        if ln.startswith("* **AI use.**"):
            buf = [ln[2:]]
            for nxt in txt[i + 1:]:
                if nxt.startswith("  ") and nxt.strip():
                    buf.append(nxt.strip())
                else:
                    break
            s = " ".join(buf)
            return re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    sys.exit("FAIL: AI-use paragraph not found in docs/paper/short.md")


def pkg_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "not installed"


def entry_points(files):
    """Modules with a __main__ block: python -m command, docstring first line, results/ files they mention."""
    rows = []
    for p in files:
        if not p.startswith(("src/", "api/")) or not p.endswith(".py") or "/viz/art" in p:
            continue
        s = show(p)
        if "__main__" not in s:
            continue
        mod = p[:-3].replace("/", ".")
        doc = ""
        m = re.match(r'\s*(?:#[^\n]*\n)*\s*"""(.*?)"""', s, re.S)
        if m:
            doc = " ".join(m.group(1).strip().split("\n\n")[0].split())
            doc = doc[:170]
        res = sorted(set(re.findall(r"results/[A-Za-z0-9_./]*\.(?:json|jsonl)", s)))
        rows.append((mod, doc, res))
    return rows


CSS = """
@page { size: A4; margin: 12mm 10mm 14mm 10mm; }
html { font-family: 'DejaVu Sans', 'Helvetica', sans-serif; font-size: 9.5pt; color: #111; }
body { margin: 0; }
h1 { font-size: 20pt; margin: 0 0 4mm 0; } h2 { font-size: 12pt; margin: 6mm 0 2mm 0; border-bottom: 0.4pt solid #888; }
.cover p, .cover li { line-height: 1.35; margin: 1.2mm 0; }
code, pre, .src { font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace; }
code { font-size: 8.3pt; background: #f0f0f0; padding: 0 1mm; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; }
td, th { border-bottom: 0.3pt solid #ccc; padding: 0.6mm 1.5mm; text-align: left; vertical-align: top; }
td.r, th.r { text-align: right; white-space: nowrap; }
.toc td { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.4pt; }
.pb { page-break-before: always; }
.filehead { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; font-weight: bold; background: #e9edf2;
            border-left: 1.2mm solid #345; padding: 1mm 2mm; margin: 0 0 1.5mm 0; page-break-after: avoid; }
.filehead span { font-weight: normal; color: #555; }
.src { font-size: 7.5pt; line-height: 1.22; margin: 0; white-space: pre-wrap; overflow-wrap: anywhere;
       tab-size: 4; }
.src .linenos { color: #8a8f98; user-select: none; }
.hl .linenos { margin-right: 1.5mm; }
"""


def render_html(commit, cdate, ai, files, entries, excluded, pagemap, python_ver, vers):
    h = HtmlFormatter(style="default", cssclass="hl", linenos="inline")
    parts = [f"<html><head><meta charset='utf-8'><title>Source code listing</title><style>{CSS}\n{h.get_style_defs('.hl')}"
             f"\n.hl pre{{margin:0}}</style></head><body>"]
    # cover
    parts.append("<div class='cover'><h1>Source Code Listing</h1>")
    parts.append(f"<p><b>Rényi-Doob Policy Optimization (RDPO)</b> · ADB Quantum Challenge, Phase 1 · {html.escape(AUTHOR)}</p>")
    parts.append(f"<p>Repository: <code>adb-quantum-challenge</code> (https://github.com/CL0908/adb-quantum-challenge)<br>"
                 f"Commit: <code>{commit}</code> · committed {cdate} · listing generated {date.today().isoformat()}</p>")
    parts.append("<p>This PDF contains every Python source file under <code>src/</code>, <code>tests/</code> and "
                 "<code>api/</code>, plus <code>requirements.txt</code>, <code>README.md</code> and <code>HANDOFF.md</code>, "
                 "exactly as committed. Cached public data and result files are reproducible and are listed by name and "
                 "size in Appendix A instead of being printed. A zip of the same commit accompanies this PDF as a convenience copy.</p>")
    parts.append("<h2>How to reproduce</h2><ol>")
    parts.append(f"<li>Python {python_ver}; <code>pip install -r requirements.txt</code></li>")
    parts.append("<li>Run the test-suite: <code>python -m pytest -q tests/</code> (tests that need Qiskit skip without it)</li>")
    parts.append("<li>Each result file is produced by a module entry point, run from the repository root as "
                 "<code>PYTHONPATH=. python -m &lt;module&gt;</code> (table below).</li>")
    parts.append("<li>Rebuild both documents (every number is filled from <code>results/*.json</code>): "
                 "<code>PYTHONPATH=. python -m src.viz.build_paper</code></li></ol>")
    parts.append("<h2>Environment and simulator details</h2><ul>")
    parts.append("<li>Operating system and hardware: Linux x86-64, CPU only (no GPU), no quantum hardware used.</li>")
    parts.append("<li>Pinned in <code>requirements.txt</code>: " + html.escape(", ".join(vers["req"])) + "</li>")
    parts.append("<li>Hardware-plan stack (installed here, not pinned): " +
                 ", ".join(f"{n} {v}" for n, v in vers["qiskit"]) + "</li>")
    parts.append("<li>Simulators: exact state-vector and resource simulators written for this project (<code>src/quantum/</code>); "
                 "the near-term hardware plan (<code>src/quantum/hardware_plan.py</code>) uses Qiskit Aer with a fake IBM backend noise snapshot.</li></ul>")
    parts.append("<h2>AI use</h2><p>" + html.escape(ai) + "</p>")
    parts.append("<h2>Entry points</h2><table><tr><th>Command</th><th>Purpose</th><th>Result files referenced</th></tr>")
    for mod, doc, res in entries:
        parts.append(f"<tr><td><code>python -m {mod}</code></td><td>{html.escape(doc)}</td>"
                     f"<td>{html.escape(' '.join(r.replace('results/', '') for r in res))}</td></tr>")
    parts.append("</table></div>")
    # TOC
    parts.append("<div class='pb toc'><h1>Contents</h1><table><tr><th>#</th><th>File</th><th class='r'>Lines</th><th class='r'>Page</th></tr>")
    for i, (p, nlines) in enumerate(files, 1):
        pg = pagemap.get(i, "")
        parts.append(f"<tr><td>{i:03d}</td><td>{html.escape(p)}</td><td class='r'>{nlines}</td><td class='r'>{pg}</td></tr>")
    parts.append(f"<tr><td></td><td>Appendix A: data and result files not printed</td><td></td><td class='r'>{pagemap.get('A', '')}</td></tr></table></div>")
    # files
    for i, (p, nlines) in enumerate(files, 1):
        try:
            lexer = get_lexer_for_filename(p)
        except Exception:
            lexer = TextLexer()
        if p.endswith(".md"):
            lexer = TextLexer()
        body = highlight(show(p).expandtabs(4), lexer, h)
        parts.append(f"<div class='pb'><div class='filehead'>&sect;{i:03d} &nbsp;{html.escape(p)} <span>· {nlines} lines</span></div>"
                     f"<div class='src'>{body}</div></div>")
    # appendix
    parts.append("<div class='pb'><h1>Appendix A: files not printed</h1>"
                 "<p>Cached public data, generated results and figures are reproducible from the code above "
                 "(or are public downloads cached for offline use). They are part of the zip and the repository.</p>"
                 "<table><tr><th>File</th><th class='r'>Size</th></tr>")
    for p, sz in excluded:
        parts.append(f"<tr><td>{html.escape(p)}</td><td class='r'>{human(sz)}</td></tr>")
    parts.append("</table></div></body></html>")
    return "".join(parts)


NODE_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const [src, out] = process.argv.slice(2);
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + src, { waitUntil: 'load' });
  await p.pdf({ path: out, format: 'A4', printBackground: true, preferCSSPageSize: true, displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate: '<div style="width:100%;font-family:sans-serif;font-size:7pt;color:#777;padding:0 10mm;display:flex;justify-content:space-between;"><span>RDPO source code listing (committed files)</span><span class="pageNumber"></span></div>' });
  await b.close();
})();
"""


def find_pages(pdf, n_files, toc_end):
    """Page on which each file heading marker lands (pdftotext, form-feed separated)."""
    txt = sh("pdftotext", "-layout", str(pdf), "-")
    pg = txt.split("\f")
    found = {}
    for pno, t in enumerate(pg, 1):
        if pno <= toc_end:
            continue
        for m in re.finditer(r"§(\d{3})\b", t):
            found.setdefault(int(m.group(1)), pno)
        if "Appendix A: files not printed" in t and "Contents" not in t and "A" not in found:
            found["A"] = pno
    return found


def code_pdf(dest):
    commit = git("rev-parse", "HEAD").strip()
    cdate = git("log", "-1", "--format=%cI").strip()[:10]
    allf = tracked()
    printed_paths = [p for p, _ in allf if p.endswith(".py") and p.startswith(CODE_ROOTS)] + \
                    [p for p in ROOT_TEXT if any(p == q for q, _ in allf)]
    printed_paths = sorted(printed_paths, key=lambda p: (p not in ROOT_TEXT[:0], p))
    # order: root text files first, then code
    printed_paths = [p for p in ROOT_TEXT if p in printed_paths] + [p for p in printed_paths if p not in ROOT_TEXT]
    files = [(p, show(p).count("\n") + (0 if show(p).endswith("\n") else 1)) for p in printed_paths]
    pset = set(printed_paths)
    excluded = [(p, s) for p, s in allf if p not in pset]
    req = [ln.strip() for ln in show("requirements.txt").splitlines() if ln.strip() and not ln.startswith("#")]
    vers = {"req": req, "qiskit": [(n, pkg_version(n)) for n in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime")]}
    python_ver = sys.version.split()[0]
    ai = ai_use_statement()
    entries = entry_points([p for p, _ in allf])

    work = OUT.parent / "_work"
    work.mkdir(exist_ok=True)
    (work / "pdf.js").write_text(NODE_JS)
    h_path, pdf_path = work / "code.html", work / "code.pdf"
    env = dict(__import__("os").environ)
    pagemap = {i: "000" for i in range(1, len(files) + 1)}
    pagemap["A"] = "000"
    toc_end = 0
    for attempt in range(2):
        h_path.write_text(render_html(commit, cdate, ai, files, entries, excluded, pagemap, python_ver, vers))
        subprocess.run(["node", str(work / "pdf.js"), str(h_path), str(pdf_path)], check=True, env=env)
        if attempt == 0:
            # TOC ends on the page before the first file heading
            first = find_pages(pdf_path, len(files), 0)
            toc_end = min(v for k, v in first.items() if k != "A") - 1
        pagemap = find_pages(pdf_path, len(files), toc_end)
        missing = [i for i in range(1, len(files) + 1) if i not in pagemap]
        if missing:
            sys.exit(f"FAIL: could not locate pages for files {missing[:5]}")
    # file headings repeat nothing, but a file can span pages: fine, TOC gives the first page
    pdf_path.replace(dest)
    return commit, cdate, files, excluded


# ---------------------------------------------------------------- step 3: orchestration
REPO_URL = "https://github.com/CL0908/adb-quantum-challenge"


def main():
    """Build the portal package as requested by the organisers' email of 15 Nov 2026:
    Technical Proposal [PDF], Lightweight Prototype [GitHub URL], Team Profile/CVs [PDF], optional presentation.

    --cv PATH may be given several times (one per team member, in order); CVs are never committed.
    --code-pdf additionally writes the printable source listing (03) and a zip of HEAD (04).
    """
    argv = sys.argv[1:]
    cvs = [argv[i + 1] for i, a in enumerate(argv) if a == "--cv"]
    for cv in cvs:
        if not Path(cv).is_file():
            sys.exit(f"FAIL: CV file not found: {cv}")
    if "submission/" not in (ROOT / ".gitignore").read_text().split():
        with open(ROOT / ".gitignore", "a") as f:
            f.write("submission/\n")
    if "--skip-build" not in argv:
        build_papers(None)          # the papers carry no CV; CVs go into the team file
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()

    short, full = ROOT / "docs/RDPO_Proposal_Short.pdf", ROOT / "docs/RDPO_Proposal.pdf"
    for f in (short, full):
        if not f.exists():
            sys.exit(f"FAIL: missing {f}")
    p1 = OUT / "01_Technical_Proposal.pdf"            # main proposal followed by the technical annex
    subprocess.run(["pdfunite", str(short), str(full), str(p1)], check=True)
    made = [p1]
    p2 = OUT / "02_Team_Profile_CVs.pdf"
    if cvs:
        if len(cvs) == 1:
            p2.write_bytes(Path(cvs[0]).read_bytes())
        else:
            subprocess.run(["pdfunite", *cvs, str(p2)], check=True)
        made.append(p2)
    else:
        print("WARNING: no --cv given; 02_Team_Profile_CVs.pdf not written")
    if "--code-pdf" in argv:
        p3 = OUT / "03_Source_Code.pdf"
        code_pdf(p3)
        made.append(p3)
        p4 = OUT / "04_Source_Code.zip"
        subprocess.run(["git", "archive", "--format=zip", "-o", str(p4), "HEAD"], check=True)
        if any(n.startswith("submission/") for n in sh("unzip", "-Z1", str(p4)).split()):
            sys.exit("FAIL: submission/ leaked into the zip")

    info = {}
    for p in made:
        if p.suffix == ".pdf":
            n = pages(p)
            if n == 0:
                sys.exit(f"FAIL: 0 pages in {p}")
            info[p.name] = (p.stat().st_size, n)
    for fm in ("short.filled.md", "proposal.filled.md"):
        fp = ROOT / "docs/paper" / fm
        if fp.exists() and "{{" in fp.read_text():
            sys.exit(f"FAIL: unfilled placeholder in {fm}")
    commit = git("rev-parse", "HEAD").strip()
    nshort = pages(short)
    L = ["ADB Quantum Innovation Challenge, Phase 1: upload package",
         f"Built from commit {commit} on {date.today()}", "",
         "WHAT THE PORTAL ASKS FOR (organisers' email, 15 Nov 2026) -> WHAT TO UPLOAD",
         "  Submission form ............ fill in on challenges.adb.org",
         f"  Technical Proposal [PDF] ... 01_Technical_Proposal.pdf  (pages 1-{nshort}: main proposal; "
         f"pages {nshort + 1}-: technical annex)",
         f"  Lightweight Prototype [URL]  {REPO_URL}   (repository must be PUBLIC before submitting)",
         "  Team Profile/CVs [PDF] ..... 02_Team_Profile_CVs.pdf  (one CV per member, rebuild with --cv for each)",
         "  Presentation [optional] .... not prepared", "", "FILES"]
    for name, (sz, n) in info.items():
        L.append(f"  {name:32s} {human(sz):>9s} {n:4d} pages")
    L += ["", "REBUILD", "  PYTHONPATH=. python -m src.viz.package_submission --cv submission/<member1>_CV.pdf "
          "--cv submission/<member2>_CV.pdf", "  (CVs live in submission/, which is gitignored)"]
    (OUT / "README_upload.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    dirty = git("status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        print("\nNOTE: uncommitted tracked changes exist; commit and push before submitting so the URL matches the PDFs.")


# ---------------------------------------------------------------- README
CHECKLIST = [
    ("1. Scientific Proposal", "01_Proposal_TanChunLoong.pdf",
     "Sections 1 (Problem and classical bottleneck), 2 (Scientific approach); full detail in 02_Technical_Annex"),
    ("2. Resource Estimation", "01_Proposal_TanChunLoong.pdf",
     "Section 4 (Resource estimation); derivations in 02_Technical_Annex"),
    ("3. Lightweight Prototype: results", "01_Proposal_TanChunLoong.pdf",
     "Section 3 (Prototype and results)"),
    ("3. Lightweight Prototype: source code", "03_Source_Code_TanChunLoong.pdf (+ 04 zip)",
     "all committed .py files, README, requirements; table of contents and Appendix A"),
    ("3. Lightweight Prototype: environment, hardware/simulator details", "03_Source_Code_TanChunLoong.pdf",
     "cover page (Python, requirements, Linux x86-64 CPU only, Qiskit stack); also 01 Section 7"),
    ("3. Lightweight Prototype: AI use", "03_Source_Code_TanChunLoong.pdf",
     "cover page, 'AI use' (verbatim from 01 Section 7)"),
    ("4. Phase 2 Plan of Action", "01_Proposal_TanChunLoong.pdf",
     "Section 5 (Phase 2 plan of action, 30 Nov 2026 - 22 Jan 2027)"),
    ("5. Team", "01_Proposal_TanChunLoong.pdf",
     "Section 6 (Team)" + "; CV appended at the end if built with --cv"),
]


def short_headings():
    return [ln for ln in (PAPER / "short.md").read_text().splitlines() if re.match(r"#{1,2} \d", ln)]


def write_readme(info, zip_path, commit, cdate, with_cv, nfiles):
    L = []
    L.append("ADB Quantum Challenge, Phase 1: upload package")
    L.append(f"Author: {AUTHOR}")
    L.append(f"Commit: {commit} ({cdate}); built {date.today().isoformat()}")
    L.append("Accepted formats on the portal: PDF, PPT, MP3, MPEG, MOV, MP4. Language: English.")
    L.append("CV appended to file 01: " + ("yes" if with_cv else "NO (rebuild with --cv to append it)"))
    L.append("")
    L.append("FILES")
    roles = {
        "01_Proposal_TanChunLoong.pdf": "main proposal (upload as the proposal / main document)",
        "02_Technical_Annex_TanChunLoong.pdf": "technical annex (upload as supporting document)",
        "03_Source_Code_TanChunLoong.pdf": "source code, environment, simulator details, AI use (upload as the prototype / code document)",
    }
    for name, (sz, n) in info.items():
        L.append(f"  {name}  {human(sz)}  {n} pages")
        L.append(f"      -> {roles[name]}")
    L.append(f"  {zip_path.name}  {human(zip_path.stat().st_size)}  (git archive of the commit; convenience copy only,")
    L.append("      zip is not an accepted portal format; attach only if the portal allows, or link the repository)")
    L.append(f"  Code PDF lists {nfiles} committed files; data/results are listed by name and size in its Appendix A.")
    L.append("")
    L.append("PHASE 1 DELIVERABLES -> FILE AND SECTION")
    for what, f, where in CHECKLIST:
        L.append(f"  [ ] {what}")
        L.append(f"        file: {f}")
        L.append(f"        where: {where}")
    L.append("")
    L.append("SECTIONS OF 01_Proposal (docs/paper/short.md)")
    for h in short_headings():
        L.append("  " + h.replace("#", " ").strip() if h.startswith("# ") else "      " + h.lstrip("# "))
    L.append("")
    L.append("REBUILD")
    L.append("  PYTHONPATH=. python -m src.viz.package_submission --cv /path/to/CV.pdf")
    L.append("  (the CV stays under submission/, which is gitignored; never commit it)")
    (OUT / "README_upload.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
