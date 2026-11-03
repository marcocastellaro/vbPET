"""Build PLAN.html (self-contained page) from PLAN.md.

    python3 docs/iipd/build_html.py

Needs the ``markdown`` package (system python3 has it).
"""

import re
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "PLAN.md"
OUT = HERE / "PLAN.html"

DIAGRAM = """
<figure class="diagram">
<svg viewBox="0 0 910 380" role="img" aria-label="L'LLM parla con l'utente e propone patch alle direttive; il planner deterministico esegue i passi del motore scientifico, i cui risultati vanno nello store; il report legge lo store e l'LLM lo spiega.">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
  </marker>
</defs>
<g fill="none" stroke="currentColor" stroke-width="1.4">
  <rect x="20" y="30" width="150" height="44" rx="6"/>
  <rect x="220" y="30" width="200" height="44" rx="6" class="llm"/>
  <rect x="470" y="30" width="180" height="44" rx="6"/>
  <rect x="700" y="30" width="180" height="44" rx="6"/>
  <rect x="20" y="185" width="150" height="50" rx="6"/>
  <rect x="200" y="150" width="690" height="120" rx="10" class="engine"/>
  <rect x="215" y="185" width="120" height="50" rx="5"/>
  <rect x="350" y="185" width="120" height="50" rx="5"/>
  <rect x="485" y="185" width="120" height="50" rx="5"/>
  <rect x="620" y="185" width="120" height="50" rx="5"/>
  <rect x="755" y="185" width="120" height="50" rx="5"/>
  <rect x="450" y="310" width="200" height="44" rx="6"/>
</g>
<g fill="none" stroke="currentColor" stroke-width="1.2">
  <line x1="172" y1="52" x2="218" y2="52" marker-start="url(#arr)" marker-end="url(#arr)"/>
  <line x1="420" y1="52" x2="468" y2="52" marker-end="url(#arr)"/>
  <line x1="650" y1="52" x2="698" y2="52" marker-end="url(#arr)"/>
  <line x1="335" y1="210" x2="348" y2="210" marker-end="url(#arr)"/>
  <line x1="470" y1="210" x2="483" y2="210" marker-end="url(#arr)"/>
  <line x1="605" y1="210" x2="618" y2="210" marker-end="url(#arr)"/>
  <line x1="740" y1="210" x2="753" y2="210" marker-end="url(#arr)"/>
  <line x1="815" y1="74" x2="815" y2="148" marker-end="url(#arr)"/>
  <line x1="550" y1="270" x2="550" y2="308" marker-end="url(#arr)"/>
  <polyline points="650,332 900,332 900,62 882,62" marker-end="url(#arr)"/>
  <polyline points="450,332 95,332 95,237" marker-end="url(#arr)"/>
  <polyline points="95,185 95,128 280,128 280,76" marker-end="url(#arr)"/>
  <polyline points="740,74 740,106 380,106 380,76" stroke-dasharray="5 4" marker-end="url(#arr)"/>
</g>
<g fill="currentColor" font-size="13" text-anchor="middle">
  <text x="95" y="57">Utente</text>
  <text x="320" y="50" font-weight="600">LLM</text>
  <text x="320" y="66" font-size="11">interpreta e spiega</text>
  <text x="560" y="50" font-weight="600">Direttive</text>
  <text x="560" y="66" font-size="11">versionate, validate</text>
  <text x="790" y="50" font-weight="600">Planner</text>
  <text x="790" y="66" font-size="11">regole deterministiche</text>
  <text x="95" y="208" font-weight="600">Report</text>
  <text x="95" y="224" font-size="11">registro evidenze</text>
  <text x="545" y="170" font-size="11" letter-spacing="0.06em">MOTORE SCIENTIFICO</text>
  <text x="275" y="207">A. Ingestione</text><text x="275" y="223" font-size="11">e QC</text>
  <text x="410" y="207">B. Esplorazione</text><text x="410" y="223" font-size="11">Patlak, Logan, SA</text>
  <text x="545" y="207">C. Fit</text><text x="545" y="223" font-size="11">per organo / ROI</text>
  <text x="680" y="207">E. Confronto</text><text x="680" y="223" font-size="11">identificabilità</text>
  <text x="815" y="207">F. Cluster</text><text x="815" y="223" font-size="11">e voxel (vbpet)</text>
  <text x="550" y="330" font-weight="600">Store</text>
  <text x="550" y="346" font-size="11">cache, risultati con ID, decisioni</text>
</g>
<g fill="currentColor" font-size="11">
  <text x="195" y="44" text-anchor="middle">chat</text>
  <text x="445" y="44" text-anchor="middle">patch</text>
  <text x="822" y="116">passi del piano</text>
  <text x="557" y="294">risultati</text>
  <text x="893" y="200" text-anchor="end">stato</text>
  <text x="270" y="326">dati del report</text>
  <text x="188" y="122">spiegazione con ID</text>
  <text x="560" y="100" text-anchor="middle">domanda bloccante</text>
</g>
</svg>
<figcaption>Separazione dei ruoli: l'LLM scrive solo patch alle direttive e legge risultati per ID; ogni calcolo passa dal planner e dal motore, e ogni numero del report viene dallo store.</figcaption>
</figure>
"""

TAGS = {
    "ESISTE": "ok", "PROPOSTA": "prop", "DA VERIFICARE": "check",
}


def convert(md_text):
    # mermaid block -> hand-drawn SVG
    md_text = re.sub(r"```mermaid.*?```", "@@DIAGRAM@@", md_text, flags=re.S)
    # links to repo files do not work in the page: keep the text as code
    md_text = re.sub(r"\[([^\]]+)\]\((?:\.\./)+[^)]+\)", r"`\1`", md_text)
    md_text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", md_text)
    html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc", "sane_lists"],
                             extension_configs={"toc": {"toc_depth": "2-3"}})
    html = html.replace("<p>@@DIAGRAM@@</p>", DIAGRAM)
    for word, cls in TAGS.items():
        html = re.sub(r"<strong>\[" + word + r"\]</strong>", f'<span class="tag {cls}">{word}</span>', html)
        html = html.replace(f"[{word}]", f'<span class="tag {cls}">{word}</span>')
    html = re.sub(r"<table>", '<div class="tw"><table>', html)
    html = html.replace("</table>", "</table></div>")
    # wrap each h2 section for spacing
    return html


def toc(html):
    items = []
    for lvl, hid, text in re.findall(r'<h([23]) id="([^"]+)">(.*?)</h\1>', html):
        text = re.sub(r"<.*?>", "", text)
        items.append(f'<li class="l{lvl}"><a href="#{hid}">{text}</a></li>')
    return "<ol>" + "".join(items) + "</ol>"


SUMMARY = """
<section class="summary" aria-label="Sintesi">
  <div class="card">
    <h3>Decisioni prese</h3>
    <ul>
      <li><b>D1</b> Ambito: PET dinamica <b>whole-body / total-body</b> (IDIF, delay e dispersione per organo, modelli per organo).</li>
      <li><b>D2</b> <b>Repository separato</b>; il fitting è ereditato da <code>vbpet</code> (dipendenza fissata a un commit).</li>
    </ul>
  </div>
  <div class="card open">
    <h3>Domande aperte</h3>
    <ul>
      <li><b>B1</b> Quale lavoro di Veronese: SA classica, SAIF, SAKE o multi-scala?</li>
      <li><b>B4</b> Metadati e risultati possono andare a un'API LLM esterna?</li>
      <li><b>B5</b> FDG o traccianti nuovi? Dati total-body locali?</li>
      <li>Hosting del nuovo repository: gitlab.dei o GitHub.</li>
    </ul>
  </div>
  <div class="card wide">
    <h3>Roadmap</h3>
    <ol class="ms">
      <li><b>M0</b><span>Repo, schemi, registro, test su vbpet</span></li>
      <li><b>M1</b><span>QC, Patlak, Logan, SA</span></li>
      <li><b>M2</b><span>Fit, incertezza, criteri, dispersione</span></li>
      <li><b>M3</b><span>Planner, audit, report</span></li>
      <li><b>M4</b><span>Direttive e LLM</span></li>
      <li><b>M5</b><span>Cluster e voxel</span></li>
      <li><b>M6</b><span>Doppio input, SAIF, RS, dati reali</span></li>
    </ol>
    <p class="legend">Etichette nel testo:
      <span class="tag ok">ESISTE</span> già nel codice
      <span class="tag prop">PROPOSTA</span> da costruire
      <span class="tag check">DA VERIFICARE</span> ipotesi non controllata</p>
  </div>
</section>
"""

CSS = """
:root {
  --paper: #f5f6f4; --surface: #ffffff; --ink: #1a2128; --muted: #5b6670;
  --line: #d8ddd9; --accent: #0d6b72; --accent-soft: #e2f0ef;
  --ok: #2f7a3e; --ok-bg: #e5f2e6; --prop: #2f5d9e; --prop-bg: #e6eef9;
  --check: #8a5a00; --check-bg: #fbf0d9;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", Roboto, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
  /* layout: sticky contents rail on the left, one reading column on the right */
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #11161a; --surface: #182026; --ink: #e4e9ea; --muted: #9aa6ad;
    --line: #2c373f; --accent: #5cc3c9; --accent-soft: #163338;
    --ok: #7fcf8c; --ok-bg: #1a3020; --prop: #8fb4ec; --prop-bg: #1b2a40;
    --check: #e8c06a; --check-bg: #3a2e14; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --paper: #11161a; --surface: #182026; --ink: #e4e9ea; --muted: #9aa6ad;
  --line: #2c373f; --accent: #5cc3c9; --accent-soft: #163338;
  --ok: #7fcf8c; --ok-bg: #1a3020; --prop: #8fb4ec; --prop-bg: #1b2a40;
  --check: #e8c06a; --check-bg: #3a2e14; color-scheme: dark;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
body { margin: 0; background: var(--paper); color: var(--ink); font: 15.5px/1.6 var(--body); }
.wrap { display: grid; grid-template-columns: 270px minmax(0, 1fr); gap: 40px;
        max-width: 1280px; margin: 0 auto; padding: 0 24px; }
nav { position: sticky; top: 0; align-self: start; max-height: 100vh; overflow-y: auto;
      padding: 28px 0 40px; font-size: 13px; }
nav .nt { font: 600 12px var(--display); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
nav ol { list-style: none; margin: 0; padding: 0; }
nav li.l2 { margin-top: 8px; font-weight: 600; }
nav li.l3 { padding-left: 12px; }
nav a { color: var(--ink); text-decoration: none; display: block; padding: 2px 6px; border-radius: 4px; }
nav li.l3 a { color: var(--muted); }
nav a:hover, nav a:focus-visible { background: var(--accent-soft); color: var(--accent); outline: none; }
main { padding: 28px 0 80px; min-width: 0; }
header.top { border-bottom: 2px solid var(--accent); padding-bottom: 16px; margin-bottom: 24px; }
header.top .eyebrow { font: 600 12px var(--display); letter-spacing: .1em; text-transform: uppercase; color: var(--accent); }
h1 { font: 600 38px/1.15 var(--display); margin: 6px 0 8px; text-wrap: balance; }
header.top p { margin: 0; color: var(--muted); max-width: 70ch; }
h2 { font: 600 27px/1.25 var(--display); margin: 56px 0 12px; padding-top: 12px; border-top: 1px solid var(--line); text-wrap: balance; scroll-margin-top: 12px; }
h3 { font: 600 20px/1.3 var(--display); margin: 32px 0 8px; scroll-margin-top: 12px; }
h4 { font: 600 17px/1.3 var(--display); margin: 28px 0 6px; color: var(--accent); scroll-margin-top: 12px; }
main > p, main > ul, main > ol, main li, blockquote p { max-width: 78ch; }
a { color: var(--accent); }
code { font: 0.86em var(--mono); background: var(--accent-soft); padding: 1px 4px; border-radius: 3px; overflow-wrap: anywhere; }
pre { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; overflow-x: auto; font-size: 13px; line-height: 1.5; }
pre code { background: none; padding: 0; overflow-wrap: normal; }
.tw { overflow-x: auto; margin: 14px 0 18px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; vertical-align: top; padding: 8px 12px; border-bottom: 1px solid var(--line); }
th { font: 600 13px var(--display); letter-spacing: .02em; background: var(--accent-soft); color: var(--ink); }
tr:last-child td { border-bottom: none; }
td code { white-space: normal; }
blockquote { margin: 14px 0; padding: 4px 18px; border-left: 3px solid var(--accent); background: var(--surface); border-radius: 0 6px 6px 0; }
blockquote p { margin: 10px 0; }
blockquote strong:first-child { color: var(--accent); }
hr { border: none; }
del { color: var(--muted); }
.tag { display: inline-block; font: 600 11px/1.5 var(--display); letter-spacing: .05em; padding: 0 7px; border-radius: 10px; vertical-align: 1px; white-space: nowrap; }
.tag.ok { color: var(--ok); background: var(--ok-bg); }
.tag.prop { color: var(--prop); background: var(--prop-bg); }
.tag.check { color: var(--check); background: var(--check-bg); }
.summary { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 8px 0 12px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 18px; }
.card.open { border-left: 4px solid var(--check); }
.card.wide { grid-column: 1 / -1; }
.card h3 { margin: 0 0 6px; font-size: 16px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.card ul { margin: 0; padding-left: 18px; }
.card li { margin: 4px 0; }
.ms { list-style: none; padding: 0; margin: 6px 0 10px; display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 6px; }
.ms li { border-top: 3px solid var(--accent); padding: 6px 4px 0; font-size: 13px; line-height: 1.35; }
.ms b { display: block; font: 600 15px var(--display); color: var(--accent); }
.legend { margin: 6px 0 0; font-size: 13px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }
figure.diagram { margin: 18px 0; padding: 16px; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow-x: auto; color: var(--ink); }
figure.diagram svg { width: 100%; min-width: 640px; height: auto; display: block; font-family: var(--body); }
figure.diagram .llm { stroke: var(--prop); stroke-width: 2; }
figure.diagram .engine { stroke: var(--accent); stroke-width: 2; }
figcaption { font-size: 13px; color: var(--muted); margin-top: 10px; max-width: 78ch; }
@media (max-width: 900px) {
  .wrap { grid-template-columns: 1fr; gap: 0; }
  nav { position: static; max-height: none; border-bottom: 1px solid var(--line); padding-bottom: 16px; }
  .summary { grid-template-columns: 1fr; }
  .ms { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  h1 { font-size: 30px; }
}
"""


def main():
    md_text = SRC.read_text()
    # drop the markdown H1 and status lines: they go in the page header
    body_md = md_text.split("\n---\n", 1)[1]
    html = convert(body_md)
    page = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Piano IIPD</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<nav aria-label="Indice"><div class="nt">Indice</div>{toc(html)}</nav>
<main>
<header class="top">
  <div class="eyebrow">Documento di pianificazione · 3 novembre 2026</div>
  <h1>InteractiveIdentificationPETData (IIPD)</h1>
  <p>Framework per identificare e fittare modelli cinetici su PET dinamica whole-body, anche per traccianti nuovi, con un assistente LLM come interfaccia e un motore scientifico verificabile basato su <code>vbpet</code>. Solo pianificazione: nessun codice IIPD scritto.</p>
</header>
{SUMMARY}
{html}
</main>
</div>
</body>
</html>
"""
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
