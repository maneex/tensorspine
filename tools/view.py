"""`--view`: a self-contained HTML visualization of an tensorspine/2.0 model.

Requires Graphviz's `dot` on PATH.

The page carries two documents. The model document is the declaration —
folded: a composition is one box with an index range. The derived document
(D1-D6, §7) is the same model unfolded — one node per emitted occurrence, one
entry per tensor, per state, per legal cut. They are two readings of one
thing, joined by the identifiers of §5.2, and the page reads both: the side
tree switches between them, the inspector puts a node's derived facts under
its declaration, a measure is painted on the diagram, and the products tab
reads a product whole.

The derived document is not computed here. It is built the way `--derive`
builds it — into a file, validated against the derived schema before it is
written — and read back; the page embeds what the emitter vouched for. A
model that cannot be derived (invalid, or a template with no assignment)
still renders: the page says so and stays on its first reading.

Division of labor: Python reads the model document, builds the top-level
graph (root occurrences <-> compositions, plus each composition's own local
graph) and hands it to Graphviz, which does the actual layout and produces
SVG — laying out a DAG correctly (independent branches side by side, real
fan-out/fan-in, no crossing minimization guesswork) is exactly graphviz's
job, not this script's. The resulting SVGs are embedded directly in the page.
Everything else — the side tree, the click-to-inspect detail panel, the raw
JSON view — stays client-side in JavaScript, reading the embedded model JSON
directly; none of that involves layout, so there is nothing graphviz buys it.

The page is a page of the documentation site and looks like one: the sidebar,
the palette, the type and the hairline rules of docs/style/catalog.css, both
in the stylesheet (TEMPLATE) and in the diagrams (the colours below). They are
restated here rather than linked because the page must stand on its own.

Assumptions made (v1 of this tool):
  - inside a composition, only edges linking the SAME index combination are
    drawn; an edge linking two different index values (e.g. a carry between
    blocks) is surfaced as a badge on the receiving node instead of a drawn
    edge — drawing it spatially in a single representative block would look
    exactly like a cycle even though it isn't one;
  - a composition's instance count, and any range label, is only shown when
    its bounds resolve to literals or literal "model_constant" quantities.
"""
import contextlib
import html
import io
import json
import model as model_mod
from expr import resolve_quantities
import math
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

# ---------- shared look ----------
# The diagrams belong to the documentation site, so they take their colours and
# their type from docs/style/catalog.css: warm neutrals, one rust accent, and
# hairline rules. Graphviz lays the labels out with the metric-compatible
# fallback faces below; the page's own CSS then swaps them for IBM Plex Mono /
# IBM Plex Sans, the faces the rest of the site uses (see TEMPLATE).
DOT_MONO = "Courier New"
DOT_SANS = "Helvetica"

BG_RAISED = "#ffffff"     # --bg-raised, node body
BG_TINT = "#efe9dc"       # --bg-tint, composition body
INK = "#22211e"           # --ink, node name
MUTED = "#6f6a60"         # --muted, secondary line
FAINT = "#8a847a"         # --faint, edges
CHIP = "#d5cfc2"          # --chip, node border
ACCENT = "#9a4a1c"        # --accent, compositions
NOTE = "#7a5a2e"          # --note-label, index range, carry and derived figures

# Every occurrence and composition node carries one more label row, kept empty
# until the page paints a derived measure into it. It is reserved at layout
# time on purpose: switching measure then repaints, and never reflows a
# diagram Graphviz has already laid out. The em dash is a placeholder with a
# glyph, so that Graphviz emits the <text> element the page writes into; the
# page empties it on load.
MEASURE_ROW = ('\u2014', 9.5, NOTE, True, False)


# ---------- minimal scalar-expression evaluator (tensorspine/2.0 §2.2) ----------
# Just enough to print index ranges / instance counts on the diagram labels.
# The full inspector-side evaluator (with conditionals, calls, domains, ...)
# stays in the page's JavaScript, where it belongs — this one only needs to
# handle what shows up in `compositions.*.indices`.

def _lit_value(quantities, name):
    """The static value of a quantity: literal, defaulted or derived (expr)."""
    key = id(quantities)
    if _RESOLVED.get('key') != key:
        _RESOLVED['key'] = key
        _RESOLVED['values'] = resolve_quantities({'quantities': quantities})
    return _RESOLVED['values'].get(name)


_RESOLVED = {}


def eval_static(e, quantities):
    if e is None:
        return None
    if 'literal' in e:
        return e['literal']
    if 'quantity' in e:
        return _lit_value(quantities, e['quantity'])
    if 'op' in e:
        args = [eval_static(a, quantities) for a in e.get('args', [])]
        if any(a is None for a in args):
            return None
        op = e['op']
        if op == 'add':
            return sum(args)
        if op == 'multiply':
            r = 1
            for a in args:
                r *= a
            return r
        if op == 'subtract':
            return args[0] - args[1]
        if op == 'divide':
            return args[0] / args[1]
        if op == 'ceil_divide':
            return -(-args[0] // args[1])
        if op == 'floor_divide':
            return args[0] // args[1]
        if op == 'modulo':
            return args[0] % args[1]
        if op == 'min':
            return min(args)
        if op == 'max':
            return max(args)
        if op == 'negate':
            return -args[0]
        if op == 'absolute':
            return abs(args[0])
    return None


def expr_str(e, quantities):
    if e is None:
        return '?'
    if 'literal' in e:
        v = e['literal']
        return json.dumps(v) if isinstance(v, str) else str(v)
    if 'quantity' in e:
        v = _lit_value(quantities, e['quantity'])
        return str(v) if v is not None else e['quantity']
    if 'index' in e:
        return e['index']
    if 'op' in e:
        args = [expr_str(a, quantities) for a in e.get('args', [])]
        if e['op'] == 'negate':
            return f"\u2212{args[0]}"
        if e['op'] == 'absolute':
            return f"|{args[0]}|"
        sym = {'add': '+', 'multiply': '\u00d7', 'subtract': '\u2212', 'divide': '\u00f7', 'modulo': 'mod'}
        return f" {sym.get(e['op'], e['op'])} ".join(args)
    return '?'


def comp_instance_count(comp, quantities):
    total = 1
    for r in comp['indices'].values():
        start = eval_static(r['start'], quantities)
        stop = eval_static(r['stop'], quantities)
        step = eval_static(r['step'], quantities)
        if start is None or stop is None or step is None:
            return None
        total *= max(0, math.ceil((stop - start) / step))
    return total


# ---------- top-level graph: root occurrences <-> compositions ----------
# Mirrors the inspector's own reading of bindings.values (kept independently
# in the page's JS, for the detail panel) but this copy only needs enough to
# hand a node/edge list to Graphviz.

def top_key(sel):
    return f"root:{sel['occurrence']}" if sel['kind'] == 'root' else f"comp:{sel['composition']}"


def build_graph(data):
    nodes = {}
    for name in data['occurrences']:
        nodes[f"root:{name}"] = {'type': 'occurrence', 'name': name}
    for name in data['compositions']:
        nodes[f"comp:{name}"] = {'type': 'composition', 'name': name}

    edge_set = set()
    edges = []

    def add_edge(a, b):
        if a != b and (a, b) not in edge_set:
            edge_set.add((a, b))
            edges.append((a, b))

    internal = {name: [] for name in data['compositions']}

    for b in data['bindings']['values'].values():
        fo, to = b['from']['occurrence'], b['to']['occurrence']
        fk, tk = top_key(fo), top_key(to)
        if fk == tk and fo['kind'] == 'generated':
            # Same composition: same index on both ends is a real edge WITHIN
            # one developed instance (drawn below). A differing index (e.g.
            # i-1 -> i) is a recurrence ACROSS instances — drawing it inside a
            # single representative block would look like a cycle without
            # being one, so it becomes a badge on the receiving node instead.
            carry = json.dumps(fo.get('indices'), sort_keys=True) != json.dumps(to.get('indices'), sort_keys=True)
            internal[fo['composition']].append({'from': fo['occurrence'], 'to': to['occurrence'], 'carry': carry})
        else:
            add_edge(fk, tk)

    for name, spec in data['interfaces']['inputs'].items():
        k = f"in:{name}"
        nodes[k] = {'type': 'input', 'name': name}
        for endpoint in spec['to']:
            add_edge(k, top_key(endpoint['occurrence']))
    for name, spec in data['interfaces']['outputs'].items():
        k = f"out:{name}"
        nodes[k] = {'type': 'output', 'name': name}
        add_edge(top_key(spec['from']['occurrence']), k)

    return nodes, edges, internal


# ---------- DOT generation ----------

def dot_qid(s):
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"') + '"'


def html_label(rows):
    """rows: list of (text, point_size, color, mono, bold)."""
    parts = ['<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1">']
    for text, size, color, mono, bold in rows:
        t = html.escape(str(text), quote=False)
        if bold:
            t = f'<B>{t}</B>'
        face = DOT_MONO if mono else DOT_SANS
        parts.append(f'<TR><TD ALIGN="CENTER"><FONT FACE="{face}" POINT-SIZE="{size}" COLOR="{color}">{t}</FONT></TD></TR>')
    parts.append('</TABLE>')
    return '<' + ''.join(parts) + '>'


def top_node_dot(node_id, node, data, quantities):
    kind = node['type']
    if kind in ('input', 'output'):
        if kind == 'input':
            label = f"in \u00b7 {node['name']}"
        else:
            spec = data['interfaces']['outputs'][node['name']]
            label = f"out \u00b7 {node['name']}" + (' \u00b7 generative' if spec.get('generative') else '')
        return (f'  {dot_qid(node_id)} [id={dot_qid(node_id)}, shape=ellipse, style="dashed", '
                f'color="{CHIP}", fontname="{DOT_MONO}", fontsize=11, '
                f'fontcolor="{MUTED}", label={dot_qid(label)}];')
    if kind == 'occurrence':
        o = data['occurrences'][node['name']]
        rows = [(node['name'], 14, INK, True, True),
                (f"{o['contract']['name']} \u00b7 {o['contract']['version']}", 10, MUTED, False, False),
                MEASURE_ROW]
        return f'  {dot_qid(node_id)} [id={dot_qid(node_id)}, color="{CHIP}", fillcolor="{BG_RAISED}", label={html_label(rows)}];'
    # composition, collapsed representation: one box, expanded structure is a
    # separate diagram rendered into its own comp-section (see comp_section_html)
    comp = data['compositions'][node['name']]
    idx_names = list(comp['indices'].keys())
    n = comp_instance_count(comp, quantities)
    range_label = ', '.join(
        f"{ix} = {expr_str(comp['indices'][ix]['start'], quantities)}\u2026{expr_str(comp['indices'][ix]['stop'], quantities)}"
        for ix in idx_names)
    badge = range_label + (f' \u00b7 \u00d7{n}' if n is not None else '')
    sites = list(comp['occurrences'].keys())
    rows = [(node['name'], 14, ACCENT, True, True),
            (badge, 9.5, NOTE, False, False),
            (f"{len(sites)} occurrence(s) per instance", 8.5, MUTED, False, False),
            MEASURE_ROW]
    return f'  {dot_qid(node_id)} [id={dot_qid(node_id)}, color="{CHIP}", fillcolor="{BG_TINT}", label={html_label(rows)}];'


def top_level_dot(data, nodes, edges, quantities):
    lines = [
        'digraph G {',
        '  bgcolor="transparent";',
        '  rankdir="TB";',
        '  nodesep=0.55; ranksep=0.6;',
        f'  node [shape=box, style="rounded,filled", fontname="{DOT_SANS}", penwidth=1, margin="0.17,0.1"];',
        f'  edge [color="{FAINT}", penwidth=1.2, arrowsize=0.7, arrowhead=vee];',
    ]
    for node_id, node in nodes.items():
        lines.append(top_node_dot(node_id, node, data, quantities))
    for a, b in edges:
        lines.append(f'  {dot_qid(a)} -> {dot_qid(b)};')
    lines.append('}')
    return '\n'.join(lines)


def comp_internal_dot(comp_name, comp_def, internal_edges):
    sites = list(comp_def['occurrences'].keys())
    carry_targets = {e['to'] for e in internal_edges if e['carry']}
    seq_edges = [(e['from'], e['to']) for e in internal_edges if not e['carry']]
    lines = [
        'digraph G {',
        '  bgcolor="transparent";',
        '  rankdir="TB";',
        '  nodesep=0.4; ranksep=0.45;',
        f'  node [shape=box, style="rounded,filled", fontname="{DOT_SANS}", penwidth=1, margin="0.15,0.09", '
        f'color="{CHIP}", fillcolor="{BG_RAISED}"];',
        f'  edge [color="{FAINT}", penwidth=1.2, arrowsize=0.6, arrowhead=vee];',
    ]
    for site in sites:
        node_id = f"{comp_name}::{site}"
        so = comp_def['occurrences'][site]
        rows = [(site, 12, INK, True, True),
                (f"{so['contract']['name']} \u00b7 {so['contract']['version']}", 9, MUTED, False, False)]
        if site in carry_targets:
            rows.append(('\u21ba carry from previous instance', 8, NOTE, False, False))
        rows.append(MEASURE_ROW)   # always last: the page finds it by position
        lines.append(f'  {dot_qid(node_id)} [id={dot_qid(node_id)}, label={html_label(rows)}];')
    for a, b in seq_edges:
        lines.append(f'  {dot_qid(comp_name + "::" + a)} -> {dot_qid(comp_name + "::" + b)};')
    lines.append('}')
    return '\n'.join(lines)


def require_dot():
    if shutil.which('dot') is None:
        print("error: graphviz's 'dot' command was not found on PATH.\n"
              "       install graphviz (e.g. 'apt install graphviz' or 'brew install graphviz') and retry.",
              file=sys.stderr)
        sys.exit(1)


def render_svg(dot_source):
    proc = subprocess.run(['dot', '-Tsvg'], input=dot_source.encode('utf-8'), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError('dot failed:\n' + proc.stderr.decode('utf-8', 'replace'))
    svg = proc.stdout.decode('utf-8')
    return svg[svg.index('<svg'):]


def comp_section_html(name, comp_def, internal_edges, quantities):
    svg = render_svg(comp_internal_dot(name, comp_def, internal_edges))
    idx_names = list(comp_def['indices'].keys())
    n = comp_instance_count(comp_def, quantities)
    range_label = ', '.join(
        f"{ix} = {expr_str(comp_def['indices'][ix]['start'], quantities)}\u2026{expr_str(comp_def['indices'][ix]['stop'], quantities)}"
        for ix in idx_names)
    badge = range_label + (f' \u00b7 \u00d7{n}' if n is not None else '')
    sites = list(comp_def['occurrences'].keys())
    toggle_key = f"canvas-{name}"
    return f'''<div class="comp-section" data-comp="{html.escape(name, quote=True)}">
  <div class="comp-section-head" data-tree-toggle="{html.escape(toggle_key, quote=True)}">
    <span class="chev">\u25b8</span>
    <span class="name">{html.escape(name)}</span>
    <span class="comp-badge">{html.escape(badge)}</span>
    <span class="comp-sub">{len(sites)} occurrence(s) per instance \u2014 {html.escape(', '.join(sites))}</span>
  </div>
  <div class="comp-section-body" data-group="{html.escape(toggle_key, quote=True)}" style="display:none">{svg}</div>
</div>'''


def canvas_html(data):
    require_dot()
    quantities = data.get('quantities', {})
    nodes, edges, internal = build_graph(data)
    top_svg = render_svg(top_level_dot(data, nodes, edges, quantities))
    sections = ''.join(
        comp_section_html(name, comp_def, internal.get(name, []), quantities)
        for name, comp_def in data['compositions'].items())
    return f'<div class="top-graph">{top_svg}</div>\n<div class="comp-sections">{sections}</div>'


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  /* The model view is one page of the documentation site, so it carries the
     same design as the rest of it: the palette, the type and the hairline
     rules of docs/style/catalog.css, restated here because this page has to
     stay self-contained. Only the webfonts are fetched; every stack falls
     back to a system face when they are not there. */
  :root {
    --bg: #f7f5f0;
    --bg-raised: #ffffff;
    --bg-tint: #efe9dc;
    --ink: #22211e;
    --ink-2: #3d3a34;
    --ink-3: #55504a;
    --muted: #6f6a60;
    --faint: #8a847a;
    --rule: #e1ddd3;
    --rule-strong: #22211e;
    --chip: #d5cfc2;
    --accent: #9a4a1c;
    --accent-dark: #6e3311;
    --note-label: #7a5a2e;
    --serif: "Newsreader", Georgia, "Times New Roman", serif;
    --sans: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
    --mono: "IBM Plex Mono", "SFMono-Regular", Menlo, Consolas, monospace;
    --nav-w: 264px;
    --insp-w: 340px;
    --bar-h: 52px;
  }

  /* ---------- base ---------- */
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body {
    background: var(--bg);
    color: var(--ink);
    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { color: var(--accent-dark); text-decoration: underline; }
  ::selection { background: #e9d2bf; }

  /* ---------- site navigation ---------- */
  /* The site's own bar, docs/style/nav.html, put here by tools/site.sh; a page
     rendered on its own keeps the wordmark and drops the links, which would
     lead nowhere. */
  .sitebar {
    display: flex; align-items: center; gap: 30px; flex-shrink: 0;
    height: var(--bar-h); padding: 0 32px;
    border-bottom: 1px solid var(--rule);
  }
  .sitebar .wordmark {
    flex-shrink: 0;
    font-family: var(--serif); font-size: 19px; font-weight: 600; letter-spacing: -0.01em;
    color: var(--ink);
  }
  .sitebar .wordmark:hover { color: var(--accent); text-decoration: none; }
  .sitebar .sitekind { font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
  .sitebar .sitelinks { display: flex; align-self: stretch; gap: 20px; overflow-x: auto; scrollbar-width: none; }
  .sitebar .sitelinks::-webkit-scrollbar { display: none; }
  .sitebar .sitelinks a {
    display: flex; align-items: center; white-space: nowrap;
    margin-bottom: -1px; border-bottom: 2px solid transparent;
    font-size: 13.5px; color: var(--ink-3);
  }
  .sitebar .sitelinks a:hover { color: var(--ink); text-decoration: none; }
  .sitebar .sitelinks a.current { color: var(--ink); font-weight: 600; border-bottom-color: var(--accent); }

  /* ---------- page frame ---------- */
  /* Sidebar, then the tool: the same 264px column, the same hairline against
     the content, as every other page of the site. */
  .page { display: flex; flex-direction: row; height: calc(100vh - var(--bar-h)); overflow: hidden; }
  .nav {
    width: var(--nav-w);
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 32px 20px 0 32px;
    border-right: 1px solid var(--rule);
  }
  .content { flex-grow: 1; min-width: 0; display: flex; flex-direction: column; }

  /* ---------- sidebar ---------- */
  .search {
    display: flex; align-items: center; gap: 8px;
    height: 36px; padding: 0 12px;
    background: var(--bg-raised); border: 1px solid var(--rule); border-radius: 6px;
    color: var(--faint); font-size: 13.5px;
  }
  .search input {
    flex-grow: 1; min-width: 0; border: 0; outline: 0; background: transparent;
    font: inherit; color: var(--ink);
  }
  .search input::placeholder { color: var(--faint); }
  .search svg { flex-shrink: 0; }

  /* The tree stands where a document's table of contents stands, and reads
     the same way: an uppercase label per group, quiet entries beneath it. */
  .tree { flex-grow: 1; min-height: 0; overflow-y: auto; margin: 0 -10px; padding: 0 10px 28px; font-size: 13.5px; }
  .tree .grp {
    padding: 5px 10px 6px; margin-top: 22px;
    font-size: 11.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
  }
  .tree .grp:first-child { margin-top: 0; }
  .tree .grp em { font-style: normal; font-family: var(--mono); font-size: 10.5px;
                  letter-spacing: 0; text-transform: none; color: var(--faint); }
  .tree .row {
    display: flex; align-items: center; gap: 8px;
    padding: 4px 10px; border-radius: 4px;
    color: var(--ink-3); cursor: pointer;
  }
  .tree .row:hover { background: var(--bg-tint); color: var(--ink); }
  .tree .row.child { padding-left: 26px; }
  .tree .row.selected { background: var(--bg-tint); color: var(--ink); font-weight: 600; box-shadow: inset 2px 0 0 var(--accent); }
  .tree .row.search-hide { display: none !important; }
  .tree .chev { flex-shrink: 0; width: 9px; font-size: 9px; color: var(--faint); cursor: pointer; }
  .swatch { width: 7px; height: 7px; border-radius: 2px; flex-shrink: 0; }
  .sw-occ { background: var(--ink-3); }
  .sw-comp { background: var(--accent); }
  .sw-state { background: var(--note-label); }
  .sw-io { background: var(--faint); }
  .count { margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--faint); }

  /* ---------- content header ---------- */
  .topbar {
    display: flex; align-items: center; gap: 16px; flex-shrink: 0;
    height: 56px; padding: 0 28px 0 32px;
    border-bottom: 1px solid var(--rule);
  }
  .doc-title { min-width: 0; font-size: 13.5px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .doc-title b { font-family: var(--mono); font-size: 15px; font-weight: 500; color: var(--ink); }
  .doc-title .schema { font-family: var(--mono); font-size: 12px; color: var(--faint); margin-left: 10px; }
  .topbar-right { margin-left: auto; display: flex; align-self: stretch; align-items: center; gap: 22px; }
  .catalog-tag { font-family: var(--mono); font-size: 11px; color: var(--faint); white-space: nowrap; }
  .tabs { display: flex; align-self: stretch; gap: 18px; }
  .tab {
    display: flex; align-items: center; margin-bottom: -1px;
    border-bottom: 2px solid transparent;
    font-size: 11.5px; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted); cursor: pointer; user-select: none;
  }
  .tab:hover { color: var(--ink-3); }
  .tab.on { color: var(--ink); border-bottom-color: var(--accent); }

  .body-row { display: flex; flex-grow: 1; min-height: 0; }

  /* ---------- canvas ---------- */
  /* The canvas holds diagrams RENDERED BY GRAPHVIZ (dot -Tsvg) at generation
     time: nothing here computes layout in the browser. A composition is one
     box in the top diagram; its own internal structure is a second,
     separately laid-out diagram in a collapsible section below. */
  .canvas {
    flex-grow: 1; min-width: 0; overflow: auto;
    padding: 34px 36px 64px;
    background-image: radial-gradient(var(--rule) 1px, transparent 1px);
    background-size: 22px 22px;
  }
  .top-graph { display: flex; justify-content: center; }
  /* Graphviz writes the layout face into every label. Swap it for the site's
     own faces — the layout used their metric-compatible fallbacks — and take
     the bold of a node's name down to the medium weight the pages load. */
  #canvas svg text[font-family^="Courier"] { font-family: var(--mono); }
  #canvas svg text[font-family^="Helvetica"] { font-family: var(--sans); }
  #canvas svg text[font-weight="bold"] { font-weight: 500; }
  #canvas g.node { cursor: pointer; }
  #canvas g.node:hover > path, #canvas g.node:hover > ellipse, #canvas g.node:hover > polygon { stroke: var(--ink-3); }
  #canvas g.node.selected > path, #canvas g.node.selected > ellipse, #canvas g.node.selected > polygon {
    stroke: var(--accent); stroke-width: 1.8px;
  }

  .comp-sections { display: flex; flex-direction: column; gap: 16px; max-width: 960px; margin: 40px auto 0; }
  .comp-section { border: 1px solid var(--rule); border-radius: 6px; background: var(--bg-raised); overflow: hidden; }
  .comp-section-head { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; padding: 12px 18px; cursor: pointer; }
  .comp-section-head:hover { background: var(--bg); }
  .comp-section-head .chev { font-size: 10px; color: var(--faint); }
  .comp-section-head .name { font-family: var(--mono); font-size: 14px; font-weight: 500; color: var(--accent); }
  .comp-section-head .comp-badge {
    font-size: 11.5px; padding: 2px 9px; border: 1px solid var(--chip); border-radius: 999px;
    color: var(--ink-3); white-space: nowrap;
  }
  .comp-section-head .comp-sub { flex-basis: 100%; font-size: 12px; color: var(--muted); }
  .comp-section-body { display: flex; justify-content: center; padding: 22px; border-top: 1px solid var(--rule); background: var(--bg); }

  /* ---------- the second reading: the derived products ---------- */
  /* Nothing new in the palette: --note-label, already the colour of an index
     range on the diagrams, is the colour of a derived figure everywhere. */
  .dnum {
    flex-shrink: 0; padding: 1px 5px; border-radius: 3px; background: var(--bg-tint);
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0; text-transform: none;
    color: var(--note-label);
  }

  /* The tree reads the model two ways; this picks which. */
  .modeseg { display: flex; flex-shrink: 0; height: 30px; padding: 2px; background: var(--bg-raised); border: 1px solid var(--rule); border-radius: 6px; }
  .modeseg div {
    flex-grow: 1; display: flex; align-items: center; justify-content: center; border-radius: 4px;
    font-size: 11.5px; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted); cursor: pointer; user-select: none;
  }
  .modeseg div:hover { color: var(--ink-3); }
  .modeseg div.on { background: var(--bg-tint); color: var(--ink); font-weight: 600; }
  .modeseg div.disabled, .modeseg div.disabled:hover { color: var(--chip); cursor: default; }
  .tree .row.child2 { padding-left: 40px; }
  .tree .row .lbl { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .tree .figure { margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--note-label); }
  .tree .row.quiet { color: var(--faint); cursor: default; }
  .tree .row.quiet:hover { background: transparent; color: var(--faint); }

  /* Says whether a derived document was built for this model. */
  .derived-tag { display: flex; align-items: center; gap: 6px; padding: 3px 9px; border: 1px solid var(--chip); border-radius: 999px; font-size: 11.5px; color: var(--ink-3); white-space: nowrap; }
  .derived-tag i { width: 6px; height: 6px; border-radius: 999px; background: var(--note-label); }
  .derived-tag.absent { color: var(--faint); }
  .derived-tag.absent i { background: var(--chip); }

  /* One row above the diagram: which derived quantity is painted on the
     nodes, and which legal cut is marked across them. */
  .measurebar { display: flex; align-items: center; gap: 10px; flex-shrink: 0; height: 42px; padding: 0 28px 0 32px; border-bottom: 1px solid var(--rule); background: var(--bg-raised); }
  .mlabel { flex-shrink: 0; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  .mchip { flex-shrink: 0; padding: 3px 10px; border: 1px solid var(--chip); border-radius: 999px; font-size: 11.5px; color: var(--ink-3); cursor: pointer; user-select: none; white-space: nowrap; }
  .mchip:hover { border-color: var(--ink-3); }
  .mchip.on { background: var(--bg-tint); border-color: var(--note-label); color: var(--note-label); font-weight: 500; }
  .mchip em { font-style: normal; font-family: var(--mono); font-size: 11px; opacity: 0.75; }
  .mright { margin-left: auto; display: flex; align-items: center; gap: 10px; position: relative; }
  .cutsel { display: flex; align-items: center; gap: 8px; max-width: 280px; height: 26px; padding: 0 10px; background: var(--bg); border: 1px solid var(--rule); border-radius: 6px; font-family: var(--mono); font-size: 11.5px; color: var(--ink-2); cursor: pointer; user-select: none; }
  .cutsel .nm { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cutsel .car { flex-shrink: 0; color: var(--faint); }
  .cutsel.on { border-color: var(--accent); color: var(--accent); }
  .cutmenu { position: absolute; top: 32px; right: 0; z-index: 6; width: 300px; max-height: 340px; overflow-y: auto; padding: 6px; background: var(--bg-raised); border: 1px solid var(--rule); border-radius: 6px; box-shadow: 0 6px 18px rgba(34, 33, 30, 0.10); }
  .cutmenu .h { padding: 5px 8px 7px; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--rule); }
  .cutmenu .r { display: flex; justify-content: space-between; gap: 10px; padding: 6px 8px; border-radius: 4px; font-family: var(--mono); font-size: 11.5px; color: var(--ink-2); cursor: pointer; }
  .cutmenu .r span { flex-shrink: 0; color: var(--faint); }
  .cutmenu .r:hover { background: var(--bg); }
  .cutmenu .r.on { background: var(--bg-tint); color: var(--accent); }

  /* A measure painted into the node Graphviz already laid out: the gauge is
     drawn inside the box, the figure goes in the label row reserved for it. */
  #canvas g.node text:nth-of-type(n+2):last-of-type { visibility: hidden; }
  #canvas.measured g.node text:nth-of-type(n+2):last-of-type { visibility: visible; }
  #canvas .gauge-track { fill: var(--rule); }
  #canvas .gauge-fill { fill: var(--note-label); }
  /* A cut marked across the diagram: a node whose expanded occurrences are all
     on the near side, some of them, or none. */
  #canvas g.node.cut-in > path, #canvas g.node.cut-in > polygon { stroke: var(--accent); stroke-width: 1.6px; }
  #canvas g.node.cut-part > path, #canvas g.node.cut-part > polygon { stroke: var(--accent); stroke-width: 1.6px; stroke-dasharray: 5 3; }
  #canvas g.node.cut-out { opacity: 0.4; }

  /* ---------- inspector ---------- */
  /* Field rows read like the reference catalog's tables: an uppercase heading
     over a strong rule, then hairline-separated rows. */
  .inspector { width: var(--insp-w); flex-shrink: 0; overflow-y: auto; border-left: 1px solid var(--rule); padding: 30px 26px 48px; }
  .insp-empty { color: var(--faint); font-size: 13px; }
  .insp-title h3 { margin: 0; font-family: var(--mono); font-size: 19px; font-weight: 400; line-height: 1.25; color: var(--ink); word-break: break-word; }
  .insp-kind { margin: 6px 0 18px; font-size: 11.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  h4.insp {
    display: flex; align-items: baseline; gap: 8px;
    margin: 24px 0 2px; padding-bottom: 8px;
    font-family: var(--sans); font-size: 11.5px; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--rule-strong);
  }
  h4.insp code { font-family: var(--mono); font-size: 12px; letter-spacing: 0; text-transform: none; color: var(--ink-3); }
  .field { display: flex; justify-content: space-between; gap: 14px; padding: 9px 0; border-bottom: 1px solid var(--rule); font-size: 13px; }
  .field .k { min-width: 68px; color: var(--muted); overflow-wrap: break-word; }
  .field .v { min-width: 0; font-family: var(--mono); font-size: 12.5px; color: var(--ink-2); text-align: right; overflow-wrap: anywhere; }
  .contract-chip {
    display: inline-block; margin-bottom: 10px; padding: 3px 9px;
    background: var(--bg-tint); border-radius: 4px;
    font-family: var(--mono); font-size: 12px; color: var(--ink-2);
  }
  .fam-chip {
    display: inline-block; margin: 0 6px 6px 0; padding: 2px 9px;
    border: 1px solid var(--chip); border-radius: 999px;
    font-size: 11.5px; color: var(--ink-3);
  }
  .members { display: flex; flex-direction: column; gap: 6px; }
  .member-row {
    display: flex; justify-content: space-between; gap: 12px;
    padding: 8px 11px; background: var(--bg-raised); border: 1px solid var(--rule); border-radius: 4px;
    font-size: 12.5px; cursor: pointer;
  }
  .member-row:hover { border-color: var(--ink-3); }
  .member-row .n { font-family: var(--mono); color: var(--accent); }
  .member-row .c { font-size: 12px; color: var(--muted); }
  /* Same shape as a maintainer's note in the documents. */
  .state-box { margin-top: 10px; background: var(--bg-tint); border-radius: 4px; padding: 12px 14px; font-size: 13px; line-height: 1.65; color: var(--ink-2); }
  .state-box b { font-weight: 500; color: var(--note-label); }

  /* ---------- JSON ---------- */
  .json-peek {
    margin-top: 10px; padding: 12px 14px;
    background: var(--bg-raised); border: 1px solid var(--rule); border-radius: 4px;
    font-family: var(--mono); font-size: 11.5px; line-height: 1.6; color: var(--ink-2);
    white-space: pre-wrap; word-break: break-word; max-height: 340px; overflow: auto;
  }
  .jkey { color: var(--ink); font-weight: 500; }
  .jstr { color: var(--accent); }
  .jnum { color: var(--note-label); }
  .jbool, .jnull { color: var(--muted); }

  /* The inspector's derived half: one block per product, under the
     declaration, for whatever is selected. */
  .instsel { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; padding: 5px 9px; background: var(--bg-raised); border: 1px solid var(--rule); border-radius: 6px; font-family: var(--mono); font-size: 12px; color: var(--ink-2); }
  .instsel b { font-weight: 500; color: var(--accent); }
  .instsel .step { padding: 0 4px; color: var(--faint); cursor: pointer; user-select: none; }
  .instsel .step:hover { color: var(--accent); }
  .instsel .agg { margin-left: auto; padding: 1px 8px; border: 1px solid var(--chip); border-radius: 999px; font-family: var(--sans); font-size: 11px; color: var(--muted); cursor: pointer; user-select: none; }
  .instsel .agg.on { background: var(--bg-tint); border-color: var(--note-label); color: var(--note-label); }
  h4.insp .n { margin-left: auto; font-family: var(--mono); font-size: 11px; letter-spacing: 0; text-transform: none; color: var(--faint); }
  .drow { display: flex; flex-direction: column; gap: 1px; padding: 8px 10px; margin-top: 6px; background: var(--bg-raised); border: 1px solid var(--rule); border-radius: 4px; }
  .drow.pick { cursor: pointer; }
  .drow.pick:hover { border-color: var(--ink-3); }
  .drow .n { font-family: var(--mono); font-size: 12px; color: var(--accent); word-break: break-all; }
  .drow .m { display: flex; justify-content: space-between; gap: 10px; font-size: 11.5px; line-height: 1.5; color: var(--muted); }
  .drow .m b { flex-shrink: 0; font-family: var(--mono); font-weight: 400; color: var(--note-label); }
  .drow .m code { font-family: var(--mono); color: var(--ink-3); }
  .insp-none { padding: 8px 0 2px; font-size: 12.5px; color: var(--faint); }

  /* ---------- products pane ---------- */
  /* A product read whole, which the diagram cannot answer: what is big, what
     is shared, what is bounded. The inspector on the right does not change
     with the tab — whatever is selected is read in the same column. */
  .prodpane { display: none; flex-grow: 1; min-height: 0; }
  .prodpane .inspector { width: 304px; padding: 22px 20px 48px; }
  .prodrail { width: 212px; flex-shrink: 0; overflow-y: auto; padding: 22px 10px 24px 24px; border-right: 1px solid var(--rule); }
  .prodrail .lbl { padding: 0 10px 8px; font-size: 11.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  .prodrail .r { display: flex; align-items: center; gap: 8px; padding: 8px 9px; border-radius: 4px; font-size: 12.5px; white-space: nowrap; color: var(--ink-3); cursor: pointer; }
  .prodrail .r:hover { background: var(--bg-tint); color: var(--ink); }
  .prodrail .r.on { background: var(--bg-tint); color: var(--ink); font-weight: 600; box-shadow: inset 2px 0 0 var(--accent); }
  .prodrail .r.on .dnum { background: var(--bg-raised); }
  .prodrail .r .count { margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--faint); }
  .prodbody { flex-grow: 1; min-width: 0; overflow: auto; padding: 22px 26px 48px; }
  .prodhead { display: flex; align-items: baseline; gap: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--rule-strong); }
  .prodhead h3 { margin: 0; flex-shrink: 0; font-family: var(--serif); font-size: 21px; font-weight: 500; color: var(--ink); white-space: nowrap; }
  .prodhead .sub { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; color: var(--muted); }
  .prodhead .facts { margin-left: auto; flex-shrink: 0; padding-left: 16px; font-family: var(--mono); font-size: 11.5px; color: var(--note-label); }
  .ptotals { display: flex; flex-wrap: wrap; gap: 10px 34px; padding: 14px 0 16px; border-bottom: 1px solid var(--rule); }
  .ptotals div span { display: block; font-size: 10.5px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); }
  .ptotals div b { font-family: var(--serif); font-size: 19px; font-weight: 500; color: var(--ink); }
  .ptotals div.q b { color: var(--note-label); }
  .ptotals div.off b { color: var(--faint); }
  table.prod { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 12.5px; }
  table.prod th { padding: 12px 10px 7px 0; border-bottom: 1px solid var(--rule-strong); font-size: 10.5px; font-weight: 500; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); text-align: left; white-space: nowrap; }
  table.prod td { padding: 7px 10px 7px 0; border-bottom: 1px solid var(--rule); color: var(--ink-2); vertical-align: top; overflow: hidden; overflow-wrap: break-word; }
  table.prod td.m { font-family: var(--mono); }
  table.prod td.id { font-family: var(--mono); color: var(--accent); white-space: nowrap; text-overflow: ellipsis; }
  table.prod td.num { font-family: var(--mono); color: var(--note-label); text-align: right; white-space: nowrap; }
  table.prod tr.fold td.id { color: var(--ink); }
  table.prod tr.fold { cursor: pointer; }
  table.prod tr.kid td.id { padding-left: 20px; }
  table.prod tr.pick { cursor: pointer; }
  table.prod tr.pick:hover td { background: var(--bg-tint); }
  table.prod tr.on td { background: var(--bg-tint); }
  table.prod td .chev { display: inline-block; width: 11px; font-size: 9px; color: var(--faint); }
  table.prod td .x { margin-left: 7px; padding: 0 6px; border: 1px solid var(--chip); border-radius: 999px; font-family: var(--sans); font-size: 10.5px; color: var(--muted); }
  .foldnote { margin-top: 14px; font-size: 12px; color: var(--muted); }
  .foldnote code { font-family: var(--mono); color: var(--note-label); }
  .prodempty { padding: 26px 0; font-size: 13px; color: var(--faint); }
  .prodempty code { font-family: var(--mono); color: var(--ink-3); }

  .rawseg { display: flex; gap: 18px; max-width: 1100px; margin: 0 auto 14px; }
  .rawseg div { padding-bottom: 6px; border-bottom: 2px solid transparent; font-family: var(--mono); font-size: 12.5px; color: var(--muted); cursor: pointer; }
  .rawseg div.on { color: var(--ink); border-bottom-color: var(--accent); }
  .rawseg div.disabled, .rawseg div.disabled:hover { color: var(--chip); cursor: default; }

  .rawpane { display: none; flex-grow: 1; min-height: 0; overflow: auto; padding: 32px 36px 64px; }
  .rawpane pre {
    margin: 0 auto; max-width: 1100px; padding: 16px 20px; overflow-x: auto;
    background: var(--bg-raised); border: 1px solid var(--rule); border-radius: 4px;
    font-family: var(--mono); font-size: 12px; line-height: 1.6; color: var(--ink-2);
  }

  /* ---------- status bar ---------- */
  /* The counts are the facts strip of a contract page: a serif figure under a
     small uppercase label. */
  .statusbar {
    flex-shrink: 0; display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 28px;
    padding: 9px 28px 10px 32px; border-top: 1px solid var(--rule);
  }
  .statusbar span { font-size: 10.5px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); }
  .statusbar b {
    margin-right: 5px; font-family: var(--serif); font-size: 16px; font-weight: 500;
    letter-spacing: 0; text-transform: none; color: var(--ink);
  }
  .statusbar span.derived b { color: var(--note-label); }
  .statusbar span.selection { margin-left: auto; font-size: 12px; letter-spacing: 0; text-transform: none; }
  .statusbar span.selection code { font-family: var(--mono); color: var(--ink-2); }

  @media (max-width: 1200px) {
    :root { --nav-w: 228px; --insp-w: 300px; }
    .sitebar { gap: 20px; padding: 0 24px; }
    .nav { padding-left: 24px; }
    .topbar, .statusbar { padding-left: 24px; }
    .canvas, .rawpane { padding: 26px 24px 48px; }
    .prodrail { width: 200px; padding-left: 20px; }
    .prodbody { padding: 22px 20px 48px; }
    .measurebar { gap: 8px; padding: 0 20px 0 24px; }
    /* Past this width the sections start to run off the bar; fade the edge
       they scroll past rather than let them end mid-word. */
    .sitebar .sitelinks {
      -webkit-mask-image: linear-gradient(to right, #000 calc(100% - 28px), transparent);
      mask-image: linear-gradient(to right, #000 calc(100% - 28px), transparent);
    }
  }
  @media print {
    .sitebar, .nav, .topbar, .measurebar, .inspector, .statusbar { display: none; }
    .page, .content, .body-row { display: block; height: auto; overflow: visible; }
    .canvas { overflow: visible; padding: 0; background-image: none; }
    body { background: #fff; }
  }
</style>
</head>
<body>
__NAVBAR__
<div class="page">

  <aside class="nav">
    <label class="search">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"></circle><path d="M20 20l-3.5-3.5"></path></svg>
      <input type="search" id="search" placeholder="Filter" autocomplete="off">
    </label>
    <div class="modeseg" id="modeseg">
      <div data-mode="declared" class="on">Declared</div>
      <div data-mode="derived">Derived</div>
    </div>
    <nav class="tree" id="tree" aria-label="Model"></nav>
  </aside>

  <div class="content">

    <header class="topbar">
      <div class="doc-title"><b id="model-name"></b><span class="schema" id="model-schema"></span></div>
      <div class="topbar-right">
        <span class="catalog-tag" id="catalog-tag"></span>
        <span class="derived-tag" id="derived-tag"><i></i><span id="derived-tag-text"></span></span>
        <div class="tabs">
          <div class="tab on" data-tab="graph">Graph</div>
          <div class="tab" data-tab="products">Products</div>
          <div class="tab" data-tab="raw">Raw JSON</div>
        </div>
      </div>
    </header>

    <div class="measurebar" id="measurebar">
      <span class="mlabel">Measure</span>
      <span class="mchip on" data-measure="none">none</span>
      <span class="mchip" data-measure="parameters">parameters <em>D3</em></span>
      <span class="mchip" data-measure="state">cache / position <em>D4</em></span>
      <span class="mchip" data-measure="operations">operations / element <em>D5</em></span>
      <div class="mright">
        <span class="mlabel">Cut</span>
        <div class="cutsel" id="cutsel"><span class="nm">none</span><span class="car">&#9662;</span></div>
        <div class="cutmenu" id="cutmenu" style="display:none"></div>
      </div>
    </div>

    <div class="body-row">
      <div class="canvas" id="canvas">__CANVAS_HTML__</div>
      <aside class="inspector" id="inspector"><div class="insp-empty">Select a node to inspect it.</div></aside>
    </div>

    <div class="prodpane" id="prodpane">
      <aside class="prodrail" id="prodrail"></aside>
      <div class="prodbody" id="prodbody"></div>
      <aside class="inspector" id="inspector2"><div class="insp-empty">Select a row to inspect it.</div></aside>
    </div>

    <div class="rawpane" id="rawpane">
      <div class="rawseg" id="rawseg">
        <div class="on" data-raw="model"></div>
        <div data-raw="derived"></div>
      </div>
      <pre id="rawcode"></pre>
    </div>

    <div class="statusbar" id="statusbar"></div>

  </div>

</div>

<script>
const RAW = __MODEL_JSON__;
// The derived document \u2014 D1 to D6 \u2014 that `--derive` would write for this
// model: built into a temporary file, schema-checked there, and read back
// (see derive_to_tempfile). Null when it could not be built; DERIVED_NOTE
// then holds the emitter's reason, in its own words.
const DERIVED = __DERIVED_JSON__;
const DERIVED_NOTE = __DERIVED_NOTE__;
const HAS = DERIVED != null;

// ---------- escaping helpers ----------
function esc(s) { return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function escAttr(s) { return String(s).replace(/&/g,'&amp;').replace(/'/g,'&#39;').replace(/"/g,'&quot;'); }

function highlightJson(text) {
  const e = esc(text);
  return e.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false)\b|\bnull\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
    m => {
      if (/^"/.test(m)) return `<span class="${/:$/.test(m) ? 'jkey' : 'jstr'}">${m}</span>`;
      if (/true|false/.test(m)) return `<span class="jbool">${m}</span>`;
      if (m === 'null') return `<span class="jnull">${m}</span>`;
      return `<span class="jnum">${m}</span>`;
    });
}

// ---------- scalar expression rendering (tensorspine/2.0 §2.2) ----------
// Used only by the inspector panel below — the diagrams themselves are
// pre-rendered SVG from Graphviz, generated by this file's Python half.
function litValue(name) {
  const q = RAW.quantities[name];
  if (q && q.source && q.source.kind === 'literal') return q.source.value;
  return undefined;
}

function exprStr(e) {
  if (e == null) return '?';
  if ('literal' in e) return typeof e.literal === 'string' ? JSON.stringify(e.literal) : String(e.literal);
  if ('quantity' in e) { const v = litValue(e.quantity); return v !== undefined ? String(v) : e.quantity; }
  if ('index' in e) return e.index;
  if ('op' in e) {
    const a = (e.args || []).map(exprStr);
    if (e.op === 'negate') return `\u2212${a[0]}`;
    if (e.op === 'absolute') return `|${a[0]}|`;
    if (e.op === 'min' || e.op === 'max') return `${e.op}(${a.join(', ')})`;
    const sym = { add: '+', multiply: '\u00d7', subtract: '\u2212', divide: '\u00f7',
                  ceil_divide: '\u2308\u00f7\u2309', floor_divide: '\u230a\u00f7\u230b', modulo: 'mod' };
    return a.join(` ${sym[e.op] || e.op} `);
  }
  if ('if' in e) return `if ${condStr(e.if)} then ${exprStr(e.then)} else ${exprStr(e.else)}`;
  if ('call' in e) return `${e.call.contract.name}@${e.call.contract.version}(\u2026)`;
  return JSON.stringify(e);
}

function condStr(c) {
  if ('boolean' in c) return String(c.boolean);
  if ('not' in c) return `not (${condStr(c.not)})`;
  if ('all' in c) return c.all.map(condStr).join(' and ');
  if ('any' in c) return c.any.map(condStr).join(' or ');
  const cp = c.compare;
  const sym = { equal: '=', not_equal: '\u2260', less: '<', less_or_equal: '\u2264', greater: '>', greater_or_equal: '\u2265' };
  return `${exprStr(cp.left)} ${sym[cp.operator]} ${exprStr(cp.right)}`;
}

function domainStr(d) {
  if (!d) return '';
  if (d.kind === 'interval') {
    const lo = d.lower ? `${d.lower.inclusive ? '[' : '('}${exprStr(d.lower.value)}` : '(\u2212\u221e';
    const hi = d.upper ? `${exprStr(d.upper.value)}${d.upper.inclusive ? ']' : ')'}` : '\u221e)';
    return `${lo}, ${hi}`;
  }
  if (d.kind === 'set') return `{${d.values.map(v => JSON.stringify(v)).join(', ')}}`;
  return '';
}

function argValueStr(v) {
  if (v && typeof v === 'object' && 'record' in v) {
    return `{ ${Object.entries(v.record).map(([k, val]) => `${k}: ${argValueStr(val)}`).join(', ')} }`;
  }
  return exprStr(v);
}

function evalStatic(e) {
  if (e == null) return undefined;
  if ('literal' in e) return e.literal;
  if ('quantity' in e) return litValue(e.quantity);
  if ('op' in e) {
    const a = (e.args || []).map(evalStatic);
    if (a.some(v => v === undefined)) return undefined;
    switch (e.op) {
      case 'add': return a.reduce((x, y) => x + y, 0);
      case 'multiply': return a.reduce((x, y) => x * y, 1);
      case 'subtract': return a[0] - a[1];
      case 'divide': return a[0] / a[1];
      case 'ceil_divide': return Math.ceil(a[0] / a[1]);
      case 'floor_divide': return Math.floor(a[0] / a[1]);
      case 'modulo': return a[0] % a[1];
      case 'min': return Math.min(...a);
      case 'max': return Math.max(...a);
      case 'negate': return -a[0];
      case 'absolute': return Math.abs(a[0]);
    }
  }
  return undefined;
}

function compInstanceCount(c) {
  let total = 1;
  for (const ix of Object.keys(c.indices)) {
    const r = c.indices[ix];
    const start = evalStatic(r.start), stop = evalStatic(r.stop), step = evalStatic(r.step);
    if ([start, stop, step].some(v => v === undefined)) return undefined;
    total *= Math.max(0, Math.ceil((stop - start) / step));
  }
  return total;
}

function topKey(sel) { return sel.kind === 'root' ? `root:${sel.occurrence}` : `comp:${sel.composition}`; }

// Precise label for one specific occurrence selector (unlike topKey, which
// collapses every site of a composition to the composition itself — correct
// for the coarse top-level diagram, wrong for naming a binding member).
function occSelLabel(sel) {
  if (sel.kind === 'root') return sel.occurrence;
  const idx = Object.entries(sel.indices || {}).map(([k, v]) => `${k}=${exprStr(v)}`).join(',');
  return `${sel.composition}/${sel.occurrence}[${idx}]`;
}

function developedCount() {
  let total = Object.keys(RAW.occurrences).length, exact = true;
  for (const c of Object.values(RAW.compositions)) {
    const n = compInstanceCount(c), sites = Object.keys(c.occurrences).length;
    if (n === undefined) { exact = false; continue; }
    total += n * sites;
  }
  return { total, exact };
}

// ---------- state bindings: lookup by occurrence ----------
function findStateBindingsFor(matcher) {
  const out = [];
  for (const [sid, b] of Object.entries(RAW.bindings.states)) if (b.members.some(m => matcher(m.occurrence))) out.push([sid, b]);
  return out;
}
const matcherRoot = name => occ => occ.kind === 'root' && occ.occurrence === name;
const matcherSite = (comp, site) => occ => occ.kind === 'generated' && occ.composition === comp && occ.occurrence === site;

// ---------- mapping between a diagram node's SVG id and a selection object ----------
// Ids are assigned by the Python half when it builds the DOT sources:
// "root:<name>", "comp:<name>", "in:<name>", "out:<name>" in the top diagram,
// "<comp>::<site>" inside a composition's own diagram.
function selectionForNodeId(id) {
  if (id.startsWith('root:')) return { kind: 'root', name: id.slice(5) };
  if (id.startsWith('comp:')) return { kind: 'composition', name: id.slice(5) };
  if (id.startsWith('in:')) return { kind: 'interface', dir: 'inputs', name: id.slice(3) };
  if (id.startsWith('out:')) return { kind: 'interface', dir: 'outputs', name: id.slice(4) };
  const sep = id.indexOf('::');
  if (sep !== -1) return { kind: 'site', comp: id.slice(0, sep), site: id.slice(sep + 2) };
  return null;
}

function nodeIdForSelection(sel) {
  if (sel.kind === 'root') return `root:${sel.name}`;
  if (sel.kind === 'composition') return `comp:${sel.name}`;
  if (sel.kind === 'site') return `${sel.comp}::${sel.site}`;
  if (sel.kind === 'interface') return sel.dir === 'inputs' ? `in:${sel.name}` : `out:${sel.name}`;
  return null;
}

// ---------- the derived document ----------
// Everything below reads DERIVED, never recomputes it. The one exception is
// stated where it happens: the per-node share of D5's operations, which the
// document reports as a model total only, and which is offered only when the
// shares add back up to that total.

const D1 = HAS ? DERIVED.d1 : null;
const NODE_IDS = HAS ? Object.keys(D1.nodes) : [];

function fmtInt(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ' '); }

function fmtBytes(n) {
  if (n == null) return '—';
  if (n === 0) return '0';
  if (n >= 2 ** 30) return `${(n / 2 ** 30).toFixed(2)} GiB`;
  if (n >= 2 ** 20) return `${(n / 2 ** 20).toFixed(1)} MiB`;
  if (n >= 1024) return `${(n / 1024).toFixed(n >= 10240 ? 0 : 1)} KiB`;
  return `${fmtInt(n)} B`;
}

function fmtOps(n) {
  if (n == null) return '—';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} Gop`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} Mop`;
  return `${fmtInt(n)} op`;
}

function fmtShape(shape) {
  return (shape || []).map(a => fmtInt(a.extent)).join(' × ') || '—';
}

function shapeAxes(shape) {
  return (shape || []).map(a => `${a.axis} ${fmtInt(a.extent)}`).join(' × ');
}

function fmtCounts(count) {
  return Object.entries(count || {}).map(([k, v]) => `${k} ${v}`).join(' + ') || '—';
}

// A member is "<node>.<slot>"; a node identifier never carries a dot of its
// own (§5.2 rule 2: the index sits in brackets), so the last one splits it.
function nodeOfMember(member) {
  const i = member.lastIndexOf('.');
  return i < 0 ? member : member.slice(0, i);
}

// The expanded nodes one thing the model declares stands for: a root
// occurrence is itself, a composition is every occurrence it generated, a
// site is every instance of that site.
function nodesForSelection(sel) {
  if (!HAS) return [];
  if (sel.kind === 'root') return NODE_IDS.filter(id => id === sel.name);
  if (sel.kind === 'composition') return NODE_IDS.filter(id => id.startsWith(sel.name + '/'));
  if (sel.kind === 'site') return NODE_IDS.filter(id => id.startsWith(`${sel.comp}/${sel.site}[`));
  return [];
}

function nodeIndex(id) {
  const m = /\[([^\]]*)\]$/.exec(id);
  return m ? m[1] : null;
}

// A listing folded along the model's own index: the 32 rows of
// decoder.attn.q[layer=0…31] are one row that opens into 32.
function foldOf(id) {
  const m = /^(.*)\[([^\]]*)\]$/.exec(id);
  if (!m) return { key: id, index: null };
  const names = m[2].split(',').map(part => part.split('=')[0].trim());
  return { key: `${m[1]}[${names.join(',')}]`, index: m[2] };
}

function foldGroups(ids) {
  const order = [], byKey = new Map();
  for (const id of ids) {
    const { key } = foldOf(id);
    if (!byKey.has(key)) { byKey.set(key, []); order.push(key); }
    byKey.get(key).push(id);
  }
  return order.map(key => {
    const members = byKey.get(key);
    return { key, members, label: members.length > 1 ? foldLabel(key, members) : members[0] };
  });
}

function foldLabel(key, members) {
  const m = /^(.*)\[([^\]]*)\]$/.exec(key);
  if (!m) return `${key} ×${members.length}`;
  const names = m[2].split(',');
  const spans = names.map((name, i) => {
    const values = members.map(id => {
      const inner = /\[([^\]]*)\]$/.exec(id);
      const part = inner ? inner[1].split(',')[i] : null;
      return part ? part.split('=').slice(1).join('=') : null;
    }).filter(v => v != null);
    const nums = values.map(Number);
    if (values.length && nums.every(n => Number.isFinite(n))) {
      const lo = Math.min(...nums), hi = Math.max(...nums);
      return `${name}=${lo === hi ? lo : `${lo}…${hi}`}`;
    }
    return name;
  });
  return `${m[1]}[${spans.join(',')}]`;
}

// ---------- what each product says about a set of nodes ----------
function tensorsOf(nodes) {
  if (!HAS) return [];
  const set = new Set(nodes);
  return DERIVED.d3.tensors.filter(t => t.members.some(m => set.has(nodeOfMember(m))));
}
function statesOf(nodes) {
  if (!HAS) return [];
  const set = new Set(nodes);
  return DERIVED.d4.states.filter(t => t.members.some(m => set.has(nodeOfMember(m))));
}
function correctionsOf(nodes) {
  if (!HAS) return [];
  const set = new Set(nodes);
  return (DERIVED.d5.corrections || []).filter(c => set.has(c.node));
}
function partitionsOf(nodes) {
  if (!HAS) return [];
  const set = new Set(nodes);
  return (DERIVED.d6.partitions || []).filter(x => set.has(x.node));
}
function lossOf(nodes) {
  if (!HAS) return [];
  const set = new Set(nodes);
  return (DERIVED.d6.information_loss || []).filter(x => set.has(x.node));
}

function partitionTarget(t) {
  if (!t || typeof t !== 'object') return String(t);
  if (t.argument_axis) return t.argument_axis;
  if (t.instance_key_axis) return t.instance_key_axis;
  if (t.payload_axis) return `${t.payload_axis.state}.${t.payload_axis.component} · ${t.payload_axis.axis}`;
  if (t.any_axis) return 'any axis';
  if (t.none) return 'none';
  return Object.keys(t)[0] || '—';
}

// ---------- measures ----------
// Bytes are attributed to every node that reads the tensor, so a tied tensor
// is shown at each of its members; D3's own total counts it once, and that is
// the figure the status bar carries.
function bytesByNode() {
  const out = {};
  for (const t of DERIVED.d3.tensors) {
    for (const m of t.members) out[nodeOfMember(m)] = (out[nodeOfMember(m)] || 0) + t.bytes;
  }
  return out;
}

function stateBytesByNode() {
  const out = {};
  for (const st of DERIVED.d4.states) {
    const b = st.bytes_per_cached_position != null ? st.bytes_per_cached_position
            : (st.bytes_bounded != null ? st.bytes_bounded : 0);
    for (const m of st.members) out[nodeOfMember(m)] = (out[nodeOfMember(m)] || 0) + b;
  }
  return out;
}

// The one figure this page derives rather than reads: D5 reports operations
// per element for the model, not per node. The inventory rule of §4.1 splits
// it — two operations per weight element per element, at the activated
// fraction of a sparsity unit — over the SLOTS a tensor satisfies, not over
// the tensors: a tied tensor is resident once but read at each member.
//
// The sparsity unit belongs to a contract (§4.5), and D3 records the unit of
// its first member only, so the fraction applies to a member whose node runs
// that contract and to no other: a table tied between an `embed` lookup and
// an `lm_head` projection is sparse at the lookup and dense at the
// projection. Element corrections land in the same total (§4.1).
//
// The split is offered only when it adds back up to what D5 says.
function operationsByNode() {
  const out = {};
  let total = 0;
  const add = (node, ops) => { total += ops; out[node] = (out[node] || 0) + ops; };
  for (const t of DERIVED.d3.tensors) {
    const fraction = t.sparsity && t.sparsity.activated_fraction != null
      ? t.sparsity.activated_fraction : 1;
    for (const m of t.members) {
      const node = nodeOfMember(m);
      const sparse = D1.nodes[node] && D1.nodes[node].contract.name === t.contract;
      add(node, 2 * t.elements * (sparse ? fraction : 1));
    }
  }
  for (const c of DERIVED.d5.corrections || []) if (c.per === 'element') add(c.node, c.value);
  const reported = ((DERIVED.d5.operations || {}).element || {}).value;
  const agrees = reported != null && (reported === 0 ? total === 0
                                      : Math.abs(total - reported) <= Math.max(1, reported * 1e-9));
  return { byNode: out, agrees };
}

const MEASURES = {
  parameters: { label: 'parameters', product: 'D3', fmt: fmtBytes },
  state: { label: 'cache / position', product: 'D4', fmt: fmtBytes },
  operations: { label: 'operations / element', product: 'D5', fmt: fmtOps },
};

let measureValues = null;      // { name -> {byNode, ok} }, built once on load

function buildMeasures() {
  if (!HAS) return null;
  const ops = operationsByNode();
  return {
    parameters: { byNode: bytesByNode(), ok: true },
    state: { byNode: stateBytesByNode(), ok: true },
    operations: { byNode: ops.byNode, ok: ops.agrees },
  };
}

// A diagram node stands for a set of expanded nodes; its figure is their sum.
function diagramMembers(nodeId) {
  if (nodeId.startsWith('root:')) { const n = nodeId.slice(5); return NODE_IDS.filter(id => id === n); }
  if (nodeId.startsWith('comp:')) { const n = nodeId.slice(5); return NODE_IDS.filter(id => id.startsWith(n + '/')); }
  const sep = nodeId.indexOf('::');
  if (sep !== -1) {
    const prefix = `${nodeId.slice(0, sep)}/${nodeId.slice(sep + 2)}[`;
    return NODE_IDS.filter(id => id.startsWith(prefix));
  }
  return [];
}

let currentMeasure = 'none';

function paintMeasure(name) {
  currentMeasure = name;
  document.querySelectorAll('.mchip').forEach(c => c.classList.toggle('on', c.dataset.measure === name));
  const spec = MEASURES[name];
  const table = spec && measureValues && measureValues[name].ok ? measureValues[name].byNode : null;
  document.getElementById('canvas').classList.toggle('measured', !!table);

  // The figure goes in the label row Graphviz reserved for it — always the
  // last <text> of the node — so nothing reflows when the measure changes.
  const figures = new Map();
  let peak = 0;
  document.querySelectorAll('#canvas g.node').forEach(g => {
    const members = diagramMembers(g.getAttribute('id'));
    if (!members.length) return;
    let v = 0;
    if (table) for (const id of members) v += table[id] || 0;
    figures.set(g, v);
    if (v > peak) peak = v;
  });

  document.querySelectorAll('#canvas .gauge').forEach(n => n.remove());
  document.querySelectorAll('#canvas g.node').forEach(g => {
    const texts = g.querySelectorAll('text');
    if (!texts.length || !figures.has(g)) return;
    const slot = texts[texts.length - 1];
    if (!table) { slot.textContent = ''; return; }
    const v = figures.get(g);
    slot.textContent = spec.fmt(v);
    if (!peak) return;
    const shape = g.querySelector('path, polygon, ellipse');
    if (!shape) return;
    const box = shape.getBBox();
    const x = box.x + 11, w = Math.max(0, box.width - 22);
    const y = box.y + box.height - 7.5;
    const NS = 'http://www.w3.org/2000/svg';
    const track = document.createElementNS(NS, 'rect');
    track.setAttribute('class', 'gauge gauge-track');
    track.setAttribute('x', x); track.setAttribute('y', y);
    track.setAttribute('width', w); track.setAttribute('height', 3); track.setAttribute('rx', 1.5);
    const fill = document.createElementNS(NS, 'rect');
    fill.setAttribute('class', 'gauge gauge-fill');
    fill.setAttribute('x', x); fill.setAttribute('y', y);
    fill.setAttribute('width', Math.max(v > 0 ? 1.5 : 0, w * (v / peak)));
    fill.setAttribute('height', 3); fill.setAttribute('rx', 1.5);
    g.appendChild(track); g.appendChild(fill);
  });
}

// ---------- legal cuts ----------
// The block of a cut is the ancestor closure of its seed set (§7): a layer cut
// seeds on a composition prefix, a family cut on a family. Computed from D1's
// own edges, and used only when its size is the size D2 reports.
function cutBlock(name) {
  if (!HAS) return null;
  let seeds;
  const layer = /^(.*)\[([A-Za-z_][A-Za-z0-9_]*)<=(-?\d+)\]$/.exec(name);
  if (layer) {
    const [, comp, index, bound] = layer;
    const re = new RegExp(`[\\[,]${index}=(-?\\d+)[,\\]]`);
    seeds = NODE_IDS.filter(id => {
      if (!id.startsWith(comp + '/')) return false;
      const m = re.exec(id);
      return m && Number(m[1]) <= Number(bound);
    });
  } else if (name.startsWith('family:')) {
    const family = name.slice(7);
    seeds = NODE_IDS.filter(id => (D1.nodes[id].families || []).includes(family));
  } else {
    return null;
  }
  const parents = new Map(NODE_IDS.map(id => [id, []]));
  for (const e of D1.edges) {
    const into = parents.get(e.to.node);
    if (into) into.push(e.from.node);
  }
  const seen = new Set(), stack = seeds.slice();
  while (stack.length) {
    const n = stack.pop();
    if (seen.has(n)) continue;
    seen.add(n);
    for (const src of parents.get(n) || []) if (!seen.has(src)) stack.push(src);
  }
  return seen;
}

let currentCut = null;

// D2 counts a template instance as one node at its caller's level (§5.2 rule
// 2), while D1 lists the occurrences inside it. When that makes this page's
// closure a different set from the one D2 counted, it marks nothing and says
// so — marking a set nobody counted would be worse than marking none.
function cutMarkable(name) {
  const cut = DERIVED.d2.cuts.find(c => c.cut === name);
  const block = cutBlock(name);
  if (!cut) return { block: null, why: 'no such cut in D2' };
  if (!block) return { block: null, why: 'this page cannot read the cut\u2019s seed set from its name' };
  if (block.size !== cut.sizes[0]) {
    return { block: null, why: (D1.instances || []).length
      ? 'D2 counts each template instance as one node (\u00a75.2 rule 2); D1 lists the occurrences inside it, so the two blocks are different sets'
      : `the closure of this page is ${block.size} nodes, D2 counted ${cut.sizes[0]}` };
  }
  return { block, why: null };
}

function markCut(name) {
  currentCut = name;
  const sel = document.querySelector('#cutsel .nm');
  if (sel) sel.textContent = name || 'none';
  document.getElementById('cutsel').classList.toggle('on', !!name);
  document.querySelectorAll('#canvas g.node').forEach(g => g.classList.remove('cut-in', 'cut-part', 'cut-out'));
  if (!name) return;
  const { block } = cutMarkable(name);
  if (!block) return;
  document.querySelectorAll('#canvas g.node').forEach(g => {
    const members = diagramMembers(g.getAttribute('id'));
    if (!members.length) return;
    const inside = members.filter(id => block.has(id)).length;
    g.classList.add(inside === members.length ? 'cut-in' : inside ? 'cut-part' : 'cut-out');
  });
}

function buildCutMenu() {
  const menu = document.getElementById('cutmenu');
  if (!HAS) return;
  const rows = DERIVED.d2.cuts.map(c =>
    `<div class="r" data-cut="${escAttr(c.cut)}"><span>${esc(c.cut)}</span><span>${esc(fmtBytes(c.bytes_per_element))}</span></div>`).join('');
  menu.innerHTML = `<div class="h">D6 · ${DERIVED.d2.cuts.length} legal cut(s)</div>` +
    `<div class="r" data-cut=""><span>none</span><span>—</span></div>${rows}`;
}

// ---------- side tree ----------
function buildTree() {
  const L = [];
  const grp = label => L.push(`<div class="grp">${esc(label)}</div>`);
  const row = (inner, selObj, opts = {}) => {
    const attrs = [];
    if (selObj) attrs.push(`data-select='${escAttr(JSON.stringify(selObj))}'`);
    if (opts.group) attrs.push(`data-group="${escAttr(opts.group)}"`);
    if (opts.toggle) attrs.push(`data-tree-toggle="${escAttr(opts.toggle)}"`);
    const style = opts.hidden ? ' style="display:none"' : '';
    L.push(`<div class="row${opts.child ? ' child' : ''}"${style} ${attrs.join(' ')}>${inner}</div>`);
  };

  grp('quantities');
  row(`quantities<span class="count">${Object.keys(RAW.quantities).length}</span>`, { kind: 'quantities' });

  grp('occurrences');
  const rootNames = Object.keys(RAW.occurrences);
  row(`<span class="chev" data-tree-toggle="occ">\u25be</span>occurrences<span class="count">${rootNames.length}</span>`);
  rootNames.forEach(n => row(`<span class="swatch sw-occ"></span>${esc(n)}`, { kind: 'root', name: n }, { child: true, group: 'occ' }));

  grp('compositions');
  for (const [cname, c] of Object.entries(RAW.compositions)) {
    const n = compInstanceCount(c);
    row(`<span class="chev" data-tree-toggle="comp-${escAttr(cname)}">\u25be</span>${esc(cname)}<span class="count">${n != null ? '\u00d7' + n : 'variable'}</span>`, { kind: 'composition', name: cname });
    Object.keys(c.occurrences).forEach(site =>
      row(`<span class="swatch sw-occ"></span>${esc(site)}`, { kind: 'site', comp: cname, site }, { child: true, group: `comp-${cname}` }));
  }

  grp('bindings');
  for (const which of ['values', 'parameters', 'constants', 'states']) {
    const ids = Object.keys(RAW.bindings[which]);
    row(`<span class="chev" data-tree-toggle="bind-${which}">\u25b8</span>${which}<span class="count">${ids.length}</span>`, { kind: 'bindingGroup', which });
    ids.forEach(id => row(`<span class="swatch sw-state"></span>${esc(id)}`, { kind: 'bindingItem', which, id }, { child: true, group: `bind-${which}`, hidden: true }));
  }

  grp('interfaces');
  Object.entries(RAW.interfaces.inputs).forEach(([n, spec]) =>
    row(`<span class="swatch sw-io"></span>${esc(n)} \u2192 ${esc(spec.to.map(e => occSelLabel(e.occurrence)).join(', '))}`, { kind: 'interface', dir: 'inputs', name: n }));
  Object.entries(RAW.interfaces.outputs).forEach(([n, spec]) =>
    row(`<span class="swatch sw-io"></span>${esc(occSelLabel(spec.from.occurrence))} \u2192 ${esc(n)}`, { kind: 'interface', dir: 'outputs', name: n }));

  return L.join('');
}

// ---------- the derived tree ----------
// The same sidebar, reading the other document: not what was declared but
// what it expands to. Counts are the products' own.
function derivedTree() {
  if (!HAS) {
    return `<div class="grp">products</div>` +
      `<div class="row quiet">no derived document</div>` +
      `<div class="row quiet" style="line-height:1.45">${esc(DERIVED_NOTE || '')}</div>`;
  }
  const L = [];
  const grp = (label, tail) => L.push(`<div class="grp">${esc(label)}${tail ? ` <em>${esc(tail)}</em>` : ''}</div>`);
  const row = (inner, selObj, opts = {}) => {
    const attrs = [];
    if (selObj) attrs.push(`data-select='${escAttr(JSON.stringify(selObj))}'`);
    if (opts.group) attrs.push(`data-group="${escAttr(opts.group)}"`);
    const cls = `row${opts.child ? ' child' : ''}${opts.deep ? ' child2' : ''}${opts.quiet ? ' quiet' : ''}`;
    L.push(`<div class="${cls}"${opts.hidden ? ' style="display:none"' : ''} ${attrs.join(' ')}>${inner}</div>`);
  };
  const head = (which, label, count, toggle) =>
    row(`<span class="chev" data-tree-toggle="${toggle}">▾</span><span class="dnum">${which.toUpperCase()}</span>${esc(label)}` +
        (count != null ? `<span class="count">${esc(String(count))}</span>` : ''),
        { kind: 'product', which });
  const kid = (label, figure, selObj, group) =>
    row(`<span class="lbl" title="${escAttr(label)}">${esc(label)}</span>` +
        (figure != null ? `<span class="${/[A-Za-z]/.test(String(figure)) ? 'figure' : 'count'}">${esc(String(figure))}</span>` : ''),
        selObj, { child: true, group });

  const d2 = DERIVED.d2, d3 = DERIVED.d3, d4 = DERIVED.d4, d5 = DERIVED.d5, d6 = DERIVED.d6;
  grp('products', DERIVED.model + '.derived.json');

  head('d1', 'expanded graph', Object.keys(D1.nodes).length, 'p-d1');
  kid('nodes', Object.keys(D1.nodes).length, { kind: 'product', which: 'd1' }, 'p-d1');
  kid('edges', D1.edges.length, { kind: 'product', which: 'd1' }, 'p-d1');
  kid('interfaces', Object.keys(D1.interfaces.inputs).length + Object.keys(D1.interfaces.outputs).length,
      { kind: 'product', which: 'd1' }, 'p-d1');

  head('d2', 'values & cuts', d2.values.length, 'p-d2');
  kid('streams', Object.keys(d2.streams).length, { kind: 'product', which: 'd2' }, 'p-d2');
  kid('values', d2.values.length, { kind: 'product', which: 'd2' }, 'p-d2');
  kid('cuts', d2.cuts.length, { kind: 'product', which: 'd2' }, 'p-d2');

  head('d3', 'parameter tensors', d3.totals.tensors, 'p-d3');
  for (const g of foldGroups(d3.tensors.map(t => t.identity)).slice(0, 8)) {
    const bytes = d3.tensors.filter(t => g.members.includes(t.identity)).reduce((a, t) => a + t.bytes, 0);
    kid(g.label, fmtBytes(bytes), { kind: 'd3', identity: g.members[0] }, 'p-d3');
  }
  kid('resident', fmtBytes(d3.totals.bytes), { kind: 'product', which: 'd3' }, 'p-d3');

  head('d4', 'states', d4.totals.identities, 'p-d4');
  for (const g of foldGroups(d4.states.map(t => t.identity)).slice(0, 8)) {
    kid(g.label, `×${g.members.length}`, { kind: 'd4', identity: g.members[0] }, 'p-d4');
  }
  kid('per cached position', fmtBytes(d4.totals.append_bytes_per_cached_position),
      { kind: 'product', which: 'd4' }, 'p-d4');

  head('d5', 'logical costs', null, 'p-d5');
  kid('parameters', fmtBytes(d5.parameters.bytes), { kind: 'product', which: 'd5' }, 'p-d5');
  kid('per element', fmtOps(d5.operations.element.value), { kind: 'product', which: 'd5' }, 'p-d5');
  kid('corrections', (d5.corrections || []).length, { kind: 'product', which: 'd5' }, 'p-d5');

  head('d6', 'cuts & partitions', null, 'p-d6');
  kid('legal cuts', d6.cuts.length, { kind: 'product', which: 'd6' }, 'p-d6');
  kid('partitions', (d6.partitions || []).length, { kind: 'product', which: 'd6' }, 'p-d6');
  kid('information loss', (d6.information_loss || []).length, { kind: 'product', which: 'd6' }, 'p-d6');

  return L.join('');
}

// ---------- the inspector's derived half ----------
// The model names one site; the products name every instance of it. This is
// the join: read one instance, or read them all added up.
const instState = { key: null, at: 0, aggregate: true };

function instanceControl(nodes) {
  if (nodes.length < 2) return '';
  const agg = instState.aggregate;
  const at = Math.min(instState.at, nodes.length - 1);
  const label = agg ? `${nodes.length} instances` : (nodeIndex(nodes[at]) || nodes[at]);
  return `<div class="instsel">` +
    (agg ? '' : `<span class="step" data-inst="-1">◂</span>`) +
    `<b>${esc(label)}</b>` +
    (agg ? '' : `<span class="step" data-inst="1">▸</span> of ${nodes.length}`) +
    `<span class="agg${agg ? ' on' : ''}" data-inst="agg">aggregate ×${nodes.length}</span></div>`;
}

function derivedSections(sel) {
  if (!HAS) {
    return `<h4 class="insp"><span class="dnum">D1–D6</span>derived</h4>` +
      `<div class="insp-none">No derived document for this model — ${esc(DERIVED_NOTE || 'not built')}.</div>`;
  }
  const all = nodesForSelection(sel);
  if (!all.length) return '';
  if (instState.key !== JSON.stringify(sel)) { instState.key = JSON.stringify(sel); instState.at = 0; instState.aggregate = true; }
  const at = Math.min(instState.at, all.length - 1);
  const nodes = (all.length > 1 && !instState.aggregate) ? [all[at]] : all;

  const L = [instanceControl(all)];
  const head = (which, label, tail) =>
    `<h4 class="insp"><span class="dnum">${which}</span>${esc(label)}${tail ? `<span class="n">${esc(tail)}</span>` : ''}</h4>`;
  const none = what => `<div class="insp-none">no ${what}</div>`;

  // D3
  const tensors = tensorsOf(nodes);
  const bytes = tensors.reduce((a, t) => a + t.bytes, 0);
  L.push(head('D3', 'parameter tensors', tensors.length ? `${tensors.length} · ${fmtBytes(bytes)}` : null));
  if (!tensors.length) L.push(none('tensor'));
  else if (nodes.length > 1) {
    for (const g of foldGroups(tensors.map(t => t.identity))) {
      const rows = tensors.filter(t => g.members.includes(t.identity));
      const t = rows[0];
      L.push(`<div class="drow pick" data-select='${escAttr(JSON.stringify({ kind: 'd3', identity: t.identity }))}'>
        <div class="n">${esc(g.label)}</div>
        <div class="m"><span>${esc(t.role)} · ${esc(t.dtype)}</span><b>${esc(fmtBytes(rows.reduce((a, x) => a + x.bytes, 0)))}</b></div>
        <div class="m"><span><code>${esc(fmtShape(t.shape))}</code> · ${esc(t.sensitivity)}${rows.length > 1 ? ` · ×${rows.length}` : ''}</span></div>
      </div>`);
    }
  } else {
    for (const t of tensors) L.push(`<div class="drow pick" data-select='${escAttr(JSON.stringify({ kind: 'd3', identity: t.identity }))}'>
      <div class="n">${esc(t.identity)}</div>
      <div class="m"><span>${esc(t.role)} · ${esc(t.dtype)}</span><b>${esc(fmtBytes(t.bytes))}</b></div>
      <div class="m"><span><code>${esc(fmtShape(t.shape))}</code> · ${esc(t.sensitivity)}${t.tied ? ' · tied' : ''}</span></div>
    </div>`);
  }

  // D4
  const states = statesOf(nodes);
  const perPos = states.reduce((a, x) => a + (x.bytes_per_cached_position || 0), 0);
  L.push(head('D4', 'states', states.length ? `${states.length} · ${fmtBytes(perPos)} / pos` : null));
  if (!states.length) L.push(none('state'));
  else {
    const groups = nodes.length > 1 ? foldGroups(states.map(x => x.identity))
                                    : states.map(x => ({ label: x.identity, members: [x.identity] }));
    for (const g of groups) {
      const rows = states.filter(x => g.members.includes(x.identity));
      const st = rows[0];
      L.push(`<div class="drow pick" data-select='${escAttr(JSON.stringify({ kind: 'd4', identity: st.identity }))}'>
        <div class="n">${esc(g.label)}</div>
        <div class="m"><span>${esc(st.law)} · ${esc(st.access)} · ${esc(st.sharing)}</span><b>${esc(fmtBytes(rows.reduce((a, x) => a + (x.bytes_per_cached_position || 0), 0)))}</b></div>
        <div class="m"><span>stream <code>${esc(st.stream.kind)} · ${esc(st.stream.stream)}</code>${st.carried_across_fragments ? ' · carried' : ''}</span></div>
        <div class="m"><span>key <code>${esc((st.instance_key || []).join(', '))}</code></span></div>
      </div>`);
    }
  }

  // D5
  const ops = measureValues && measureValues.operations.ok
    ? nodes.reduce((a, id) => a + (measureValues.operations.byNode[id] || 0), 0) : null;
  const corrections = correctionsOf(nodes);
  L.push(head('D5', 'logical cost'));
  if (ops != null) {
    const share = DERIVED.d5.operations.element.value;
    L.push(`<div class="field"><span class="k">operations / element</span><span class="v"><em>${esc(fmtOps(ops))}</em>${share ? ` · ${(100 * ops / share).toFixed(1)} %` : ''}</span></div>`);
  }
  for (const c of corrections.slice(0, 4)) {
    L.push(`<div class="field"><span class="k">correction / ${esc(c.per)}</span><span class="v"><em>+${esc(fmtInt(c.value))}</em> ${esc(c.status)}</span></div>`);
  }
  if (corrections.length > 4) L.push(`<div class="insp-none">and ${corrections.length - 4} more correction(s)</div>`);
  if (ops == null && !corrections.length) L.push(none('cost attributable to this node'));

  // D6
  const partitions = partitionsOf(nodes);
  const seen = new Set(), unique = [];
  for (const x of partitions) {
    const k = `${partitionTarget(x.target)}→${x.communication}`;
    if (!seen.has(k)) { seen.add(k); unique.push(x); }
  }
  L.push(head('D6', 'partitions', unique.length ? String(unique.length) : null));
  if (!unique.length) L.push(none('partition declared'));
  for (const x of unique) {
    L.push(`<div class="field"><span class="k">${esc(partitionTarget(x.target))}</span><span class="v">${esc(x.communication)}</span></div>`);
  }
  const loss = lossOf(nodes);
  if (loss.length) {
    const slots = [...new Set(loss.map(x => `${x.slot} · ${x.axis}`))];
    L.push(head('D6', 'information loss', String(loss.length)));
    L.push(`<div class="state-box"><b>${esc(slots.join(', '))}</b> — flattened axes with no declared factors (O5.10): partitionability along their factors is unknown, not absent.</div>`);
  }
  return L.join('');
}

// ---------- one entry of a product, in the inspector ----------
function d3Body(t) {
  const rows = [
    ['members', t.members.join(', ')],
    ['contract · slot', `${t.contract} · ${t.slot}`],
    ['role', t.role],
    ['sensitivity', t.sensitivity],
    ['dtype', t.dtype],
    ['shape', fmtShape(t.shape)],
    ['axes', (t.shape || []).map(a => a.axis).join(' × ')],
    ['elements', fmtInt(t.elements)],
    ['bytes', fmtBytes(t.bytes)],
    ['tied', String(!!t.tied)],
  ];
  if (t.multiplicity && t.multiplicity !== 1) rows.splice(6, 0, ['multiplicity', String(t.multiplicity)]);
  let out = `<h4 class="insp"><span class="dnum">D3</span>parameter tensor</h4>` + fields(rows);
  if (t.sparsity) {
    out += `<h4 class="insp"><span class="dnum">D3</span>sparsity unit</h4>` + fields([
      ['axis', t.sparsity.axis],
      ['activated per element', t.sparsity.activated_per_element != null
        ? fmtInt(t.sparsity.activated_per_element) : '—'],
      ['units', t.sparsity.units != null ? fmtInt(t.sparsity.units) : '—'],
      ['activated fraction', t.sparsity.activated_fraction != null
        ? t.sparsity.activated_fraction.toPrecision(3) : 'not resolvable'],
    ]);
  }
  return out;
}

function d4Body(st) {
  let out = `<h4 class="insp"><span class="dnum">D4</span>state identity</h4>` + fields([
    ['members', st.members.join(', ')],
    ['contract · state', `${st.contract} · ${st.state}`],
    ['law', st.law],
    ['access', st.access],
    ['sharing', st.sharing],
    ['stream', `${st.stream.kind} · ${st.stream.stream}`],
    ['indexed by source', String(!!st.indexed_by_source)],
    ['carried across fragments', String(!!st.carried_across_fragments)],
    ['instance key', (st.instance_key || []).join(', ')],
    ['operations', (st.operations || []).join(', ')],
  ].concat(st.span ? [['span', JSON.stringify(st.span)]] : [])
   .concat(st.stride ? [['stride', JSON.stringify(st.stride)]] : []));
  out += `<h4 class="insp"><span class="dnum">D4</span>payload<span class="n">${esc(fmtBytes(st.bytes_per_cached_position))} / pos</span></h4>`;
  out += fields(st.payload.map(c => [`${c.component} · ${c.role}`, `${fmtShape(c.shape)} ${c.dtype} · ${fmtBytes(c.bytes)}`]));
  if (st.bytes_bounded != null) out += fields([['bounded', fmtBytes(st.bytes_bounded)]]);
  out += `<div class="state-box"><b>visits</b> — written ${esc(st.visits.write)}, read ${esc(st.visits.read)}.</div>`;
  return out;
}

function cutBody(name) {
  const c = DERIVED.d2.cuts.find(x => x.cut === name);
  if (!c) return '';
  const payload = c.payload.map(v =>
    `<div class="drow"><div class="n">${esc(v.value)}</div>
      <div class="m"><span>bytes / element</span><b>${esc(fmtBytes(v.bytes_per_element))}</b></div>
      <div class="m"><span>count</span><b>${esc(fmtCounts(v.count))}</b></div></div>`).join('');
  const { block, why } = cutMarkable(name);
  const nodes = block ? [...block] : [];
  const loss = lossOf(nodes).length;
  const comms = {};
  for (const x of partitionsOf(nodes)) comms[x.communication] = (comms[x.communication] || 0) + 1;
  if (why) {
    return `<h4 class="insp"><span class="dnum">D2</span>legal cut</h4>` + fields([
        ['kind', c.kind],
        ['blocks', `${fmtInt(c.sizes[0])} | ${fmtInt(c.sizes[1])} nodes`],
        ['crossing values', String(c.payload.length)],
        ['bytes per element', fmtBytes(c.bytes_per_element)],
        ['bytes per invocation', Object.entries(c.bytes_per_invocation).map(([k, v]) => `${fmtBytes(v)} × ${k}`).join(', ')],
      ]) +
      `<h4 class="insp"><span class="dnum">D2</span>payload<span class="n">${c.payload.length}</span></h4>${payload}` +
      `<div class="state-box"><b>not marked on the diagram</b> — ${esc(why)}.</div>`;
  }
  return `<h4 class="insp"><span class="dnum">D2</span>legal cut</h4>` + fields([
      ['kind', c.kind === 'layer' ? 'layer — ancestor closure of a composition prefix'
                                  : 'family — ancestor closure of a family'],
      ['blocks', `${fmtInt(c.sizes[0])} | ${fmtInt(c.sizes[1])} nodes`],
      ['crossing values', String(c.payload.length)],
      ['bytes per element', fmtBytes(c.bytes_per_element)],
      ['bytes per invocation', Object.entries(c.bytes_per_invocation).map(([k, v]) => `${fmtBytes(v)} × ${k}`).join(', ')],
    ]) +
    `<h4 class="insp"><span class="dnum">D2</span>payload<span class="n">${c.payload.length}</span></h4>${payload}` +
    `<div class="state-box">Legal by construction — the closure is downward closed, so every crossing edge points out of it. Counts are never numbers (§10.3): a consumer multiplies by the counts it knows.</div>` +
    `<h4 class="insp"><span class="dnum">D6</span>the near block<span class="n">${fmtInt(nodes.length)} nodes</span></h4>` +
    fields([
      ['information loss', `${loss} flattened axes`],
    ].concat(Object.entries(comms).sort((a, b) => b[1] - a[1])
      .map(([k, v]) => [k, `${fmtInt(v)} node(s)`]))) +
    `<div class="state-box">Which of these cuts is a <b>good</b> one is not decided here: partitions are semantic, and the machine and the workload are inputs a consumer adds (§10.3).</div>`;
}

function fields(rows) {
  return rows.map(([k, v]) =>
    `<div class="field"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join('');
}

// ---------- inspector panel ----------
function occurrenceBody(o, families, matcher) {
  const args = Object.entries(o.arguments || {}).map(([k, v]) =>
    `<div class="field"><span class="k">${esc(k)}</span><span class="v">${esc(argValueStr(v))}</span></div>`).join('');
  const famHtml = (families || []).map(f => `<span class="fam-chip">${esc(f)}</span>`).join('');
  const states = findStateBindingsFor(matcher).map(([sid, b]) => stateBody(sid, b, false)).join('');
  return `
    <span class="contract-chip">${esc(o.contract.name)} \u00b7 ${esc(o.contract.version)}</span>
    <div>${famHtml}</div>
    <h4 class="insp">arguments</h4>
    ${args || '<div class="field"><span class="k">\u2014</span></div>'}
    ${states}`;
}

function compositionBody(name, c) {
  const idxNames = Object.keys(c.indices);
  const rangeRows = idxNames.map(ix => {
    const r = c.indices[ix];
    return `<div class="field"><span class="k">${esc(ix)}</span><span class="v">${esc(exprStr(r.start))} \u2026 ${esc(exprStr(r.stop))} step ${esc(exprStr(r.step))}</span></div>`;
  }).join('');
  const n = compInstanceCount(c);
  const famHtml = (c.families || []).map(f => `<span class="fam-chip">${esc(f)}</span>`).join('');
  const members = Object.entries(c.occurrences).map(([site, so]) => {
    const sel = escAttr(JSON.stringify({ kind: 'site', comp: name, site }));
    return `<div class="member-row" data-select='${sel}'><span class="n">${esc(site)}</span><span class="c">${esc(so.contract.name)}@${esc(so.contract.version)}</span></div>`;
  }).join('');
  return `
    <div>${famHtml}</div>
    <h4 class="insp">index</h4>
    ${rangeRows}
    <div class="field"><span class="k">instances</span><span class="v">${n != null ? n : 'not statically resolvable'}</span></div>
    <h4 class="insp">contains \u00b7 ${Object.keys(c.occurrences).length}</h4>
    <div class="members">${members}</div>`;
}

function stateBody(sid, b, bare) {
  const membersHtml = bare ? `<div class="field"><span class="k">members</span><span class="v">${esc(b.members.map(m => `${occSelLabel(m.occurrence)}.${m.state}`).join(', '))}</span></div>` : '';
  const indices = Object.keys((b.identity && b.identity.indices) || {});
  const key = [...indices, 'session', 'branch'].join(' \u00d7 ');
  const dtype = b.dtype ? (typeof b.dtype === 'string' ? b.dtype : '@' + b.dtype.quantity) : 'role default';
  return `${bare ? '' : `<h4 class="insp">state \u00b7 <code>${esc(sid)}</code></h4>`}
    ${membersHtml}
    <div class="state-box">
      <b>instance key</b> \u2014 ${esc(key)} (derived)<br>
      <b>members</b> \u2014 ${b.members.length}<br>
      <b>dtype</b> \u2014 ${esc(dtype)} \u00b7 carrying across fragments is derived (\u00a75.3)
    </div>`;
}

function bindingItemBody(which, id) {
  const b = RAW.bindings[which][id];
  if (which === 'values') {
    const fe = b.for_each ? Object.entries(b.for_each).map(([k, r]) => `${k}: ${exprStr(r.start)}\u2026${exprStr(r.stop)} step ${exprStr(r.step)}`).join('; ') : null;
    return `
      <div class="field"><span class="k">from</span><span class="v">${esc(occSelLabel(b.from.occurrence))}.${esc(b.from.port)}</span></div>
      <div class="field"><span class="k">to</span><span class="v">${esc(occSelLabel(b.to.occurrence))}.${esc(b.to.port)}</span></div>
      ${fe ? `<div class="field"><span class="k">for_each</span><span class="v">${esc(fe)}</span></div>` : ''}`;
  }
  if (which === 'parameters') {
    const members = b.members.map(m => `${occSelLabel(m.occurrence)}.${m.parameter}`).join(', ');
    return `
      <div class="field"><span class="k">tensor</span><span class="v">${esc(b.tensor.name)}</span></div>
      <div class="field"><span class="k">members</span><span class="v">${esc(members)}</span></div>
      ${b.members.length > 1 ? '<div class="field"><span class="k">shared</span><span class="v">yes \u2014 tied</span></div>' : ''}`;
  }
  if (which === 'states') return stateBody(id, b, true);
  return `
    <div class="field"><span class="k">identity</span><span class="v">${esc((b.identity || {}).digest || '\u2014')}</span></div>
    <div class="field"><span class="k">dtype</span><span class="v">${esc(b.dtype || '\u2014')}</span></div>`;
}

function interfaceBody(dir, spec) {
  if (dir === 'inputs') return `
    <div class="field"><span class="k">to</span><span class="v">${esc(spec.to.map(e => `${occSelLabel(e.occurrence)}.${e.port}`).join(', '))}</span></div>
    <div class="field"><span class="k">kind</span><span class="v">${esc(spec.kind)}${spec.stream ? ' \u00b7 joins ' + esc(spec.stream) : ''}${spec.fragmented ? ' \u00b7 fragmented' : ''}</span></div>`;
  return `
    <div class="field"><span class="k">from</span><span class="v">${esc(occSelLabel(spec.from.occurrence))}.${esc(spec.from.port)}</span></div>
    <div class="field"><span class="k">domain</span><span class="v">derived from the port (\u00a75.3)</span></div>
    <div class="field"><span class="k">generative</span><span class="v">${spec.generative}</span></div>`;
}

function qtyRow(name, q) {
  let val;
  if (q.source.kind === 'literal') val = JSON.stringify(q.source.value);
  else if (q.source.kind === 'external') val = `external "${q.source.name}"`;
  else val = `${exprStr(q.source.derivation.expression)} (${q.source.derivation.status})`;
  if (q.domain) val += ` \u2208 ${domainStr(q.domain)}`;
  return { regime: q.regime, kind: q.type.kind, val };
}

function selLabel(sel) {
  if (sel.kind === 'root') return sel.name;
  if (sel.kind === 'product') return sel.which.toUpperCase();
  if (sel.kind === 'd3' || sel.kind === 'd4') return sel.identity;
  if (sel.kind === 'd1node') return sel.id;
  if (sel.kind === 'cut') return sel.name;
  if (sel.kind === 'composition') return `${sel.name} (composition)`;
  if (sel.kind === 'site') return `${sel.comp}/${sel.site}`;
  if (sel.kind === 'bindingGroup') return sel.which;
  if (sel.kind === 'bindingItem') return sel.id;
  if (sel.kind === 'interface') return sel.name;
  if (sel.kind === 'quantities') return 'quantities';
  return '';
}

let currentSelection = null;

function renderInspector(sel) {
  const insp = document.getElementById('inspector');
  if (!sel) { insp.innerHTML = '<div class="insp-empty">Select a node to inspect it.</div>'; return; }
  let title, kindLabel, bodyHtml, rawFragment;

  if (sel.kind === 'root') {
    const o = RAW.occurrences[sel.name];
    title = sel.name; kindLabel = 'occurrence'; rawFragment = o;
    bodyHtml = occurrenceBody(o, o.families, matcherRoot(sel.name));
  } else if (sel.kind === 'composition') {
    const c = RAW.compositions[sel.name];
    title = sel.name; kindLabel = 'composition'; rawFragment = c;
    bodyHtml = compositionBody(sel.name, c);
  } else if (sel.kind === 'site') {
    const so = RAW.compositions[sel.comp].occurrences[sel.site];
    const compFam = RAW.compositions[sel.comp].families || [];
    title = `${sel.comp}/${sel.site}`; kindLabel = 'occurrence (composed)'; rawFragment = so;
    bodyHtml = occurrenceBody(so, [...new Set([...compFam, ...(so.families || [])])], matcherSite(sel.comp, sel.site));
  } else if (sel.kind === 'bindingGroup') {
    rawFragment = RAW.bindings[sel.which]; title = sel.which; kindLabel = 'bindings';
    const ids = Object.keys(rawFragment);
    bodyHtml = `<h4 class="insp">${esc(sel.which)} \u00b7 ${ids.length}</h4>` +
      (ids.map(id => `<div class="member-row" data-select='${escAttr(JSON.stringify({ kind: 'bindingItem', which: sel.which, id }))}'><span class="n">${esc(id)}</span></div>`).join('') || '<div class="field"><span class="k">\u2014</span></div>');
  } else if (sel.kind === 'bindingItem') {
    rawFragment = RAW.bindings[sel.which][sel.id]; title = sel.id; kindLabel = `binding \u00b7 ${sel.which}`;
    bodyHtml = bindingItemBody(sel.which, sel.id);
  } else if (sel.kind === 'interface') {
    rawFragment = RAW.interfaces[sel.dir][sel.name]; title = sel.name; kindLabel = `interface \u00b7 ${sel.dir}`;
    bodyHtml = interfaceBody(sel.dir, rawFragment);
  } else if (sel.kind === 'quantities') {
    rawFragment = RAW.quantities; title = 'quantities'; kindLabel = 'namespace';
    bodyHtml = Object.entries(RAW.quantities).map(([n, q]) => {
      const info = qtyRow(n, q);
      return `<div class="field"><span class="k">${esc(n)} <span style="opacity:.55">(${esc(info.regime)}, ${esc(info.kind)})</span></span><span class="v">${esc(info.val)}</span></div>`;
    }).join('');
  } else if (sel.kind === 'product') {
    rawFragment = HAS ? DERIVED[sel.which] : null;
    title = sel.which.toUpperCase();
    kindLabel = 'product \u00b7 derived document';
    bodyHtml = HAS ? `<div class="insp-none">Read whole in the products tab.</div>` : '';
  } else if (sel.kind === 'd3') {
    rawFragment = DERIVED.d3.tensors.find(t => t.identity === sel.identity);
    title = sel.identity; kindLabel = 'parameter tensor \u00b7 D3';
    bodyHtml = rawFragment ? d3Body(rawFragment) : '';
  } else if (sel.kind === 'd4') {
    rawFragment = DERIVED.d4.states.find(t => t.identity === sel.identity);
    title = sel.identity; kindLabel = 'state identity \u00b7 D4';
    bodyHtml = rawFragment ? d4Body(rawFragment) : '';
  } else if (sel.kind === 'd1node') {
    rawFragment = D1.nodes[sel.id];
    title = sel.id; kindLabel = 'occurrence \u00b7 D1';
    bodyHtml = rawFragment
      ? `<span class="contract-chip">${esc(rawFragment.contract.name)} \u00b7 ${esc(rawFragment.contract.version)}</span>` +
        `<div>${(rawFragment.families || []).map(f => `<span class="fam-chip">${esc(f)}</span>`).join('')}</div>` +
        `<h4 class="insp">arguments</h4>` +
        fields(Object.entries(rawFragment.arguments || {}).map(([k, v]) =>
          [k, typeof v === 'object' ? JSON.stringify(v) : String(v)]))
      : '';
  } else if (sel.kind === 'cut') {
    rawFragment = DERIVED.d2.cuts.find(c => c.cut === sel.name);
    title = sel.name; kindLabel = 'legal cut \u00b7 D2 \u00b7 D6';
    bodyHtml = cutBody(sel.name);
  }

  // The declaration, then what it expands to. The second reading is appended
  // to the first for anything the model itself names.
  if (['root', 'composition', 'site'].includes(sel.kind)) bodyHtml += derivedSections(sel);

  const html = `
    <div class="insp-title"><h3>${esc(title)}</h3></div>
    <div class="insp-kind">${esc(kindLabel)}</div>
    ${bodyHtml}
    <h4 class="insp">raw json</h4>
    <div class="json-peek">${highlightJson(JSON.stringify(rawFragment, null, 2))}</div>`;
  insp.innerHTML = html;
  document.getElementById('inspector2').innerHTML = html;
}

// ---------- the products tab ----------
// A product read whole, which the diagram cannot answer: what is big, what is
// shared, what is bounded, what crosses a cut. Listings fold along the model's
// own index — [layer=0…31] is one row that opens into 32.
const PRODUCTS = [
  ['d1', 'expanded graph'], ['d2', 'values & cuts'], ['d3', 'parameter tensors'],
  ['d4', 'states'], ['d5', 'logical costs'], ['d6', 'cuts & partitions'],
];
let currentProduct = 'd3';
const openFolds = new Set();

function productCount(which) {
  if (!HAS) return '';
  if (which === 'd1') return Object.keys(D1.nodes).length;
  if (which === 'd2') return DERIVED.d2.values.length;
  if (which === 'd3') return DERIVED.d3.totals.tensors;
  if (which === 'd4') return DERIVED.d4.totals.identities;
  if (which === 'd5') return (DERIVED.d5.corrections || []).length;
  if (which === 'd6') return (DERIVED.d6.partitions || []).length;
  return '';
}

function buildProductRail() {
  document.getElementById('prodrail').innerHTML = '<div class="lbl">products</div>' +
    PRODUCTS.map(([which, label]) =>
      `<div class="r${which === currentProduct ? ' on' : ''}" data-product="${which}">` +
      `<span class="dnum">${which.toUpperCase()}</span>${esc(label)}` +
      `<span class="count">${esc(String(productCount(which)))}</span></div>`).join('');
}

function pHead(title, sub, facts) {
  return `<div class="prodhead"><h3>${esc(title)}</h3><span class="sub">${esc(sub)}</span>` +
    `<span class="facts">${esc(facts)}</span></div>`;
}

function pTotals(entries) {
  return `<div class="ptotals">` + entries.map(([label, value, cls]) =>
    `<div class="${cls || ''}"><span>${esc(label)}</span><b>${esc(String(value))}</b></div>`).join('') + '</div>';
}

function pTable(headers, rows) {
  const cols = headers.map(h => `<col${h.w ? ` style="width:${h.w}"` : ''}>`).join('');
  const th = headers.map(h => `<th${h.right ? ' style="text-align:right"' : ''}>${esc(h.label)}</th>`).join('');
  return `<table class="prod"><colgroup>${cols}</colgroup><tr>${th}</tr>${rows.join('')}</table>`;
}

// One listing, folded: a group row that opens into its members.
function foldedRows(items, identityOf, cells, selectOf) {
  const groups = foldGroups(items.map(identityOf));
  const byId = new Map(items.map(x => [identityOf(x), x]));
  const out = [];
  for (const g of groups) {
    const members = g.members.map(id => byId.get(id));
    if (members.length === 1) {
      const sel = selectOf(members[0]);
      out.push(`<tr class="pick"${sel ? ` data-select='${escAttr(JSON.stringify(sel))}'` : ''}>` +
        cells(members[0], g.label, 1) + '</tr>');
      continue;
    }
    const open = openFolds.has(g.key);
    out.push(`<tr class="fold" data-fold="${escAttr(g.key)}">` + cells(members[0], g.label, members.length, open) + '</tr>');
    if (open) for (const m of members) {
      const sel = selectOf(m);
      out.push(`<tr class="kid pick"${sel ? ` data-select='${escAttr(JSON.stringify(sel))}'` : ''}>` +
        cells(m, identityOf(m), 1) + '</tr>');
    }
  }
  return out;
}

function chev(label, count, open) {
  const text = `<span title="${escAttr(label)}">${esc(label)}</span>`;
  if (count <= 1) return text;
  return `<span class="chev">${open ? '▾' : '▸'}</span>${text}<span class="x">×${count}</span>`;
}

function productBody(which) {
  if (!HAS) {
    return `<div class="prodempty">No derived document for this model — ${esc(DERIVED_NOTE || 'not built')}.<br><br>` +
      `The products are built by <code>tensorspine --derive</code>; this page builds them into a temporary file before rendering, and renders without them when it cannot.</div>`;
  }
  const d2 = DERIVED.d2, d3 = DERIVED.d3, d4 = DERIVED.d4, d5 = DERIVED.d5, d6 = DERIVED.d6;

  if (which === 'd1') {
    const rows = foldedRows(Object.entries(D1.nodes).map(([id, n]) => ({ id, ...n })), x => x.id,
      (x, label, n, open) =>
        `<td class="id">${chev(label, n, open)}</td><td class="m">${esc(x.contract.name)} · ${esc(x.contract.version)}</td>` +
        `<td>${esc((x.families || []).join(', '))}</td>`,
      x => ({ kind: 'd1node', id: x.id }));
    return pHead('Expanded graph', 'one entry per emitted occurrence',
        `${Object.keys(D1.nodes).length} nodes · ${D1.edges.length} edges`) +
      pTotals([['nodes', Object.keys(D1.nodes).length], ['edges', D1.edges.length],
               ['inputs', Object.keys(D1.interfaces.inputs).length],
               ['outputs', Object.keys(D1.interfaces.outputs).length],
               ['template instances', (D1.instances || []).length, (D1.instances || []).length ? '' : 'off']]) +
      pTable([{ label: 'occurrence', w: '44%' }, { label: 'contract', w: '30%' },
              { label: 'families' }], rows);
  }

  if (which === 'd2') {
    const cuts = d2.cuts.map(c =>
      `<tr class="pick" data-select='${escAttr(JSON.stringify({ kind: 'cut', name: c.cut }))}'>` +
      `<td class="id">${esc(c.cut)}</td><td class="m">${esc(c.kind)}</td>` +
      `<td class="m">${fmtInt(c.sizes[0])} | ${fmtInt(c.sizes[1])}</td>` +
      `<td class="num">${esc(String(c.payload.length))}</td>` +
      `<td class="num">${esc(fmtBytes(c.bytes_per_element))}</td></tr>`);
    const streams = Object.entries(d2.streams).map(([name, st]) =>
      `<tr><td class="id">${esc(name)}</td><td class="m">${esc(st.kind)}</td><td class="m">${esc(fmtCounts(st.count))}</td></tr>`);
    return pHead('Values & cuts', 'the value inventory and the payload of every legal cut',
        `${d2.values.length} values · ${d2.cuts.length} cuts`) +
      pTotals([['streams', Object.keys(d2.streams).length], ['values', d2.values.length],
               ['legal cuts', d2.cuts.length],
               ['widest cut', fmtBytes(Math.max(...d2.cuts.map(c => c.bytes_per_element), 0)), 'q']]) +
      `<div class="foldnote" style="margin:16px 0 0">streams</div>` +
      pTable([{ label: 'stream', w: '34%' }, { label: 'kind', w: '22%' }, { label: 'count' }], streams) +
      `<div class="foldnote" style="margin:22px 0 0">legal cuts — pick one to mark it on the diagram</div>` +
      pTable([{ label: 'cut', w: '34%' }, { label: 'kind', w: '11%' }, { label: 'blocks', w: '17%' },
              { label: 'crossing', w: '14%', right: true },
              { label: 'payload / element', right: true }], cuts);
  }

  if (which === 'd3') {
    const rows = foldedRows(d3.tensors, t => t.identity,
      (t, label, n, open) => {
        const group = n > 1 ? d3.tensors.filter(x => foldOf(x.identity).key === foldOf(t.identity).key) : [t];
        return `<td class="id">${chev(label, n, open)}</td><td class="m">${esc(t.role)}</td>` +
          `<td class="m">${esc(t.dtype)}</td><td class="m">${esc(fmtShape(t.shape))}</td>` +
          `<td>${esc(t.sensitivity)}${t.tied ? ' · tied' : ''}</td>` +
          `<td class="num">${esc(fmtBytes(group.reduce((a, x) => a + x.bytes, 0)))}</td>`;
      },
      t => ({ kind: 'd3', identity: t.identity }));
    return pHead('Parameter tensors', 'one entry per identity instance, a tied tensor once',
        `${d3.totals.tensors} tensors · ${fmtBytes(d3.totals.bytes)}`) +
      pTotals([['tensors', d3.totals.tensors], ['elements', fmtInt(d3.totals.elements)],
               ['resident', fmtBytes(d3.totals.bytes), 'q'],
               ['tied', d3.totals.tied, d3.totals.tied ? '' : 'off']]) +
      pTable([{ label: 'identity', w: '32%' }, { label: 'role', w: '19%' }, { label: 'dtype', w: '8%' },
              { label: 'shape', w: '15%' }, { label: 'sensitivity', w: '14%' },
              { label: 'bytes', w: '12%', right: true }], rows) +
      `<div class="foldnote">Rows fold along the model's own index — <code>[layer=0…31]</code> is one row that opens into its instances.</div>`;
  }

  if (which === 'd4') {
    const rows = foldedRows(d4.states, st => st.identity,
      (st, label, n, open) => {
        const group = n > 1 ? d4.states.filter(x => foldOf(x.identity).key === foldOf(st.identity).key) : [st];
        return `<td class="id">${chev(label, n, open)}</td><td class="m">${esc(st.law)}</td>` +
          `<td class="m">${esc(st.access)}</td><td class="m">${esc(st.sharing)}</td>` +
          `<td class="m">${esc(st.stream.stream)}</td><td class="m">${esc((st.instance_key || []).join(', '))}</td>` +
          `<td class="num">${esc(fmtBytes(group.reduce((a, x) => a + (x.bytes_per_cached_position || 0), 0)))}</td>`;
      },
      st => ({ kind: 'd4', identity: st.identity }));
    const byLaw = d4.totals.by_law || {};
    return pHead('Complete state', 'one entry per state identity instance',
        `${d4.totals.identities} identities · ${fmtBytes(d4.totals.append_bytes_per_cached_position)} / cached position`) +
      pTotals([['append', byLaw.append || 0, byLaw.append ? '' : 'off'],
               ['window', byLaw.window || 0, byLaw.window ? '' : 'off'],
               ['fixed', byLaw.fixed || 0, byLaw.fixed ? '' : 'off'],
               ['per cached position', fmtBytes(d4.totals.append_bytes_per_cached_position), 'q'],
               ['bounded', fmtBytes(d4.totals.bounded_bytes), d4.totals.bounded_bytes ? '' : 'off'],
               ['carried', (d4.totals.carried || []).length, (d4.totals.carried || []).length ? '' : 'off']]) +
      pTable([{ label: 'identity', w: '23%' }, { label: 'law', w: '10%' }, { label: 'access', w: '17%' },
              { label: 'sharing', w: '15%' }, { label: 'stream', w: '9%' },
              { label: 'instance key', w: '15%' }, { label: 'B / pos', w: '11%', right: true }], rows);
  }

  if (which === 'd5') {
    const op = d5.operations;
    const opRows = ['element', 'cached_position', 'sequence', 'invocation'].filter(k => op[k]).map(k =>
      `<tr><td class="id">per ${esc(k.replace('_', ' '))}</td><td>${esc(op[k].status)}</td>` +
      `<td class="num">${esc(fmtOps(op[k].value))}</td></tr>`);
    const corr = foldedRows((d5.corrections || []).map((c, i) => ({ ...c, identity: `${c.node}#${c.entry}` })),
      c => c.node,
      (c, label, n, open) =>
        `<td class="id">${chev(label, n, open)}</td>` +
        `<td class="m">per ${esc(c.per.replace('_', ' '))}</td><td>${esc(c.status)}</td>` +
        `<td class="num">+${esc(fmtInt(c.value))}</td>`,
      c => ({ kind: 'd1node', id: c.node }));
    const sparsity = (d5.sparsity || []).map(x =>
      `<tr><td class="id">${esc(x.node)}</td><td class="m">${esc(x.contract)}</td>` +
      `<td class="num">${esc(x.units != null ? fmtInt(x.units) : '—')}</td>` +
      `<td class="num">${esc(x.activated_fraction != null ? x.activated_fraction.toPrecision(3) : '—')}</td>` +
      `<td class="num">${esc(fmtInt(x.union_per_invocation.value))} <span style="color:var(--faint)">${esc(x.union_per_invocation.status)}</span></td></tr>`);
    return pHead('Logical costs', 'the inventory rule of §4.1 and every declared correction — never operations executed',
        `${fmtOps(op.element.value)} / element`) +
      pTotals([['parameters', fmtBytes(d5.parameters.bytes), 'q'],
               ['per element', fmtOps(op.element.value), 'q'],
               ['per cached position', fmtOps(op.cached_position.value), op.cached_position.value ? 'q' : 'off'],
               ['corrections', (d5.corrections || []).length],
               ['sparsity units', (d5.sparsity || []).length, (d5.sparsity || []).length ? '' : 'off']]) +
      `<div class="foldnote" style="margin:16px 0 0">operations</div>` +
      pTable([{ label: 'basis', w: '46%' }, { label: 'status', w: '26%' },
              { label: 'value', right: true }], opRows) +
      (corr.length ? `<div class="foldnote" style="margin:22px 0 0">corrections</div>` +
        pTable([{ label: 'node', w: '44%' }, { label: 'per', w: '22%' }, { label: 'status', w: '14%' },
                { label: 'value', right: true }], corr) : '') +
      (sparsity.length ? `<div class="foldnote" style="margin:22px 0 0">sparsity units</div>` +
        pTable([{ label: 'node', w: '30%' }, { label: 'contract', w: '20%' },
                { label: 'units', w: '13%', right: true },
                { label: 'activated fraction', w: '17%', right: true },
                { label: 'union / invocation', right: true }], sparsity) : '');
  }

  // d6
  const cuts = d6.cuts.map(c =>
    `<tr class="pick" data-select='${escAttr(JSON.stringify({ kind: 'cut', name: c.cut }))}'>` +
    `<td class="id">${esc(c.cut)}</td><td class="m">${esc(c.kind)}</td>` +
    `<td class="m">${fmtInt(c.sizes[0])} | ${fmtInt(c.sizes[1])}</td>` +
    `<td class="num">${esc(String(c.crossing_values))}</td></tr>`);
  const parts = foldedRows((d6.partitions || []).map(x => ({ ...x, identity: x.node })), x => x.node,
    (x, label, n, open) =>
      `<td class="id">${chev(label, n, open)}</td><td class="m">${esc(x.contract)}</td>` +
      `<td class="m">${esc(partitionTarget(x.target))}</td><td class="m">${esc(x.communication)}</td>`,
    x => ({ kind: 'd1node', id: x.node }));
  const loss = foldedRows((d6.information_loss || []).map(x => ({ ...x, identity: x.node })), x => x.node,
    (x, label, n, open) =>
      `<td class="id">${chev(label, n, open)}</td><td class="m">${esc(x.slot)}</td><td class="m">${esc(x.axis)}</td>`,
    x => ({ kind: 'd1node', id: x.node }));
  return pHead('Legal cuts & partitions', 'semantic, not a plan — the machine and the workload are a consumer’s inputs (§10.3)',
      `${d6.cuts.length} cuts · ${(d6.partitions || []).length} partitions`) +
    pTotals([['legal cuts', d6.cuts.length], ['partitions', (d6.partitions || []).length],
             ['information loss', (d6.information_loss || []).length,
              (d6.information_loss || []).length ? 'q' : 'off']]) +
    `<div class="foldnote" style="margin:16px 0 0">legal cuts — pick one to mark it on the diagram</div>` +
    pTable([{ label: 'cut', w: '42%' }, { label: 'kind', w: '14%' }, { label: 'blocks', w: '22%' },
            { label: 'crossing values', right: true }], cuts) +
    `<div class="foldnote" style="margin:22px 0 0">partitions</div>` +
    pTable([{ label: 'node', w: '34%' }, { label: 'contract', w: '22%' },
            { label: 'target', w: '26%' }, { label: 'communication' }], parts) +
    (loss.length ? `<div class="foldnote" style="margin:22px 0 0">information loss — a flattened axis with no declared factors (O5.10)</div>` +
      pTable([{ label: 'node', w: '42%' }, { label: 'slot', w: '22%' }, { label: 'axis' }], loss) : '');
}

function renderProducts() {
  buildProductRail();
  document.getElementById('prodbody').innerHTML = productBody(currentProduct);
}

function openComposition(name) {
  const key = `canvas-${name}`;
  const kids = document.querySelectorAll(`[data-group="${CSS.escape(key)}"]`);
  if (kids.length && kids[0].style.display === 'none') {
    kids.forEach(k => { k.style.display = ''; });
    const chev = document.querySelector(`[data-tree-toggle="${CSS.escape(key)}"] .chev`);
    if (chev) chev.textContent = '\u25be';
  }
  const section = document.querySelector(`.comp-section[data-comp="${CSS.escape(name)}"]`);
  if (section) section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function select(obj) {
  currentSelection = obj;
  const target = JSON.stringify(obj);
  document.querySelectorAll('[data-select]').forEach(n => n.classList.toggle('selected', n.dataset.select === target));
  document.querySelectorAll('#canvas g.node.selected').forEach(n => n.classList.remove('selected'));
  const nid = nodeIdForSelection(obj);
  if (nid != null) {
    const g = document.querySelector(`#canvas g.node[id="${CSS.escape(nid)}"]`);
    if (g) g.classList.add('selected');
  }
  document.querySelectorAll('#prodbody tr').forEach(r => r.classList.toggle('on', r.dataset.select === target));
  renderInspector(obj);
  document.getElementById('statusbar').innerHTML = buildStatus();
  if (obj.kind === 'product') { currentProduct = obj.which; renderProducts(); switchTab('products'); }
  else if (['d3', 'd4', 'd1node', 'cut'].includes(obj.kind)) { if (currentTab === 'raw') switchTab('products'); }
  else switchTab('graph');
  if (obj.kind === 'cut') markCut(obj.name);
  if (obj.kind === 'composition') openComposition(obj.name);
}

// ---------- status bar ----------
function buildStatus() {
  const dc = developedCount();
  const parts = [];
  if (HAS) {
    // The expansion is a fact now, not an estimate: D1 counted the nodes.
    const t3 = DERIVED.d3.totals, t4 = DERIVED.d4.totals, d5 = DERIVED.d5;
    parts.push(
      `<span><b>${Object.keys(D1.nodes).length}</b> occurrences developed</span>`,
      `<span class="derived"><b>${t3.tensors}</b> tensors</span>`,
      `<span class="derived"><b>${fmtBytes(t3.bytes)}</b> parameters</span>`,
      `<span class="derived"><b>${t4.identities}</b> states</span>`,
      `<span class="derived"><b>${fmtBytes(t4.append_bytes_per_cached_position)}</b> / position</span>`,
      `<span class="derived"><b>${fmtOps(d5.operations.element.value)}</b> / element</span>`,
      `<span class="derived"><b>${DERIVED.d2.cuts.length}</b> legal cuts</span>`);
  } else {
    parts.push(
      `<span><b>${dc.total}</b>${dc.exact ? '' : '+'} occurrences developed</span>`,
      `<span><b>${Object.keys(RAW.compositions).length}</b> composition(s)</span>`,
      `<span><b>${Object.keys(RAW.bindings.states).length}</b> state descriptor(s)</span>`,
      `<span><b>${Object.keys(RAW.interfaces.inputs).length + Object.keys(RAW.interfaces.outputs).length}</b> public port(s)</span>`);
  }
  if (currentSelection) parts.push(`<span class="selection">selection <code>${esc(selLabel(currentSelection))}</code></span>`);
  return parts.join('');
}

// ---------- events ----------
// A "data-tree-toggle" / "data-group" pair is a plain disclosure widget: it
// is used both by the tree (occurrences / compositions / bindings groups)
// and, in the canvas, by each composition's collapsible internal diagram —
// there is no layout to redo on toggle any more, Graphviz already laid out
// both diagrams once, at generation time.
function onToggle(ev) {
  const t = ev.target.closest('[data-tree-toggle]');
  if (!t) return false;
  const key = t.getAttribute('data-tree-toggle');
  const kids = document.querySelectorAll(`[data-group="${CSS.escape(key)}"]`);
  if (!kids.length) return true;
  const willShow = kids[0].style.display === 'none';
  kids.forEach(k => { k.style.display = willShow ? '' : 'none'; });
  const chev = t.classList.contains('chev') ? t : t.querySelector('.chev');
  if (chev) chev.textContent = willShow ? '\u25be' : '\u25b8';
  return true;
}

function onTreeClick(ev) {
  if (onToggle(ev)) return;
  const s = ev.target.closest('[data-select]');
  if (s) select(JSON.parse(s.dataset.select));
}

function onCanvasClick(ev) {
  if (onToggle(ev)) return;
  const g = ev.target.closest('g.node');
  if (g) {
    const sel = selectionForNodeId(g.getAttribute('id'));
    if (sel) select(sel);
    return;
  }
  const s = ev.target.closest('[data-select]');
  if (s) select(JSON.parse(s.dataset.select));
}

function onSearch(ev) {
  const q = ev.target.value.trim().toLowerCase();
  document.querySelectorAll('#tree .row').forEach(r => r.classList.toggle('search-hide', !!q && !r.textContent.toLowerCase().includes(q)));
}

let currentTab = 'graph';

function switchTab(name) {
  currentTab = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t.dataset.tab === name));
  document.querySelector('.body-row').style.display = name === 'graph' ? 'flex' : 'none';
  document.getElementById('measurebar').style.display = name === 'graph' ? 'flex' : 'none';
  document.getElementById('prodpane').style.display = name === 'products' ? 'flex' : 'none';
  document.getElementById('rawpane').style.display = name === 'raw' ? 'block' : 'none';
  if (name === 'products') renderProducts();
}

// ---------- the second reading: mode, measure, cut, products, instance ----
let treeMode = 'declared';

function setTreeMode(mode) {
  if (mode === 'derived' && !HAS) return;
  treeMode = mode;
  document.querySelectorAll('#modeseg div').forEach(d => d.classList.toggle('on', d.dataset.mode === mode));
  document.getElementById('tree').innerHTML = mode === 'derived' ? derivedTree() : buildTree();
  const q = document.getElementById('search');
  if (q && q.value) onSearch({ target: q });
}

function showRaw(which) {
  document.querySelectorAll('#rawseg div').forEach(d => d.classList.toggle('on', d.dataset.raw === which));
  const doc = which === 'derived' ? DERIVED : RAW;
  document.getElementById('rawcode').innerHTML = highlightJson(JSON.stringify(doc, null, 2));
}

function onModeClick(ev) {
  const d = ev.target.closest('[data-mode]');
  if (d && !d.classList.contains('disabled')) setTreeMode(d.dataset.mode);
}

function onMeasureClick(ev) {
  const chip = ev.target.closest('.mchip');
  if (chip && !chip.classList.contains('disabled')) { paintMeasure(chip.dataset.measure); return; }
  const sel = ev.target.closest('#cutsel');
  const menu = document.getElementById('cutmenu');
  if (sel) { menu.style.display = menu.style.display === 'none' ? 'block' : 'none'; return; }
  const row = ev.target.closest('#cutmenu .r');
  if (row) {
    menu.style.display = 'none';
    document.querySelectorAll('#cutmenu .r').forEach(r => r.classList.toggle('on', r === row));
    const name = row.dataset.cut || null;
    markCut(name);
    if (name) select({ kind: 'cut', name });
  }
}

// Clicks inside either inspector: a member row, a derived row, or the
// instance control that steps between a site's instances.
function onInspectorClick(ev) {
  const step = ev.target.closest('[data-inst]');
  if (step) {
    const all = currentSelection ? nodesForSelection(currentSelection) : [];
    if (step.dataset.inst === 'agg') instState.aggregate = !instState.aggregate;
    else if (all.length) {
      instState.aggregate = false;
      instState.at = (instState.at + Number(step.dataset.inst) + all.length) % all.length;
    }
    renderInspector(currentSelection);
    return;
  }
  const s = ev.target.closest('[data-select]');
  if (s) select(JSON.parse(s.dataset.select));
}

function onProductsClick(ev) {
  const rail = ev.target.closest('[data-product]');
  if (rail) { currentProduct = rail.dataset.product; renderProducts(); return; }
  const fold = ev.target.closest('[data-fold]');
  if (fold) {
    const key = fold.dataset.fold;
    if (openFolds.has(key)) openFolds.delete(key); else openFolds.add(key);
    renderProducts();
    return;
  }
  const s = ev.target.closest('[data-select]');
  if (s) select(JSON.parse(s.dataset.select));
}

function init() {
  document.getElementById('model-name').textContent = RAW.model;
  document.getElementById('model-schema').textContent = RAW.schema;
  document.getElementById('catalog-tag').textContent = 'catalog: ' + RAW.catalog.map(b => b.base).join(' \u00b7 ');

  // What the page found beside the model, said plainly.
  const tag = document.getElementById('derived-tag');
  tag.classList.toggle('absent', !HAS);
  document.getElementById('derived-tag-text').textContent = HAS ? 'derived \u00b7 D1\u2013D6' : 'no derived document';
  if (!HAS) tag.title = DERIVED_NOTE || '';

  measureValues = buildMeasures();
  if (!HAS) {
    document.getElementById('measurebar').style.display = 'none';
    document.querySelector('#modeseg [data-mode="derived"]').classList.add('disabled');
  } else {
    document.querySelectorAll('.mchip').forEach(c => {
      const m = c.dataset.measure;
      if (m !== 'none' && !measureValues[m].ok) {
        c.classList.add('disabled');
        c.title = 'the per-node shares do not add up to what D5 reports';
      }
    });
    buildCutMenu();
  }

  document.getElementById('tree').innerHTML = buildTree();
  document.querySelector('#rawseg [data-raw="model"]').textContent = RAW.model + '.json';
  const rawDerived = document.querySelector('#rawseg [data-raw="derived"]');
  rawDerived.textContent = RAW.model + '.derived.json';
  if (!HAS) rawDerived.classList.add('disabled');
  showRaw('model');
  document.getElementById('statusbar').innerHTML = buildStatus();
  renderProducts();
  paintMeasure('none');       // empties the label row every node reserved

  document.getElementById('tree').addEventListener('click', onTreeClick);
  document.getElementById('canvas').addEventListener('click', onCanvasClick);
  document.getElementById('modeseg').addEventListener('click', onModeClick);
  document.getElementById('measurebar').addEventListener('click', onMeasureClick);
  document.getElementById('prodpane').addEventListener('click', onProductsClick);
  document.getElementById('inspector').addEventListener('click', onInspectorClick);
  document.getElementById('inspector2').addEventListener('click', onInspectorClick);
  document.getElementById('rawseg').addEventListener('click', ev => {
    const d = ev.target.closest('[data-raw]');
    if (d && !d.classList.contains('disabled')) showRaw(d.dataset.raw);
  });
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => switchTab(t.dataset.tab)));
  document.getElementById('search').addEventListener('input', onSearch);
}
document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


# The bar of a page that stands on its own: the site's, without the links.
STANDALONE_NAV = ('<nav class="sitebar">\n'
                  '  <span class="wordmark">Tensorspine</span>\n'
                  '  <span class="sitekind">Model view</span>\n'
                  '</nav>')


def derive_note(log):
    """The emitter's own reason for there being no products, out of the line
    it printed. Its words, not this viewer's."""
    for line in log.splitlines():
        m = re.search(r'\s(?:failed|skipped|catalog refused): (.*)$', line.rstrip())
        if m:
            return m.group(1).strip()
    if 'off the derived schema' in log:
        return 'the emitted document is off the derived schema'
    return 'the products could not be derived'


def derive_to_tempfile(model_path, bases=None, assignment=None, schema_dir=None):
    """Build the model's derived document \u2014 D1 to D6, §7 \u2014 into a temporary
    file, exactly as `--derive` writes one, and read it back.

    Going through the file is the point: `derive.run` validates what it emits
    against the derived schema and refuses to write a document it cannot vouch
    for, so what the page embeds is a document that passed that gate, not
    something this viewer computed on the side. The file is temporary because
    the page is the artifact; `--derive` is how you keep one.

    Returns (document, note): the document and None, or None and the reason
    there is none.
    """
    import derive as derive_mod
    import d1 as d1_mod
    log = io.StringIO()
    with tempfile.TemporaryDirectory(prefix='tensorspine-view-') as tmp:
        try:
            with contextlib.redirect_stdout(log):
                failed, skipped = derive_mod.run([model_path], bases, tmp, assignment,
                                                 schema_dir=schema_dir)
        except Exception as e:                      # a refusal is not a crash of --view
            return None, f"{type(e).__name__}: {e}"
        written = pathlib.Path(tmp) / d1_mod.output_name(model_path, 'derived')
        if failed or skipped or not written.is_file():
            return None, derive_note(log.getvalue())
        return json.loads(written.read_text(encoding='utf-8')), None


def js_payload(obj):
    """A JSON literal that cannot end the <script> element that carries it.
    Written compactly: the page holds two whole documents, and the raw tab
    pretty-prints them again for reading."""
    return (json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
            .replace('</', '<\\/').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))


def build_html(data, derived, note, title, nav_html=None):
    canvas = canvas_html(data)
    title_html = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return (TEMPLATE
            .replace('__TITLE__', title_html)
            .replace('__NAVBAR__', nav_html or STANDALONE_NAV)
            .replace('__CANVAS_HTML__', canvas)
            .replace('__MODEL_JSON__', js_payload(data))
            .replace('__DERIVED_JSON__', js_payload(derived))
            .replace('__DERIVED_NOTE__', js_payload(note)))


def run(model_paths, output=None, site_nav=None, bases=None, assignment=None,
        schema_dir=None):
    """Render each model to a self-contained page. Returns the number that
    failed. With no output, the page lands beside its model. `site_nav` is a
    file holding the navigation of the documentation site, with its links
    already resolved for the directory the pages are written to; without it the
    pages carry the wordmark alone.

    Every page is rendered from two documents: the model, and the derived
    document built for it into a temporary file first (`derive_to_tempfile`).
    A model whose products cannot be derived still gets a page \u2014 the page says
    why, and offers its first reading only."""
    failed = 0
    nav_html = pathlib.Path(site_nav).read_text(encoding='utf-8').strip() if site_nav else None
    for path in model_paths:
        src = pathlib.Path(path)
        data = model_mod.normalise(json.loads(src.read_text(encoding='utf-8')))
        if data.get('schema') != 'tensorspine/2.0':
            print(f"  {src.name}: warning: schema '{data.get('schema')}' "
                  f"!= 'tensorspine/2.0' \u2014 this viewer assumes that format", file=sys.stderr)

        derived, note = derive_to_tempfile(path, bases, assignment, schema_dir)

        model_name = data.get('model', src.stem)
        if output:
            out = pathlib.Path(output)
            if out.is_dir():
                out = out / (src.stem + '.html')
        else:
            out = src.with_suffix('.html')
        try:
            page = build_html(data, derived, note, f"{model_name} \u2013 Tensorspine", nav_html)
        except RuntimeError as e:
            print(f"  {src.name:34s} failed: {e}", file=sys.stderr)
            failed += 1
            continue
        out.write_text(page, encoding='utf-8')
        if derived:
            t3, t4 = derived['d3']['totals'], derived['d4']['totals']
            facts = (f" \u00b7 {t3['tensors']} tensors, {t4['identities']} states, "
                     f"{len(derived['d2']['cuts'])} cuts")
        else:
            facts = f" \u00b7 no products: {note}"
        print(f"  {src.name:34s} -> {out} ({len(page)} bytes){facts}")
    return failed
