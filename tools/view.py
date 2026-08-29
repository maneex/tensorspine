"""`--view`: a self-contained HTML visualization of an armature/2.0 model.

Requires Graphviz's `dot` on PATH.

Division of labor: Python reads the model document, builds the top-level
graph (root occurrences <-> compositions, plus each composition's own local
graph) and hands it to Graphviz, which does the actual layout and produces
SVG — laying out a DAG correctly (independent branches side by side, real
fan-out/fan-in, no crossing minimization guesswork) is exactly graphviz's
job, not this script's. The resulting SVGs are embedded directly in the page.
Everything else — the side tree, the click-to-inspect detail panel, the raw
JSON view — stays client-side in JavaScript, reading the embedded model JSON
directly; none of that involves layout, so there is nothing graphviz buys it.

Assumptions made (v1 of this tool):
  - inside a composition, only edges linking the SAME index combination are
    drawn; an edge linking two different index values (e.g. a carry between
    blocks) is surfaced as a badge on the receiving node instead of a drawn
    edge — drawing it spatially in a single representative block would look
    exactly like a cycle even though it isn't one;
  - a composition's instance count, and any range label, is only shown when
    its bounds resolve to literals or literal "model_constant" quantities.
"""
import html
import json
import math
import pathlib
import shutil
import subprocess
import sys

DOT_MONO = "Courier New"
DOT_SANS = "Helvetica"


# ---------- minimal scalar-expression evaluator (armature/2.0 §7.8) ----------
# Just enough to print index ranges / instance counts on the diagram labels.
# The full inspector-side evaluator (with conditionals, calls, domains, ...)
# stays in the page's JavaScript, where it belongs — this one only needs to
# handle what shows up in `compositions.*.indices`.

def _lit_value(quantities, name):
    q = quantities.get(name)
    if q and q.get('source', {}).get('kind') == 'literal':
        return q['source']['value']
    return None


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
    if 'context' in e:
        return e['context']
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
        add_edge(k, top_key(spec['to']['occurrence']))
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
                f'color="#3a4152", fillcolor="#0d0f14", fontname="{DOT_MONO}", fontsize=11, '
                f'fontcolor="#8890a0", label={dot_qid(label)}];')
    if kind == 'occurrence':
        o = data['occurrences'][node['name']]
        rows = [(node['name'], 14, '#eef0f5', True, True),
                (f"{o['contract']['name']} \u00b7 {o['contract']['version']}", 10, '#6a7185', False, False)]
        return f'  {dot_qid(node_id)} [id={dot_qid(node_id)}, color="#262b36", fillcolor="#161a22", label={html_label(rows)}];'
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
    rows = [(node['name'], 14, '#d9c8fb', True, True),
            (badge, 9.5, '#b18cf0', False, False),
            (f"{len(sites)} occurrence(s) per instance", 8.5, '#6a7185', False, False)]
    return f'  {dot_qid(node_id)} [id={dot_qid(node_id)}, color="#3a2f57", fillcolor="#14121c", label={html_label(rows)}];'


def top_level_dot(data, nodes, edges, quantities):
    lines = [
        'digraph G {',
        '  bgcolor="transparent";',
        '  rankdir="TB";',
        '  nodesep=0.55; ranksep=0.6;',
        f'  node [shape=box, style="rounded,filled", fontname="{DOT_SANS}", penwidth=1];',
        '  edge [color="#3a4152", penwidth=1.4, arrowsize=0.7, arrowhead=vee];',
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
        f'  node [shape=box, style="rounded,filled", fontname="{DOT_SANS}", penwidth=1, color="#24283a", fillcolor="#0f1119"];',
        '  edge [color="#3a4152", penwidth=1.3, arrowsize=0.6, arrowhead=vee];',
    ]
    for site in sites:
        node_id = f"{comp_name}::{site}"
        so = comp_def['occurrences'][site]
        rows = [(site, 12, '#eef0f5', True, True),
                (f"{so['contract']['name']} \u00b7 {so['contract']['version']}", 9, '#6a7185', False, False)]
        if site in carry_targets:
            rows.append(('\u21ba carry from previous instance', 8, '#f0b45f', False, False))
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
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body {
    background: #0d0f14; color: #d7dbe4;
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
  }
  .mono { font-family: ui-monospace, "Cascadia Mono", "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; }
  a { color: #6fd3e6; }

  .page { display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

  .tabbar { display: flex; align-items: center; gap: 14px; padding: 10px 18px; border-bottom: 1px solid #1f232c; background: #10131a; flex-shrink: 0; }
  .tab { font-size: 12.5px; padding: 6px 12px; border-radius: 5px; color: #8890a0; cursor: pointer; user-select: none; }
  .tab.on { background: #1c2028; color: #e6e9f0; }
  .breadcrumb { margin-left: 4px; font-size: 12.5px; color: #8890a0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .breadcrumb b { color: #e6e9f0; font-family: ui-monospace, monospace; }
  .tabbar-right { margin-left: auto; display: flex; align-items: center; gap: 12px; }
  .catalog-tag { font-size: 10.5px; color: #6a7185; white-space: nowrap; }
  .search { border: 1px solid #262b36; border-radius: 5px; padding: 6px 12px; font-size: 11.5px; color: #d7dbe4; width: 220px; background: #14171f; outline: none; }
  .search::placeholder { color: #565d70; }

  .body-row { display: flex; flex: 1; min-height: 0; }

  .tree { width: 270px; flex-shrink: 0; padding: 14px 8px; border-right: 1px solid #1c2028; overflow: auto; font-size: 12.5px; }
  .tree .grp { padding: 5px 10px; color: #565d70; text-transform: uppercase; font-size: 10px; letter-spacing: 0.08em; margin-top: 12px; }
  .tree .grp:first-child { margin-top: 0; }
  .tree .row { display: flex; align-items: center; gap: 7px; padding: 5px 10px; border-radius: 4px; color: #b7bccb; cursor: pointer; }
  .tree .row:hover { background: #14171f; }
  .tree .row.child { padding-left: 26px; }
  .tree .row.selected { background: #1c2733; color: #eafaff; }
  .tree .row.search-hide { display: none !important; }
  .tree .chev { color: #4a5063; font-size: 9px; width: 9px; flex-shrink: 0; cursor: pointer; }
  .swatch { width: 7px; height: 7px; border-radius: 2px; flex-shrink: 0; }
  .sw-occ { background: #7f93f0; }
  .sw-comp { background: #b18cf0; }
  .sw-state { background: #f0b45f; }
  .sw-io { background: #7fd99a; }
  .count { margin-left: auto; color: #4a5063; font-size: 10.5px; }

  /* The canvas holds diagrams RENDERED BY GRAPHVIZ (dot -Tsvg) at generation
     time: nothing here computes layout in the browser. A composition is one
     box in the top diagram; its own internal structure is a second,
     separately laid-out diagram in a collapsible section below. */
  .canvas { flex: 1; min-width: 0; padding: 26px 30px; overflow: auto; background-image: radial-gradient(#171b24 1px, transparent 1px); background-size: 22px 22px; }
  .top-graph { display: flex; justify-content: center; }
  #canvas g.node { cursor: pointer; }
  #canvas g.node:hover path { stroke: #5a6478; }
  #canvas g.node.selected path { stroke: #6fd3e6; stroke-width: 2px; }

  .comp-sections { display: flex; flex-direction: column; gap: 14px; max-width: 960px; margin: 30px auto 0; }
  .comp-section { border: 1px solid #262b36; border-radius: 9px; background: #10131a; overflow: hidden; }
  .comp-section-head { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; padding: 10px 16px; cursor: pointer; }
  .comp-section-head:hover { background: #14171f; }
  .comp-section-head .chev { color: #6a7185; font-size: 11px; }
  .comp-section-head .name { font-family: ui-monospace, monospace; font-weight: 600; color: #d9c8fb; font-size: 13.5px; }
  .comp-section-head .comp-badge { font-size: 10px; background: #2a2140; color: #d9c8fb; padding: 2px 8px; border-radius: 999px; white-space: nowrap; }
  .comp-section-head .comp-sub { font-size: 11px; color: #6a7185; flex-basis: 100%; }
  .comp-section-body { padding: 16px; display: flex; justify-content: center; border-top: 1px solid #1c2028; }

  .inspector { width: 340px; flex-shrink: 0; border-left: 1px solid #1c2028; padding: 18px 18px; overflow: auto; }
  .insp-empty { color: #565d70; font-size: 12.5px; padding: 30px 4px; text-align: center; }
  .insp-title h3 { margin: 0; font-family: ui-monospace, monospace; font-size: 17px; color: #eef0f5; word-break: break-word; }
  .insp-kind { font-size: 10.5px; color: #b18cf0; text-transform: uppercase; letter-spacing: 0.06em; margin: 4px 0 12px; }
  .contract-chip { display: inline-block; font-size: 11px; padding: 3px 9px; background: #1c2028; color: #eef0f5; border-radius: 4px; font-family: ui-monospace, monospace; margin-bottom: 8px; }
  .fam-chip { font-size: 10.5px; border: 1px solid #33394a; border-radius: 999px; padding: 2px 9px; color: #a9aebc; margin: 0 6px 6px 0; display: inline-block; }
  h4.insp { font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em; color: #6a7185; margin: 16px 0 8px; }
  .field { display: flex; justify-content: space-between; gap: 12px; font-size: 12px; padding: 6px 0; border-bottom: 1px solid #1a1e27; }
  .field .k { color: #8890a0; }
  .field .v { font-family: ui-monospace, monospace; color: #d7dbe4; text-align: right; word-break: break-word; }
  .members { display: flex; flex-direction: column; gap: 6px; }
  .member-row { display: flex; justify-content: space-between; background: #14171f; border: 1px solid #1e222c; border-radius: 5px; padding: 7px 10px; font-size: 11.5px; cursor: pointer; }
  .member-row:hover { border-color: #3a4152; }
  .member-row .n { font-family: ui-monospace, monospace; color: #d9c8fb; }
  .member-row .c { color: #6a7185; }
  .state-box { border: 1px solid #4a3a20; background: #1c1710; border-radius: 6px; padding: 10px 12px; font-size: 11.5px; line-height: 1.6; color: #d9bf94; }
  .state-box b { color: #f0b45f; }
  .json-peek { margin-top: 4px; background: #0b0d12; border: 1px solid #1c2028; border-radius: 6px; padding: 10px 12px; font-family: ui-monospace, monospace; font-size: 10.5px; line-height: 1.6; color: #8890a0; white-space: pre-wrap; word-break: break-word; max-height: 320px; overflow: auto; }
  .json-peek .jkey { color: #7fd99a; }
  .json-peek .jstr { color: #f0b45f; }
  .json-peek .jnum { color: #6fd3e6; }
  .json-peek .jbool, .json-peek .jnull { color: #d9c8fb; }

  .statusbar { border-top: 1px solid #1c2028; padding: 8px 22px; display: flex; gap: 24px; font-size: 11px; color: #6a7185; background: #10131a; flex-shrink: 0; flex-wrap: wrap; }
  .statusbar b { color: #b7bccb; }

  .rawpane { display: none; flex: 1; min-height: 0; overflow: auto; background: #0b0d12; padding: 20px 26px; }
  .rawpane pre { margin: 0; font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.6; color: #8890a0; }
</style>
</head>
<body>
<div class="page">

  <div class="tabbar">
    <div class="tab on mono" data-tab="graph">Graph</div>
    <div class="tab mono" data-tab="raw">Raw JSON</div>
    <div class="breadcrumb mono"><b id="model-name"></b>&nbsp;<span id="model-schema"></span></div>
    <div class="tabbar-right">
      <span class="catalog-tag mono" id="catalog-tag"></span>
      <input class="search mono" id="search" placeholder="find\u2026">
    </div>
  </div>

  <div class="body-row">
    <div class="tree" id="tree"></div>
    <div class="canvas" id="canvas">__CANVAS_HTML__</div>
    <div class="inspector" id="inspector"><div class="insp-empty">Select a node to inspect it.</div></div>
  </div>

  <div class="statusbar mono" id="statusbar"></div>

  <div class="rawpane" id="rawpane"><pre id="rawcode"></pre></div>

</div>

<script>
const RAW = __MODEL_JSON__;

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

// ---------- scalar expression rendering (armature/2.0 §7.8) ----------
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
  if ('context' in e) return e.context;
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
    row(`<span class="swatch sw-io"></span>${esc(n)} \u2192 ${esc(occSelLabel(spec.to.occurrence))}`, { kind: 'interface', dir: 'inputs', name: n }));
  Object.entries(RAW.interfaces.outputs).forEach(([n, spec]) =>
    row(`<span class="swatch sw-io"></span>${esc(occSelLabel(spec.from.occurrence))} \u2192 ${esc(n)}`, { kind: 'interface', dir: 'outputs', name: n }));

  return L.join('');
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
  const sharing = b.keys.sharing.equal_on.join(', ');
  const liveness = `${exprStr(b.liveness.max_active_classes.expression)} (${b.liveness.max_active_classes.status})`;
  const visits = (b.visits || []).map(v => `${v.unit}/${v.phase} \u2264 ${exprStr(v.upper_bound.expression)}`).join('; ');
  return `${bare ? '' : `<h4 class="insp">state \u00b7 ${esc(sid)}</h4>`}
    ${membersHtml}
    <div class="state-box">
      <b>sharing</b> \u2014 equal on ${esc(sharing)}<br>
      <b>liveness</b> \u2264 ${esc(liveness)}<br>
      <b>visits</b> \u2014 ${esc(visits) || '\u2014'}
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
    <div class="field"><span class="k">to</span><span class="v">${esc(occSelLabel(spec.to.occurrence))}.${esc(spec.to.port)}</span></div>
    <div class="field"><span class="k">domain</span><span class="v">${esc(spec.domain.kind)} (${esc(spec.domain.source)})</span></div>`;
  return `
    <div class="field"><span class="k">from</span><span class="v">${esc(occSelLabel(spec.from.occurrence))}.${esc(spec.from.port)}</span></div>
    <div class="field"><span class="k">domain</span><span class="v">${esc(spec.domain.kind)} (${esc(spec.domain.source)})</span></div>
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
  }

  insp.innerHTML = `
    <div class="insp-title"><h3>${esc(title)}</h3></div>
    <div class="insp-kind">${esc(kindLabel)}</div>
    ${bodyHtml}
    <h4 class="insp">raw json</h4>
    <div class="json-peek">${highlightJson(JSON.stringify(rawFragment, null, 2))}</div>`;
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
  renderInspector(obj);
  document.getElementById('statusbar').innerHTML = buildStatus();
  if (obj.kind === 'composition') openComposition(obj.name);
}

// ---------- status bar ----------
function buildStatus() {
  const dc = developedCount();
  const parts = [
    `<span><b>${dc.total}</b>${dc.exact ? '' : '+'} occurrences developed</span>`,
    `<span><b>${Object.keys(RAW.compositions).length}</b> composition(s)</span>`,
    `<span><b>${Object.keys(RAW.bindings.states).length}</b> state descriptor(s)</span>`,
    `<span><b>${Object.keys(RAW.interfaces.inputs).length + Object.keys(RAW.interfaces.outputs).length}</b> public port(s)</span>`,
  ];
  if (currentSelection) parts.push(`<span>selection: ${esc(selLabel(currentSelection))}</span>`);
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

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t.dataset.tab === name));
  const graphOn = name === 'graph';
  document.querySelector('.body-row').style.display = graphOn ? 'flex' : 'none';
  document.getElementById('statusbar').style.display = graphOn ? 'flex' : 'none';
  document.getElementById('rawpane').style.display = graphOn ? 'none' : 'block';
}

function init() {
  document.getElementById('model-name').textContent = RAW.model;
  document.getElementById('model-schema').textContent = RAW.schema;
  document.getElementById('catalog-tag').textContent = 'catalog: ' + (Array.isArray(RAW.catalog)
    ? RAW.catalog.map(b => b.base).join(' \u00b7 ')
    : RAW.catalog.uri + (RAW.catalog.version ? ' \u00b7 ' + RAW.catalog.version : ''));
  document.getElementById('tree').innerHTML = buildTree();
  document.getElementById('rawcode').innerHTML = highlightJson(JSON.stringify(RAW, null, 2));
  document.getElementById('statusbar').innerHTML = buildStatus();

  document.getElementById('tree').addEventListener('click', onTreeClick);
  document.getElementById('canvas').addEventListener('click', onCanvasClick);
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => switchTab(t.dataset.tab)));
  document.getElementById('search').addEventListener('input', onSearch);
}
document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


def build_html(data, title):
    canvas = canvas_html(data)
    payload = json.dumps(data, ensure_ascii=False)
    payload = payload.replace('</', '<\\/').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    title_html = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return (TEMPLATE
            .replace('__TITLE__', title_html)
            .replace('__CANVAS_HTML__', canvas)
            .replace('__MODEL_JSON__', payload))


def run(model_paths, output=None):
    """Render each model to a self-contained page. Returns the number that
    failed. With no output, the page lands beside its model."""
    failed = 0
    for path in model_paths:
        src = pathlib.Path(path)
        data = json.loads(src.read_text(encoding='utf-8'))
        if data.get('schema') != 'armature/2.0':
            print(f"  {src.name}: warning: schema '{data.get('schema')}' "
                  f"!= 'armature/2.0' \u2014 this viewer assumes that format", file=sys.stderr)

        model_name = data.get('model', src.stem)
        if output:
            out = pathlib.Path(output)
            if out.is_dir():
                out = out / (src.stem + '.html')
        else:
            out = src.with_suffix('.html')
        try:
            page = build_html(data, f"{model_name} \u2014 armature inspector")
        except RuntimeError as e:
            print(f"  {src.name:34s} failed: {e}", file=sys.stderr)
            failed += 1
            continue
        out.write_text(page, encoding='utf-8')
        print(f"  {src.name:34s} -> {out} ({len(page)} bytes)")
    return failed
