"""`--document catalog`: the catalog, read back as one Markdown document.

A contract is written for a validator: tagged expressions, ordered rules,
conditions over arguments. This command writes it back for a person, and
adds nothing the catalog does not say. Two sources feed the page:

  * the DEFINITION of every unit — arguments, ports, slots, states, costs,
    partitions — rendered in full, with expressions printed in infix and
    conditions in words. This is the part that exists today and is never
    optional: a contract with no prose is still documented by its facts;
  * the DOCUMENTATION fields of the units — `summary`, `description`,
    `external_docs`, `tags`, `deprecated`, the `description` of every
    argument, port, slot, state, rule — whose shape is fixed by
    schemas/tensorspine-documentation.schema.json. They are inert everywhere
    else (§10.2): --validate and --d1 never read them.

A unit's page says what the unit declares, and nothing about who uses it:
there is no "cited by" index. The template of a template contract is read from
the models base the catalog was loaded with (catalog.template_path); the PATHs
of the command line are not read.

The generator never invents prose. A unit without a summary is rendered
without one and counted in the coverage appendix; it is not summarised from
its `note`, which is a maintainer's aside (the why), not a description of
the unit (the what). A malformed documentation field is a refusal with its
cause (I7) — the command exits 1 and writes nothing. Findings that are legal
but worth knowing — a grammar key this generator does not know, a condition
citing an undeclared argument, a tag no base declares — go to the findings
appendix and never block.

The output is deterministic: the same catalog gives the same bytes at the
same logical location. The site build writes it to temporary storage; the
generated Markdown is never tracked.
"""
import collections
import glob
import json
import os
import re
import sys

from jsonschema import Draft202012Validator

import catalog as catalog_mod
import schema as schema_mod
from expr import contract_value

DOC_SCHEMA_ROLE = 'documentation'
DOC_SCHEMA_ID = 'https://tensorspine.dev/schema/2.0/documentation.json'

# --- grammar: the keys each site may carry -----------------------------------
# The catalog schema is the authority; this table lets the generator notice a
# key it will not render (advisory), and separate documentation from meaning.

GRAMMAR = {
    'contract': {'version', 'arguments', 'ports', 'parameters', 'constants', 'state_ports',
                 'effects', 'logical_cost', 'sparsity', 'partitions', 'note',
                 'domain_transforms'},
    'template': {'version', 'template', 'note'},
    'argument': {'type', 'required', 'structural', 'default', 'present_when', 'note'},
    'port': {'shape', 'domain', 'multiplicity', 'present_when', 'note', 'role'},
    'parameter': {'role', 'shape', 'present_when', 'multiplicity', 'note', 'views', 'sharing'},
    'constant': {'shape', 'present_when', 'note', 'role'},
    'state': {'present_when', 'payload', 'rules', 'operations', 'key_axes', 'carried_across', 'note'},
    'carrying': {'when', 'note'},
    'component': {'shape', 'multiplicity', 'note', 'present_when', 'role'},
    'rule': {'when', 'law', 'access', 'sharing', 'indexed_by', 'span', 'stride', 'note'},
    'operation': {'effect', 'note'},
    'cost': {'when', 'expression', 'status', 'per', 'note'},
    'sparsity': {'unit', 'policy', 'activated_per_element', 'union_per_invocation', 'note'},
    'partition': {'target', 'communication', 'granularity', 'when', 'note'},
    'transform': {'from_port', 'to_port', 'relation', 'factor', 'note'},
    'axis': {'space', 'note'},
    'precision_role': {'admissible', 'default', 'sensitivity', 'note'},
    'base': {'catalog', 'templates', 'note'},
}

# Which fragment of the documentation schema governs the documentation of a
# site, and therefore which keys are documentation there.
FRAGMENT = {
    'contract': 'unit_documentation', 'template': 'unit_documentation',
    'axis': 'unit_documentation', 'precision_role': 'unit_documentation',
    'base': 'base_documentation', 'argument': 'argument_documentation',
}
for _site in ('port', 'parameter', 'constant', 'state', 'carrying', 'component', 'rule',
              'operation', 'cost', 'sparsity', 'partition', 'transform'):
    FRAGMENT[_site] = 'element_documentation'


# --- the report: refusals block, findings inform -------------------------------

class Report:
    def __init__(self):
        self.refusals = []
        self.findings = []
        self.coverage = collections.OrderedDict()   # site kind -> [documented, total]
        self.undocumented = collections.defaultdict(list)

    def refuse(self, where, what):
        self.refusals.append(f"{where}: {what}")

    def find(self, where, what):
        self.findings.append(f"{where}: {what}")

    def cover(self, site_kind, label, documented):
        c = self.coverage.setdefault(site_kind, [0, 0])
        c[1] += 1
        if documented:
            c[0] += 1
        else:
            self.undocumented[site_kind].append(label)


class Docs:
    """Extracts, checks and counts the documentation of one site."""

    def __init__(self, schema_dir, report):
        self.report = report
        self.registry = schema_mod.registry(schema_dir)
        if schema_mod.locate(schema_dir, DOC_SCHEMA_ROLE) is None:
            raise FileNotFoundError(
                f"no schema with $id ending in /{DOC_SCHEMA_ROLE}.json under {schema_dir}/ — "
                f"the documentation model is the authority on documentation fields, "
                f"nothing is rendered without it")
        self._validators = {}

    def _validator(self, fragment):
        if fragment not in self._validators:
            self._validators[fragment] = Draft202012Validator(
                {"$ref": f"{DOC_SCHEMA_ID}#/$defs/{fragment}"}, registry=self.registry)
        return self._validators[fragment]

    def keys(self, site_kind):
        """Documentation keys admissible at a site, from the schema itself."""
        fragment = FRAGMENT[site_kind]
        resource = self.registry.get_or_retrieve(DOC_SCHEMA_ID).value.contents
        return set(resource['$defs'][fragment]['properties'])

    def of(self, site_kind, node, where, label=None, primary='description'):
        """The documentation of a site: checked, counted, returned."""
        counted_as = 'contract' if site_kind == 'template' else site_kind
        doc_keys = self.keys(site_kind)
        docs = {k: node[k] for k in doc_keys if k in node}
        for error in sorted(self._validator(FRAGMENT[site_kind]).iter_errors(docs),
                            key=lambda e: list(e.absolute_path)):
            path = '/'.join(str(p) for p in error.absolute_path) or '<documentation>'
            self.report.refuse(where, f"documentation {path}: {error.message}")
        unknown = sorted(set(node) - GRAMMAR[site_kind] - doc_keys)
        for k in unknown:
            self.report.find(where, f"key '{k}' is neither grammar this generator knows "
                                    f"nor documentation; not rendered")
        self.report.cover(counted_as, label or where, primary in docs)
        return docs


# --- rendering of the closed languages -----------------------------------------

_PREC = {'add': 1, 'subtract': 1, 'multiply': 2, 'divide': 2, 'modulo': 2}
_CMP = {'equal': '=', 'not_equal': '!=', 'less': '<', 'less_or_equal': '<=',
        'greater': '>', 'greater_or_equal': '>='}


def _prec(e):
    if isinstance(e, dict) and 'op' in e:
        if e['op'] in _PREC:
            return _PREC[e['op']]
        if e['op'] == 'negate':
            return 3
        return 4
    if isinstance(e, dict) and 'if' in e:
        return 0
    return 4


def _paren(e, need):
    s = expr(e)
    return f"({s})" if _prec(e) < need else s


def expr(e):
    """A contract (or model) expression, in infix."""
    if not isinstance(e, dict):
        return json.dumps(e)
    if 'literal' in e:
        return json.dumps(e['literal'])
    if 'argument' in e:
        return e['argument']
    if 'quantity' in e:
        return e['quantity']
    if 'index' in e:
        return e['index']
    if 'axis_extent' in e:
        return f"extent({e['axis_extent']})"
    if 'if' in e:
        return f"if {cond(e['if'])} then {expr(e['then'])} else {expr(e['else'])}"
    if 'call' in e:
        c = e['call']
        inner = ', '.join(f"{k}={expr(v)}" for k, v in c.get('arguments', {}).items())
        return f"{c['contract']}({inner}).{c['result']}"
    if 'op' in e:
        op, args = e['op'], e['args']
        if op == 'negate':
            return f"-{_paren(args[0], 3)}"
        if op == 'absolute':
            return f"abs({expr(args[0])})"
        if op in ('min', 'max'):
            return f"{op}({', '.join(expr(a) for a in args)})"
        if op == 'ceil_divide':
            return f"ceil({expr(args[0])} / {expr(args[1])})"
        if op == 'floor_divide':
            return f"floor({expr(args[0])} / {expr(args[1])})"
        if op == 'modulo':
            return f"{_paren(args[0], 2)} mod {_paren(args[1], 3)}"
        sym = {'add': ' + ', 'multiply': '*', 'subtract': ' - ', 'divide': ' / '}[op]
        p = _PREC[op]
        right_need = p + 1 if op in ('subtract', 'divide') else p
        return sym.join([_paren(args[0], p)] + [_paren(a, right_need) for a in args[1:]])
    return json.dumps(e)


def cond(c):
    """A contract condition, in words."""
    if 'boolean' in c:
        return 'always' if c['boolean'] else 'never'
    if 'present' in c:
        return f"{c['present']} present"
    if 'not' in c:
        inner = c['not']
        if 'present' in inner:
            return f"{inner['present']} absent"
        return f"not ({cond(inner)})"
    if 'all' in c:
        return ' and '.join(_group(x) for x in c['all'])
    if 'any' in c:
        return ' or '.join(_group(x) for x in c['any'])
    cp = c['compare']
    return f"{expr(cp['left'])} {_CMP[cp['operator']]} {expr(cp['right'])}"


def _group(c):
    s = cond(c)
    return f"({s})" if ('all' in c or 'any' in c) else s


def arguments_cited(node, out=None):
    """Every argument an expression or condition tree refers to, by `argument`
    or by `present`."""
    out = out if out is not None else set()
    if isinstance(node, dict):
        for key in ('argument', 'present'):
            if key in node and isinstance(node[key], str) and len(node) == 1:
                out.add(node[key])
        for v in node.values():
            arguments_cited(v, out)
    elif isinstance(node, list):
        for v in node:
            arguments_cited(v, out)
    return out


def declared_argument_paths(arguments, prefix=''):
    """`window` and `window.span`: every name a condition may cite."""
    paths = set()
    for name, decl in arguments.items():
        paths.add(prefix + name)
        t = decl.get('type', {})
        if t.get('kind') == 'record':
            paths |= declared_argument_paths(t['fields'], prefix + name + '.')
    return paths


def domain_str(d):
    origin = d.get('from', {})
    if origin.get('self'):
        where = 'self'
    elif 'argument' in origin:
        where = f"argument `{origin['argument']}`"
    elif 'port' in origin:
        where = f"port `{origin['port']}`"
    else:
        where = json.dumps(origin)
    return f"{d['kind']} ({where})"


def origin_str(o):
    if o.get('self'):
        return 'self'
    if 'argument' in o:
        return f"argument {o['argument']}"
    if 'port' in o:
        return f"port {o['port']}"
    return json.dumps(o)


# --- Markdown helpers -----------------------------------------------------------

def cell(s):
    """One table cell: no pipes, no line breaks, no leading/trailing blanks."""
    if s is None:
        return ''
    s = str(s).strip().replace('|', '\\|')
    return s.replace('\r\n', '\n').replace('\n\n', '<br><br>').replace('\n', '<br>')


def code(s):
    return f"`{s}`" if s not in (None, '') else ''


def table(headers, rows):
    lines = ['| ' + ' | '.join(cell(h) for h in headers) + ' |',
             '|' + '|'.join('---' for _ in headers) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(cell(c) for c in row) + ' |')
    return lines + ['']


def anchor(kind, name, version=None):
    ident = f"{kind}-{name}" + (f"-{version}" if version else '')
    return ident.replace('/', '-')


def link(kind, name, version=None, text=None):
    label = text if text is not None else (f"{name}@{version}" if version else name)
    return f"[{label}](#{anchor(kind, name, version)})"


def heading(level, text, ident=None):
    lines = []
    if ident:
        # A blank line follows: pandoc's markdown does not let a heading
        # interrupt the paragraph an inline anchor starts.
        lines.append(f'<a id="{ident}" name="{ident}"></a>')
        lines.append('')
    lines.append(f"{'#' * level} {text}")
    lines.append('')
    return lines


def prose(text):
    return [text.strip(), '']


def note_block(note):
    return [f"> **Note (maintainers).** {note.strip()}", ''] if note else []


def describe(docs, node):
    """Description then note, for a table cell."""
    parts = []
    if 'description' in docs:
        parts.append(docs['description'])
    if node.get('note'):
        parts.append(f"*Note: {node['note']}*")
    return '<br>'.join(cell(p) for p in parts)


def is_absolute_url(url):
    return bool(re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', url)) or url.startswith('#')


def external_docs_lines(docs, rewrite=lambda url: url):
    lines = []
    for d in docs.get('external_docs', []):
        url = d['url'] if is_absolute_url(d['url']) else rewrite(d['url'])
        title = d.get('title') or d['url']
        kind = f" *({d['kind'].replace('_', ' ')})*" if d.get('kind') else ''
        desc = f" — {d['description']}" if d.get('description') else ''
        lines.append(f"- [{title}]({url}){kind}{desc}")
    return lines + [''] if lines else []


# --- the catalog, with envelopes -----------------------------------------------

def load_units(bases):
    """Every unit of the bases with its envelope and path, first base to answer
    wins; plus the base manifests, in order."""
    units = collections.OrderedDict()
    manifests = []
    for base in bases:
        if os.path.isdir(base):
            for path in sorted(glob.glob(os.path.join(base, '**', '*.json'), recursive=True)):
                with open(path, encoding='utf-8') as f:
                    unit = json.load(f)
                if unit.get('schema') != catalog_mod.UNIT_SCHEMA:
                    raise ValueError(f"{path}: schema {unit.get('schema')!r}, "
                                     f"expected {catalog_mod.UNIT_SCHEMA!r}")
                kind, name = unit.get('kind'), unit.get('name')
                definition = unit.get('definition') or {}
                record = {'kind': kind, 'name': name, 'definition': definition,
                          'path': path, 'base': base}
                if kind == 'base':
                    manifests.append(record)
                    continue
                version = definition.get('version') if kind == 'contract' else None
                units.setdefault((kind, name, version), record)
        else:
            with open(base, encoding='utf-8') as f:
                mono = json.load(f)
            manifests.append({'kind': 'base', 'name': mono.get('catalog', base),
                              'definition': {k: v for k, v in mono.items()
                                             if k not in ('schema', 'contracts', 'axes',
                                                          'precision')},
                              'path': base, 'base': base})
            for section, kind in catalog_mod.SECTIONS:
                for name, d in mono.get(section, {}).items():
                    version = d.get('version') if kind == 'contract' else None
                    units.setdefault((kind, name, version),
                                     {'kind': kind, 'name': name, 'definition': d,
                                      'path': base, 'base': base})
    return manifests, units


def find_template(cat, definition):
    """The template of a template contract, as the catalog pinned it (§4.6).
    Returns (path, model) or None."""
    path = catalog_mod.template_path(cat, definition)
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        return path, json.load(f)


# --- vocabulary in use -----------------------------------------------------------

class Vocabulary:
    def __init__(self):
        self.seen = collections.defaultdict(lambda: collections.defaultdict(set))

    def add(self, field, value, who):
        if value is None:
            return
        self.seen[field][json.dumps(value) if not isinstance(value, str) else value].add(who)

    def rows(self):
        for field in sorted(self.seen):
            for value in sorted(self.seen[field]):
                yield field, value, len(self.seen[field][value])


# --- the renderer -----------------------------------------------------------------

class Renderer:
    def __init__(self, manifests, units, cat, model_paths, docs, report, bases,
                 relative_to, output_dir=None):
        self.manifests = manifests
        self.units = units
        self.cat = cat
        self.model_paths = model_paths
        self.docs = docs
        self.report = report
        self.bases = bases
        self.relative_to = relative_to
        self.output_dir = output_dir
        self.vocab = Vocabulary()
        self.contracts = sorted((u for u in units.values() if u['kind'] == 'contract'),
                                key=lambda u: (u['name'], catalog_mod._semver(u['definition']['version'])))
        self.axes = sorted((u for u in units.values() if u['kind'] == 'axis'),
                           key=lambda u: u['name'])
        self.roles = sorted((u for u in units.values() if u['kind'] == 'precision_role'),
                            key=lambda u: u['name'])
        self.declared_tags = {}
        for m in manifests:
            for t in m['definition'].get('tags', []) or []:
                if isinstance(t, dict) and 'name' in t:
                    self.declared_tags.setdefault(t['name'], t)
        self.axis_names = {u['name'] for u in self.axes}
        self.role_names = {u['name'] for u in self.roles}

    # -- helpers -----------------------------------------------------------------------
    def rel(self, path):
        """A path shown relative to the repository when it is inside it, absolute
        otherwise — `../../../tmp/x` names nothing a reader can find."""
        if not self.relative_to:
            return path
        shown = os.path.relpath(path, self.relative_to)
        return os.path.abspath(path) if shown.startswith('..') else shown

    def rewrite_url(self, url):
        """A root-relative URL, made relative to where the page is written. On
        stdout the page has no location, so the URL stays root-relative."""
        if self.output_dir is None or self.relative_to is None:
            return url
        return os.path.relpath(os.path.join(self.relative_to, url), self.output_dir)

    def link_axis(self, name):
        return link('axis', name) if name in self.axis_names else f"`{name}`"

    def link_role(self, name):
        return link('role', name) if name in self.role_names else f"`{name}`"

    def link_contract(self, name, version=None):
        if version is None:
            d = self.cat['contracts'].get(name)
            version = d['version'] if d else None
        if ('contract', name, version) in self.units:
            return link('contract', name, version)
        return f"`{name}@{version}`" if version else f"`{name}`"

    def shape_symbolic(self, shape):
        axes = shape.get('axes', [])
        if not axes:
            return 'scalar'
        parts = []
        for a in axes:
            s = f"{a['name']}: {expr(a['extent'])}"
            if a.get('factors'):
                s += ' = ' + ' × '.join(f"{f['name']}: {expr(f['extent'])}" for f in a['factors'])
            parts.append(f"[{s}]")
        return ' × '.join(parts)

    def shape_axes(self, shape):
        axes = shape.get('axes', [])
        if not axes:
            return '—'
        parts = []
        for a in axes:
            s = f"{self.link_axis(a['axis'])} ({a['nature']})"
            if a.get('factors'):
                s += ' = ' + ' × '.join(self.link_axis(f['axis']) for f in a['factors'])
            if a.get('coordinate_domain'):
                s += f", coordinate in {domain_str(a['coordinate_domain'])}"
            parts.append(s)
        return ' × '.join(parts)

    def tags_line(self, docs, where):
        tags = docs.get('tags') or []
        for t in tags:
            if t not in self.declared_tags:
                self.report.find(where, f"tag '{t}' is declared by no base manifest")
        if not tags:
            return []
        return ['Tags: ' + ', '.join(link('tag', t, text=f"`{t}`") if t in self.declared_tags
                                    else f"`{t}`" for t in tags), '']

    def deprecated_lines(self, docs, where):
        dep = docs.get('deprecated')
        if not dep:
            return []
        text = f"> **Deprecated.** {dep['reason'].strip()}"
        sup = dep.get('superseded_by')
        if isinstance(sup, dict):
            key = ('contract', sup['name'], sup['version'])
            if key not in self.units:
                self.report.find(where, f"superseded_by names '{sup['name']}@{sup['version']}', "
                                        f"which the catalog does not carry")
            text += f" Superseded by {self.link_contract(sup['name'], sup['version'])}."
        elif isinstance(sup, str):
            text += f" Use `{sup}` instead."
        return [text, '']

    # -- document ---------------------------------------------------------------------
    def render(self):
        out = []
        out += self.head()
        out += self.overview()
        out += self.contracts_section()
        out += self.axes_section()
        out += self.roles_section()
        out += self.tags_section()
        out += self.appendix()
        return '\n'.join(out).rstrip('\n') + '\n'

    def head(self):
        out = []
        primary = self.manifests[0] if self.manifests else None
        if primary:
            d = primary['definition']
            where = self.rel(primary['path'])
            docs = self.docs.of('base', d, where, primary='summary')
            title = docs.get('title') or f"Catalog {d.get('catalog', primary['name'])}"
            out += heading(1, title)
            out.append(f"*Catalog `{d.get('catalog', primary['name'])}`, base "
                       f"`{self.rel(primary['base'])}` — generated by "
                       f"`tensorspine --document catalog`; edit the units, not this file.*")
            out.append('')
            if 'summary' in docs:
                out += [f"**{docs['summary']}**", '']
            if 'description' in docs:
                out += prose(docs['description'])
            out += note_block(d.get('note'))
            meta = []
            if 'contact' in docs:
                c = docs['contact']
                bits = [c.get('name'), f"<{c['email']}>" if c.get('email') else None,
                        c.get('url')]
                meta.append('Contact: ' + ' '.join(b for b in bits if b))
            if 'license' in docs:
                lic = docs['license']
                name = lic['name'] + (f" ({lic['identifier']})" if lic.get('identifier') else '')
                meta.append('License: ' + (f"[{name}]({lic['url']})" if lic.get('url') else name))
            for m in meta:
                out.append(f"- {m}")
            if meta:
                out.append('')
            ext = external_docs_lines(docs, self.rewrite_url)
            if ext:
                out += ['External documentation:', ''] + ext
            for extra in self.manifests[1:]:
                self.docs.of('base', extra['definition'], self.rel(extra['path']), primary='summary')
        else:
            out += heading(1, 'Catalog')
            self.report.find('catalog', 'no base manifest (`catalog.json`) in any base')
        bases = ', '.join(f"`{self.rel(b)}`" for b in self.bases)
        out.append(f"Bases consulted, in order: {bases}. "
                   f"{len(self.contracts)} contracts, {len(self.axes)} axes, "
                   f"{len(self.roles)} precision roles.")
        out.append('')
        out += heading(2, 'Contents')
        out += ['- [How to read this document](#how-to-read)',
                '- [Overview](#overview)',
                '- [Contracts](#contracts)']
        for ns, group in self.groups():
            out.append(f"  - [{ns}](#{anchor('namespace', ns.rstrip('.*'))}): " +
                       ', '.join(self.link_contract(u['name'], u['definition']['version'])
                                 for u in group))
        out += ['- [Axes](#axes)', '- [Precision roles](#precision-roles)']
        if self.declared_tags:
            out.append('- [Tags](#tags)')
        out += ['- [Appendix A — Closed vocabulary in use](#appendix-a)',
                '- [Appendix B — Documentation coverage](#appendix-b)',
                '- [Appendix C — Findings](#appendix-c)', '']
        out += heading(2, 'How to read this document', 'how-to-read')
        schema_link = self.rewrite_url('schemas/tensorspine-documentation.schema.json')
        out += [
            "Every unit is rendered from its definition first, then from its documentation. "
            "Facts — types, defaults, shapes, laws — come from the definition and cannot "
            "disagree with it; prose comes from `summary` and `description` fields "
            f"({schema_link}), and a maintainer's `note` is quoted as written. Nothing is "
            "inferred: a unit without a summary has none.",
            '',
            "- **Expressions** are contract arguments by name; `a.b` is a field of a record "
            "argument. Operators: `+ - * /`, `mod`, `ceil(a / b)`, `floor(a / b)`, `min`, "
            "`max`, `abs`. "
            "Strings are quoted.",
            "- **Conditions** read as prose: `x present` / `x absent` test an optional "
            "argument; comparisons use `= != < <= > >=`; `always` and `never` are the "
            "constant conditions.",
            "- **Shapes** list axes in declaration order as `[local name: extent]`; `= a × b` "
            "spells out the declared factors of a flattened axis (O5.10); `scalar` is rank 0. "
            "The *Axes* column gives each axis's catalog identity and nature; shapes unify by "
            "axis identity and extent (V4), never by position alone.",
            "- **Structural** marks an argument with `structural`: it decides which slots, "
            "ports or states exist or what shape they have. A non-structural argument changes "
            "only the computation.",
            "- **State rules** are ordered; the first rule whose condition holds decides the "
            "law, access geometry, sharing and indexing of the state (§4.3).",
            '',
        ]
        return out

    def groups(self):
        groups = collections.OrderedDict()
        for u in self.contracts:
            ns = u['name'].split('.')[0] + '.*' if '.' in u['name'] else 'unqualified'
            groups.setdefault(ns, []).append(u)
        return sorted(groups.items(), key=lambda kv: (kv[0] == 'unqualified', kv[0]))

    # -- overview -----------------------------------------------------------------------
    def overview(self):
        out = heading(2, 'Overview', 'overview')
        rows = []
        for u in self.contracts:
            d = u['definition']
            name, version = u['name'], d['version']
            if 'template' in d:
                shape = 'template'
            else:
                states = ', '.join(f"`{s}`" for s in d.get('state_ports', {}))
                shape = (f"{len(d['arguments'])} args · "
                         f"{len(d['ports']['inputs'])}→{len(d['ports']['outputs'])} ports · "
                         f"{len(d.get('parameters', {}))} params"
                         + (f" · state {states}" if states else ''))
            rows.append([self.link_contract(name, version),
                         ('*deprecated* ' if 'deprecated' in d else '') + d.get('summary', ''),
                         shape])
        out += ['### Contracts', '']
        out += table(['Contract', 'Summary', 'Shape'], rows)
        out += ['### Axes', '']
        out += table(['Axis', 'Space', 'Summary'],
                     [[link('axis', u['name']), u['definition']['space'],
                       u['definition'].get('summary', '')]
                      for u in self.axes])
        out += ['### Precision roles', '']
        out += table(['Role', 'Admissible', 'Default', 'Sensitivity', 'Summary'],
                     [[link('role', u['name']),
                       ', '.join(f"`{t}`" for t in u['definition']['admissible']),
                       code(u['definition']['default']),
                       u['definition'].get('sensitivity', '—'),
                       u['definition'].get('summary', '')]
                      for u in self.roles])
        return out

    # -- contracts -----------------------------------------------------------------------
    def contracts_section(self):
        out = heading(2, 'Contracts', 'contracts')
        out += ["Grouped by namespace, then by name and version. A contract is pinned by "
                "`{name, version}`; two versions of one name are two contracts (§8.2).", '']
        for ns, group in self.groups():
            out += heading(3, ns, anchor('namespace', ns.rstrip('.*')))
            for u in group:
                out += self.contract(u)
        return out

    def contract(self, u):
        d = u['definition']
        name, version = u['name'], d['version']
        where = self.rel(u['path'])
        label = f"{name}@{version}"
        is_template = 'template' in d
        site = 'template' if is_template else 'contract'
        docs = self.docs.of(site, d, where, label, primary='summary')
        out = heading(4, f"`{label}`" + (' — template' if is_template else ''),
                      anchor('contract', name, version))
        out.append(f"*{where}*")
        out.append('')
        out += self.deprecated_lines(docs, where)
        if 'summary' in docs:
            out += [f"**{docs['summary']}**", '']
        out += self.tags_line(docs, label)
        if 'description' in docs:
            out += prose(docs['description'])
        out += note_block(d.get('note'))
        ext = external_docs_lines(docs, self.rewrite_url)
        if ext:
            out += ['External documentation:', ''] + ext
        if is_template:
            out += self.template_section(u, where)
        else:
            out += self.at_a_glance(d)
            out += self.arguments(d, where, label)
            out += self.ports(d, where, label)
            out += self.parameters(d, where, label)
            out += self.constants(d, where, label)
            out += self.states(d, where, label)
            out += self.effects(d, where, label)
            out += self.cost(d, where, label)
            out += self.sparsity(d, where, label)
            out += self.partitions(d, where, label)
            out += self.transforms(d, where, label)
            self.check_conditions(d, where)
        return out

    def at_a_glance(self, d):
        declared = d['arguments']
        structural = sum(1 for a in declared.values() if a.get('structural'))
        required = sum(1 for a in declared.values() if a.get('required'))
        states = d.get('state_ports', {})
        laws = set()
        for s in states.values():
            for r in s['rules']:
                laws.add(r['law'])
        cost = f"{len(d.get('logical_cost', []))} correction(s)" if d.get('logical_cost') else '—'
        rows = [[f"{len(declared)} ({required} required, {structural} structural)",
                 str(len(d['ports']['inputs'])), str(len(d['ports']['outputs'])),
                 str(len(d.get('parameters', {}))), str(len(d.get('constants', {}))),
                 (', '.join(f"`{s}`" for s in states) + (f" ({', '.join(sorted(laws))})" if laws else '')) or 'none',
                 str(len(d.get('partitions', []))), cost]]
        return table(['Arguments', 'Inputs', 'Outputs', 'Parameters', 'Constants', 'State ports',
                      'Partitions', 'Logical cost'], rows)

    def argument_rows(self, arguments, where, label, prefix='', depth=0):
        rows, enums = [], []
        for aname, a in arguments.items():
            full = prefix + aname
            adocs = self.docs.of('argument', a, f"{where} argument {full}", f"{label}.{full}")
            t = a['type']
            kind = t['kind']
            if kind == 'enum':
                tdesc = 'enum: ' + ', '.join(f"`{json.dumps(v)}`" for v in t['values'])
            elif kind == 'physical':
                tdesc = f"physical ({t['unit']})"
            elif kind == 'record':
                tdesc = f"record of {len(t['fields'])} field(s)"
            else:
                tdesc = kind
            self.vocab.add('argument type', kind, label.split('@')[0])
            default = code(expr(a['default'])) if 'default' in a else ''
            desc = describe(adocs, a)
            if 'deprecated' in adocs:
                dep = adocs['deprecated']
                sup = f" Use `{dep['superseded_by']}` instead." if dep.get('superseded_by') else ''
                desc = f"**Deprecated.** {dep['reason']}{sup}" + (f"<br>{desc}" if desc else '')
            indent = '&nbsp;&nbsp;&nbsp;&nbsp;' * depth
            if 'present_when' in a:
                desc = (desc + '<br>' if desc else '') + f"*Applicable when {cond(a['present_when'])}.*"
            rows.append([f"{indent}`{full}`", tdesc, 'yes' if a.get('required') else 'no',
                         default, 'yes' if a.get('structural') else 'no', desc])
            if kind == 'enum' and 'value_descriptions' in adocs:
                for v in adocs['value_descriptions']:
                    if not any(str(x) == v for x in t['values']):
                        self.report.refuse(f"{where} argument {full}",
                                           f"value_descriptions names '{v}', not a declared value")
                enums.append((full, t['values'], adocs['value_descriptions']))
            elif 'value_descriptions' in adocs:
                self.report.refuse(f"{where} argument {full}",
                                   "value_descriptions on a non-enum argument")
            if kind == 'record':
                sub_rows, sub_enums = self.argument_rows(t['fields'], where, label,
                                                         full + '.', depth + 1)
                rows += sub_rows
                enums += sub_enums
        return rows, enums

    def arguments(self, d, where, label):
        out = ['##### Arguments', '']
        if not d['arguments']:
            return out + ['This contract takes no argument.', '']
        rows, enums = self.argument_rows(d['arguments'], where, label)
        out += table(['Argument', 'Type', 'Required', 'Default', 'Structural', 'Description'], rows)
        for full, values, descriptions in enums:
            out.append(f"Values of `{full}`:")
            out.append('')
            for v in values:
                text = descriptions.get(str(v), '')
                out.append(f"- `{json.dumps(v)}`" + (f" — {text}" if text else ''))
            out.append('')
        return out

    def ports(self, d, where, label):
        out = ['##### Ports', '']
        for side, title in (('inputs', 'Inputs'), ('outputs', 'Outputs')):
            ports = d['ports'][side]
            out += [f"{title}:", '']
            if not ports:
                out += ['none', '']
                continue
            rows = []
            for pname, p in ports.items():
                pdocs = self.docs.of('port', p, f"{where} port {pname}", f"{label}.{pname}")
                self.vocab.add('port domain', p['domain']['kind'], label.split('@')[0])
                for a in p['shape'].get('axes', []):
                    self.vocab.add('axis nature', a['nature'], label.split('@')[0])
                flags = []
                if 'present_when' in p:
                    flags.append(f"present when {cond(p['present_when'])}")
                if 'multiplicity' in p:
                    flags.append(f"multiplicity {expr(p['multiplicity'])}")
                rows.append([code(pname), code(self.shape_symbolic(p['shape'])),
                             self.shape_axes(p['shape']), domain_str(p['domain']),
                             self.link_role(p['role']), '; '.join(flags) or 'always',
                             describe(pdocs, p)])
            out += table(['Port', 'Shape', 'Axes', 'Domain', 'Role', 'Presence', 'Description'],
                         rows)
        return out

    def parameters(self, d, where, label):
        out = ['##### Parameters', '']
        params = d.get('parameters', {})
        if not params:
            return out + ['This primitive owns no learned tensor.', '']
        rows = []
        views = []
        for sname, s in params.items():
            sdocs = self.docs.of('parameter', s, f"{where} parameter {sname}", f"{label}.{sname}")
            for a in s['shape'].get('axes', []):
                self.vocab.add('axis nature', a['nature'], label.split('@')[0])
            sharing = s['sharing']['kind']
            self.vocab.add('parameter sharing', sharing, label.split('@')[0])
            if s['sharing'].get('roles'):
                sharing += ' with ' + ', '.join(self.link_role(r) for r in s['sharing']['roles'])
            if s['sharing'].get('note'):
                sharing += f"<br>*Note: {s['sharing']['note']}*"
            presence = cond(s['present_when']) if 'present_when' in s else 'always'
            if 'multiplicity' in s:
                presence += f"; × {expr(s['multiplicity'])}"
            rows.append([code(sname), self.link_role(s['role']),
                         code(self.shape_symbolic(s['shape'])), self.shape_axes(s['shape']),
                         sharing, presence, describe(sdocs, s)])
            for v in s.get('views', []):
                views.append(f"- `{sname}` may be viewed as " +
                             ' × '.join(self.link_axis(a) for a in v['axes']) +
                             (f" — {v['note']}" if v.get('note') else ''))
        out += table(['Slot', 'Role', 'Shape', 'Axes', 'Sharing', 'Presence / multiplicity',
                      'Description'], rows)
        if views:
            out += ['Declared views:', ''] + views + ['']
        return out

    def constants(self, d, where, label):
        consts = d.get('constants', {})
        if not consts:
            return []
        out = ['##### Constant slots', '']
        rows = []
        for cname, c in consts.items():
            cdocs = self.docs.of('constant', c, f"{where} constant {cname}", f"{label}.{cname}")
            rows.append([code(cname), self.link_role(c['role']),
                         code(self.shape_symbolic(c['shape'])), self.shape_axes(c['shape']),
                         cond(c['present_when']) if 'present_when' in c else 'always',
                         describe(cdocs, c)])
        out += table(['Slot', 'Role', 'Shape', 'Axes', 'Presence', 'Description'], rows)
        return out

    def states(self, d, where, label):
        out = ['##### State ports', '']
        states = d.get('state_ports', {})
        if not states:
            return out + ['None: this primitive carries no state. Only a sequence operator '
                          'does (§4.1).', '']
        who = label.split('@')[0]
        for sname, s in states.items():
            sdocs = self.docs.of('state', s, f"{where} state {sname}", f"{label}.{sname}")
            out += [f"**State port `{sname}`**", '']
            if 'description' in sdocs:
                out += prose(sdocs['description'])
            out += note_block(s.get('note'))
            facts = [f"- Present when: {cond(s['present_when'])}"]
            if s.get('key_axes'):
                facts.append('- Instance key axes: ' +
                             ', '.join(self.link_axis(a) for a in s['key_axes']))
            if 'carried_across' in s:
                ca = s['carried_across']
                cadocs = self.docs.of('carrying', ca, f"{where} state {sname} carried_across",
                                      f"{label}.{sname} carrying")
                extra = describe(cadocs, ca)
                facts.append(f"- Carried across fragments of its stream when: {cond(ca['when'])}"
                             + (f" — {extra}" if extra else ''))
            out += facts + ['']
            rows = []
            for cname, c in s['payload'].items():
                cdocs = self.docs.of('component', c, f"{where} state {sname} component {cname}",
                                     f"{label}.{sname}.{cname}")
                for a in c['shape'].get('axes', []):
                    self.vocab.add('axis nature', a['nature'], who)
                presence = cond(c['present_when']) if 'present_when' in c else 'always'
                if 'multiplicity' in c:
                    presence += f"; × {expr(c['multiplicity'])}"
                rows.append([code(cname), code(self.shape_symbolic(c['shape'])),
                             self.shape_axes(c['shape']), self.link_role(c['role']), presence,
                             describe(cdocs, c)])
            out += ['Payload, per indexed position:', '']
            out += table(['Component', 'Shape', 'Axes', 'Role', 'Presence', 'Description'], rows)
            rows = []
            for oname, o in s['operations'].items():
                odocs = self.docs.of('operation', o, f"{where} state {sname} operation {oname}",
                                     f"{label}.{sname}.{oname}")
                self.vocab.add('state operation effect', o['effect'], who)
                rows.append([code(oname), o['effect'], describe(odocs, o)])
            out += ['Permitted operations:', '']
            out += table(['Operation', 'Effect', 'Description'], rows)
            rows = []
            for i, r in enumerate(s['rules'], 1):
                rdocs = self.docs.of('rule', r, f"{where} state {sname} rule {i}",
                                     f"{label}.{sname}#{i}")
                self.vocab.add('state law', r['law'], who)
                self.vocab.add('state access', r['access'], who)
                self.vocab.add('state sharing', r['sharing'], who)
                extent = []
                if 'span' in r:
                    extent.append(f"span {expr(r['span'])}")
                if 'stride' in r:
                    extent.append(f"stride {expr(r['stride'])}")
                rows.append([str(i), cond(r['when']), r['law'], r['access'], r['sharing'],
                             origin_str(r['indexed_by']), '; '.join(extent) or '—',
                             describe(rdocs, r)])
            out += ['Derivation rules, in order — the first whose condition holds applies; a '
                    'state indexed by a port is frozen once that stream is complete (§5.3):', '']
            out += table(['#', 'When', 'Law', 'Access', 'Sharing', 'Indexed by', 'Extent',
                          'Description'], rows)
        return out

    def effects(self, d, where, label):
        e = d['effects']
        out = ['##### Effects', '']
        out.append('- Reads: ' + (', '.join(f"`{p}`" for p in e['reads']) or 'nothing'))
        out.append('- Writes: ' + (', '.join(f"`{p}`" for p in e['writes']) or 'nothing'))
        return out + ['']

    def cost(self, d, where, label):
        costs = d.get('logical_cost')
        if not costs:
            return []
        out = ['##### Logical cost', '',
               'Corrections to the derived cost (§4.1): every entry whose condition holds '
               'contributes, on top of two operations per weight element per element of the '
               'output domain.', '']
        rows = []
        for i, c in enumerate(costs, 1):
            cdocs = self.docs.of('cost', c, f"{where} logical_cost {i}", f"{label} cost {i}")
            self.vocab.add('cost status', c['status'], label.split('@')[0])
            self.vocab.add('cost per', c['per'], label.split('@')[0])
            rows.append([str(i), cond(c['when']) if 'when' in c else 'always',
                         code(expr(c['expression'])), c['status'], c['per'].replace('_', ' '),
                         describe(cdocs, c)])
        out += table(['#', 'When', 'Expression', 'Status', 'Per', 'Description'], rows)
        return out

    def partitions(self, d, where, label):
        out = ['##### Semantic partitions', '']
        parts = d.get('partitions', [])
        if not parts:
            return out + ['No partition is stated — the catalog grammar refuses this; a contract '
                          'says `none` when no cut preserves meaning.', '']
        rows = []
        who = label.split('@')[0]
        for i, p in enumerate(parts, 1):
            pdocs = self.docs.of('partition', p, f"{where} partition {i}", f"{label} partition {i}")
            t = p['target']
            if 'argument_axis' in t:
                target = f"axis {self.link_axis(t['argument_axis'])} (argument)"
                kind = 'argument_axis'
            elif 'value_axis' in t:
                target = f"port `{t['value_axis']['port']}` · axis {self.link_axis(t['value_axis']['axis'])} (value)"
                kind = 'value_axis'
            elif 'payload_axis' in t:
                pa = t['payload_axis']
                target = f"state `{pa['state']}` · component `{pa['component']}` · axis {self.link_axis(pa['axis'])} (payload)"
                kind = 'payload_axis'
            elif 'instance_key_axis' in t:
                target = f"instance key axis {self.link_axis(t['instance_key_axis'])}"
                kind = 'instance_key_axis'
            elif 'any_axis' in t:
                target = "any axis (elementwise)"
                kind = 'any_axis'
            else:
                target = "none: no cut preserves meaning"
                kind = 'none'
            self.vocab.add('partition target', kind, who)
            communications = p['communication'] if isinstance(p['communication'], list) else [p['communication']]
            for c in communications:
                self.vocab.add('partition communication', c, who)
            rows.append([target, ' or '.join(communications),
                         expr(p['granularity']) if 'granularity' in p else '1',
                         cond(p['when']) if 'when' in p else 'always', describe(pdocs, p)])
        out += table(['Target', 'Communication', 'Granularity', 'When', 'Description'], rows)
        return out

    def sparsity(self, d, where, label):
        sp = d.get('sparsity')
        if not sp:
            return []
        out = ['##### Structured sparsity', '',
               'Activatable units (§4.5): a lookup table is the limiting case, one row per '
               'element.', '']
        rows = []
        for i, u in enumerate(sp, 1):
            udocs = self.docs.of('sparsity', u, f"{where} sparsity {i}", f"{label} sparsity {i}")
            unit = u['unit']
            policy = u['policy']
            if 'argument' in policy:
                selector = f"argument `{policy['argument']}`"
            elif 'port' in policy:
                selector = f"the value on port `{policy['port']}`"
            else:
                selector = "the element itself"
            self.vocab.add('sparsity policy', next(iter(policy)), label.split('@')[0])
            ub = u['union_per_invocation']
            self.vocab.add('cost status', ub['status'], label.split('@')[0])
            rows.append([str(i), ', '.join(code(p) for p in unit['parameters'])
                         + f" along {self.link_axis(unit['axis'])}", selector,
                         code(expr(u['activated_per_element'])),
                         f"{code(expr(ub['expression']))} ({ub['status']})", describe(udocs, u)])
        out += table(['#', 'Unit', 'Selected by', 'Activated per element',
                      'Union per invocation', 'Description'], rows)
        return out

    def transforms(self, d, where, label):
        ts = d.get('domain_transforms')
        if not ts:
            return []
        out = ['##### Domain transforms', '']
        rows = []
        for i, t in enumerate(ts, 1):
            tdocs = self.docs.of('transform', t, f"{where} domain_transform {i}",
                                 f"{label} transform {i}")
            self.vocab.add('domain transform relation', t['relation'], label.split('@')[0])
            rows.append([code(t['from_port']), code(t['to_port']), t['relation'],
                         code(expr(t['factor'])) if 'factor' in t else '—', describe(tdocs, t)])
        out += table(['From port', 'To port', 'Relation', 'Factor', 'Description'], rows)
        return out

    def check_conditions(self, d, where):
        """A condition or expression citing an argument the contract does not
        declare can never fire (present) or resolve (compare): worth knowing."""
        declared = declared_argument_paths(d['arguments'])
        sites = []
        for side in ('inputs', 'outputs'):
            for pname, p in d['ports'][side].items():
                sites.append((f"port {pname}", p))
        for section in ('parameters', 'constants', 'state_ports'):
            for sname, s in d.get(section, {}).items():
                sites.append((f"{section} {sname}", s))
        for i, p in enumerate(d.get('partitions', []), 1):
            sites.append((f"partition {i}", p))
        for i, c in enumerate(d.get('logical_cost', []), 1):
            sites.append((f"logical_cost {i}", c))
        for i, u in enumerate(d.get('sparsity', []), 1):
            sites.append((f"sparsity {i}", u))
        for aname, a in d['arguments'].items():
            if 'default' in a:
                sites.append((f"argument {aname} default", a['default']))
        for label, node in sites:
            for ref in sorted(arguments_cited(node) - declared):
                self.report.find(f"{where} {label}",
                                 f"cites argument '{ref}', which this contract does not "
                                 f"declare — the condition can never hold")

    def template_section(self, u, where):
        d = u['definition']
        m = d['template']
        out = ['##### Template', '']
        out.append(f"- Template `{m['name']}`, model id `{m['id']}`"
                   + (f", template version `{m['version']}`" if m.get('version') else ''))
        if m.get('note'):
            out.append(f"- *Note: {m['note']}*")
        found = find_template(self.cat, d)
        if found is None:
            out.append(f"- Template `{m['name']}` {m['version']} not found where the base "
                       f"declares its templates.")
            self.report.find(where, "template not found at the declared location")
            return out + ['']
        path, model = found
        declared_id = model.get('model', '?')
        out.append(f"- Resolved to `{self.rel(path)}`"
                   + (f" (declares model id `{declared_id}`)" if declared_id != m['id'] else ''))
        if declared_id != m['id']:
            self.report.find(where, f"template declares model id '{declared_id}', contract says '{m['id']}'")
        out.append('')
        rows = []
        for qname, q in model.get('quantities', {}).items():
            src = q.get('source', {})
            if src.get('kind') != 'external':
                continue
            t = q['type']
            tdesc = t['kind'] + (': ' + ', '.join(f"`{json.dumps(v)}`" for v in t['values']) if t['kind'] == 'enum' else '')
            rows.append([code(qname), tdesc, self.quantity_domain(q.get('domain')),
                         code(expr(src['default'])) if 'default' in src else ''])
        out += ['Arguments — the external quantities of the template (§4.6), supplied by an '
                'assignment at the call site:', '']
        out += table(['Argument', 'Type', 'Domain', 'Default'], rows) if rows else ['none', '']
        rows = []
        for side in ('inputs', 'outputs'):
            for pname, p in model.get('interfaces', {}).get(side, {}).items():
                if side == 'inputs':
                    dom = p['kind'] + (f", joins `{p['stream']}`" if 'stream' in p else '') \
                        + (', fragmented' if p.get('fragmented') else '')
                else:
                    dom = 'derived from the port (§5.3)'
                rows.append([side[:-1], code(pname), dom,
                             'yes' if p.get('generative') else ('no' if side == 'outputs' else '—')])
        out += ['Ports — the public interfaces of the template:', '']
        out += table(['Side', 'Port', 'Domain', 'Generative'], rows)
        closure = self.closure(model)
        out += ['Contracts the template cites, transitively — the capabilities a consumer needs '
                '(§8.1), never the composite itself:', '']
        for name, version in sorted(closure):
            out.append(f"- {self.link_contract(name, version)}")
        out += ['', 'Parameter slots, state ports, logical cost and semantic partitions are '
                'derived from the expanded template (§4.6, D3–D6); this document does not '
                'expand templates.', '']
        return out

    def closure(self, model, seen=None):
        seen = seen if seen is not None else set()
        refs = set()
        for o in model.get('occurrences', {}).values():
            refs.add((o['contract']['name'], o['contract']['version']))
        for comp in model.get('compositions', {}).values():
            for o in comp['occurrences'].values():
                refs.add((o['contract']['name'], o['contract']['version']))
        for ref in refs:
            if ref in seen:
                continue
            seen.add(ref)
            d = self.cat['contracts'].get(ref[0])
            if d is not None and 'template' in d:
                found = find_template(self.cat, d)
                if found is not None:
                    self.closure(found[1], seen)
        return seen

    def quantity_domain(self, dom):
        if not dom:
            return '—'
        if dom.get('kind') == 'set':
            return '{' + ', '.join(json.dumps(v) for v in dom['values']) + '}'
        if dom.get('kind') == 'interval':
            lo, hi = dom.get('lower'), dom.get('upper')
            left = ('[' if lo and lo.get('inclusive') else '(') + (expr(lo['value']) if lo else '-∞')
            right = (expr(hi['value']) if hi else '∞') + (']' if hi and hi.get('inclusive') else ')')
            return code(f"{left}, {right}")
        return code(json.dumps(dom))

    # -- axes and roles --------------------------------------------------------------------
    def unit_details(self, u, docs, where):
        """The prose of an axis or a role, below its table row."""
        if not any(k in docs for k in ('description', 'external_docs', 'deprecated', 'tags')):
            return []
        out = [f"**`{u['name']}`**", ''] + self.deprecated_lines(docs, where)
        if 'description' in docs:
            out += prose(docs['description'])
        out += self.tags_line(docs, u['name'])
        out += external_docs_lines(docs, self.rewrite_url)
        return out

    def axes_section(self):
        out = heading(2, 'Axes', 'axes')
        out += ["An axis names a dimension. Shapes unify by axis identity (V4): the same "
                "extent on two different axes is two different things. `value` axes index "
                "tensors; `instance` axes key state instances (session, branch).", '']
        rows = []
        details = []
        for u in self.axes:
            d = u['definition']
            where = self.rel(u['path'])
            docs = self.docs.of('axis', d, where, u['name'], primary='summary')
            self.vocab.add('axis space', d['space'], u['name'])
            rows.append([f'<a id="{anchor("axis", u["name"])}"></a>`{u["name"]}`', d['space'],
                         describe({'description': docs['summary']} if 'summary' in docs else {}, d)])
            details += self.unit_details(u, docs, where)
        out += table(['Axis', 'Space', 'Summary'], rows)
        if details:
            out += ['### Details', ''] + details
        return out

    def roles_section(self):
        out = heading(2, 'Precision roles', 'precision-roles')
        out += ["A precision role bounds the storage types a slot, port or state component may "
                "take. A model that selects a dtype outside the admissible set is refused "
                "(V14); a model that selects none gets the default.", '']
        rows = []
        details = []
        for u in self.roles:
            d = u['definition']
            where = self.rel(u['path'])
            docs = self.docs.of('precision_role', d, where, u['name'], primary='summary')
            self.vocab.add('precision sensitivity', d.get('sensitivity'), u['name'])
            for t in d['admissible']:
                self.vocab.add('dtype admissible', t, u['name'])
            rows.append([f'<a id="{anchor("role", u["name"])}"></a>`{u["name"]}`',
                         ', '.join(f"`{t}`" for t in d['admissible']), code(d['default']),
                         d.get('sensitivity', '—'),
                         describe({'description': docs['summary']} if 'summary' in docs else {}, d)])
            details += self.unit_details(u, docs, where)
        out += table(['Role', 'Admissible', 'Default', 'Sensitivity', 'Summary'], rows)
        if details:
            out += ['### Details', ''] + details
        return out

    def tags_section(self):
        if not self.declared_tags:
            return []
        out = heading(2, 'Tags', 'tags')
        out += ['Editorial groupings declared by the base manifest; a unit that carries a tag '
                'says so in its own section.', '']
        for t in sorted(self.declared_tags):
            decl = self.declared_tags[t]
            out += heading(3, f"`{t}`", anchor('tag', t))
            if 'summary' in decl:
                out += [f"**{decl['summary']}**", '']
            if 'description' in decl:
                out += prose(decl['description'])
            out += external_docs_lines(decl, self.rewrite_url)
        return out

    # -- appendices --------------------------------------------------------------------------
    def appendix(self):
        out = heading(2, 'Appendix A — Closed vocabulary in use', 'appendix-a')
        out += ['Every value of every closed enumeration that at least one unit uses, and how '
                'many units use it. A runtime that implements these values implements the '
                'whole catalog as it stands.', '']
        out += table(['Field', 'Value', 'Units'],
                     [[f, code(v), str(n)] for f, v, n in self.vocab.rows()])

        out += heading(2, 'Appendix B — Documentation coverage', 'appendix-b')
        out += ['Sites that carry a `summary` (units) or a `description` (elements). A missing '
                'entry is rendered as absent, never invented.', '']
        rows = []
        for site_kind, (done, total) in self.report.coverage.items():
            pct = f"{100 * done / total:.0f}%" if total else '—'
            rows.append([site_kind, str(done), str(total), pct])
        out += table(['Site', 'Documented', 'Total', 'Coverage'], rows)
        for site_kind, labels in self.report.undocumented.items():
            if site_kind in ('contract', 'axis', 'precision_role', 'base'):
                out.append(f"- Undocumented {site_kind}s: " + ', '.join(f"`{l}`" for l in labels))
        out.append('')

        out += heading(2, 'Appendix C — Findings', 'appendix-c')
        if self.report.findings:
            out += ['Legal, and worth knowing. None of these blocks generation.', '']
            for f in self.report.findings:
                out.append(f"- {f}")
            out.append('')
        else:
            out += ['Nothing to report.', '']
        return out


# --- entry point --------------------------------------------------------------------------

def run(model_paths, catalog_bases, schema_dir, output=None, relative_to=None,
        models_base=None, output_dir=None):
    """Render the catalog of the given bases to Markdown. Returns the exit status:
    0 written, 1 refused (malformed documentation, unreadable catalog), with the
    causes on stderr. Templates are resolved where each base declares them."""
    status = sys.stderr if output is None else sys.stdout
    try:
        cat = catalog_mod.load(*catalog_bases, models_base=models_base)
        manifests, units = load_units(catalog_bases)
    except (ValueError, OSError, KeyError, catalog_mod.CatalogError) as e:
        print(f"  catalog not readable: {e}", file=sys.stderr)
        return 1
    report = Report()
    try:
        docs = Docs(schema_dir, report)
    except FileNotFoundError as e:
        print(f"  {e}", file=sys.stderr)
        return 1
    target = None
    if output is not None:
        target = os.path.join(output, 'catalog.md') if os.path.isdir(output) else output
    output_dir = (os.path.abspath(output_dir) if output_dir else
                  os.path.dirname(os.path.abspath(target)) if target else None)
    renderer = Renderer(manifests, units, cat, model_paths, docs, report, catalog_bases,
                        relative_to, output_dir)
    text = renderer.render()
    if report.refusals:
        print(f"  refused: {len(report.refusals)} documentation error(s)", file=sys.stderr)
        for line in report.refusals:
            print(f"    {line}", file=sys.stderr)
        return 1
    if target is None:
        sys.stdout.write(text)
        target = '<stdout>'
    else:
        with open(target, 'w', encoding='utf-8') as f:
            f.write(text)
    covered = report.coverage.get('contract', [0, 0])
    print(f"  {len(renderer.contracts)} contracts, {len(renderer.axes)} axes, "
          f"{len(renderer.roles)} precision roles -> {target}", file=status)
    print(f"  {covered[0]}/{covered[1]} contracts carry a summary; "
          f"{len(report.findings)} advisory finding(s)", file=status)
    return 0
