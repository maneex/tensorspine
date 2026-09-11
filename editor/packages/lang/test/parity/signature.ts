import { createHash } from 'node:crypto';

import {
  comparePythonStrings,
  formatNumber,
  isRecord,
  items,
  member,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';

/**
 * `tests/signature.py`, ported for the parity suite alone.
 *
 * The signature is "what must not change when the document is rewritten — sugar rearranged, sites
 * merged under `when`, bindings scoped — while the denoted graph stays the same (§1.1: functional
 * denotation is about the graph, not the text)". It is computed from D1 and the validator's
 * derivations and *never from identifiers*, which is what makes it the one figure that survives a
 * renaming: §5.2 rule 2 calls identifiers representation, and two expanded graphs are the same
 * when a one-to-one correspondence of instances preserves primitives, arguments, edges and
 * identities.
 *
 * It lives in the test layer, not in `packages/lang`: `tests/signature.py` is one of the
 * repository's own suites, not a tool the editor ports (plan §5.3 lists what is ported), and
 * nothing in the editor reads a Weisfeiler-Lehman hash. What the core *does* supply is everything
 * the hash is taken over — the expanded graph and the counters — so the only thing written twice
 * is the hashing itself, and it is written against the Python line by line.
 *
 * The one delicate part is `_h`: `sha256(json.dumps(x, sort_keys=True, default=str))` truncated to
 * sixteen hex digits. Reproducing it means reproducing CPython's `json.dumps` exactly — the
 * default separators `", "` and `": "`, `ensure_ascii=True`, keys sorted by code point, `bool`
 * before `int` (a Python `bool` is an `int`), and `repr(float)` for a float, which is the
 * formatter feature 0.3 built. A digest is unforgiving: any of those wrong and every hash differs.
 */

/** `json.dumps(value, sort_keys=True)`, CPython's own defaults. */
export function pythonDumps(value: PyValue): string {
  if (value === null) return 'null';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'bigint') return value.toString();
  if (typeof value === 'number') {
    // `json.encoder.floatstr`: the three non-finite names, and `repr(float)` for everything else.
    if (Number.isNaN(value)) return 'NaN';
    if (value === Infinity) return 'Infinity';
    if (value === -Infinity) return '-Infinity';
    return formatNumber(value, true);
  }
  if (typeof value === 'string') return escapeAscii(value);
  if (Array.isArray(value)) {
    return `[${(value as readonly PyValue[]).map((one) => pythonDumps(one)).join(', ')}]`;
  }
  if (isRecord(value)) {
    const names = Object.keys(value).sort(comparePythonStrings);
    const written = names.map(
      (name) => `${escapeAscii(name)}: ${pythonDumps(member(value, name) as PyValue)}`,
    );
    return `{${written.join(', ')}}`;
  }
  // `default=str` never fires on what a signature is taken over; a value that reached here is a
  // defect of the caller, not something to stringify quietly.
  throw new TypeError(`a signature carries no ${typeof value}`);
}

/**
 * `py_encode_basestring_ascii`: `ESCAPE_ASCII = re.compile(r'([\\"]|[^ -~])')`, so everything
 * outside the printable ASCII range is escaped, DEL included.
 *
 * The walk is over UTF-16 *code units*, not code points, because that is what Python writes for a
 * character above the basic plane: its two surrogates, one `\uXXXX` each.
 */
function escapeAscii(text: string): string {
  let out = '"';
  for (let at = 0; at < text.length; at += 1) {
    const unit = text[at] as string;
    const code = text.charCodeAt(at);
    if (unit === '"') out += '\\"';
    else if (unit === '\\') out += '\\\\';
    else if (unit === '\n') out += '\\n';
    else if (unit === '\r') out += '\\r';
    else if (unit === '\t') out += '\\t';
    else if (unit === '\b') out += '\\b';
    else if (unit === '\f') out += '\\f';
    else if (code >= 0x20 && code <= 0x7e) out += unit;
    else out += `\\u${code.toString(16).padStart(4, '0')}`;
  }
  return `${out}"`;
}

/** `_h`: the first sixteen hex digits of the SHA-256 of that text. */
export function shortHash(value: PyValue): string {
  return createHash('sha256').update(pythonDumps(value), 'utf8').digest('hex').slice(0, 16);
}

/** One end of an emitted edge, as the hash reads it: `(from port, to port, the other node)`. */
type Incidence = [string, string, string];

/**
 * `wl_hash`: a Weisfeiler-Lehman hash of the value graph — nodes labelled by primitive and
 * resolved arguments, edges by their ports, four refinement rounds.
 *
 * A node's label after a round is the hash of its previous label and the *sorted* labels of what
 * it feeds and what feeds it, so the answer depends on the graph and not on the listing; the whole
 * graph's is the hash of the sorted node labels.
 */
export function wlHash(graph: PyValue, rounds = 4): string {
  const nodes = items(member(graph as PyRecord, 'nodes') as PyValue);
  let label = new Map<string, string>(
    nodes.map(([name, node]) => [
      name,
      shortHash([
        member(member(node as PyRecord, 'primitive') as PyRecord, 'name') as PyValue,
        member(node as PyRecord, 'arguments') as PyValue,
      ]),
    ]),
  );
  const out = new Map<string, Incidence[]>(nodes.map(([name]) => [name, []]));
  const into = new Map<string, Incidence[]>(nodes.map(([name]) => [name, []]));
  for (const edge of member(graph as PyRecord, 'edges') as readonly PyValue[]) {
    const from = member(edge as PyRecord, 'from') as PyRecord;
    const to = member(edge as PyRecord, 'to') as PyRecord;
    const fromNode = member(from, 'node') as string;
    const toNode = member(to, 'node') as string;
    const fromPort = member(from, 'port') as string;
    const toPort = member(to, 'port') as string;
    (out.get(fromNode) as Incidence[]).push([fromPort, toPort, toNode]);
    (into.get(toNode) as Incidence[]).push([fromPort, toPort, fromNode]);
  }
  for (let round = 0; round < rounds; round += 1) {
    const refined = new Map<string, string>();
    for (const [name] of nodes) {
      refined.set(
        name,
        shortHash([
          label.get(name) as string,
          relabelled(out.get(name) as Incidence[], label),
          relabelled(into.get(name) as Incidence[], label),
        ]),
      );
    }
    label = refined;
  }
  return shortHash([...label.values()].sort(comparePythonStrings));
}

/** `sorted((p, q, label[m]) for p, q, m in …)`: tuples of three strings, compared member by member. */
function relabelled(incidences: readonly Incidence[], label: ReadonlyMap<string, string>): PyValue {
  const relabel = incidences.map(
    ([from, to, other]) => [from, to, label.get(other) as string] as Incidence,
  );
  relabel.sort(compareTriples);
  return relabel;
}

function compareTriples(one: Incidence, other: Incidence): number {
  for (let at = 0; at < 3; at += 1) {
    const order = comparePythonStrings(one[at] as string, other[at] as string);
    if (order !== 0) return order;
  }
  return 0;
}

/** `Counter(…)` read out as `dict(sorted(counter.items()))`. */
export function counted(values: readonly string[]): Record<string, number> {
  const tally = new Map<string, number>();
  for (const one of values) tally.set(one, (tally.get(one) ?? 0) + 1);
  return Object.fromEntries(
    [...tally].sort((one, other) => comparePythonStrings(one[0], other[0])),
  );
}
