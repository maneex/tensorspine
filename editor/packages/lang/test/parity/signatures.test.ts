import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  analyseText,
  expandText,
  formatSemanticProblems,
  items,
  loadLibrary,
  member,
  parse,
  toPython,
  type Analysis,
  type Library,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { nodeSource } from '../library/source.js';
import { repositorySchemas } from '../schema/repository.js';
import { decodeRecord } from './encoding.js';
import { counted, wlHash } from './signature.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// The signature suite (feature 1.7): every figure of `tests/signatures/*.json`, reproduced.
//
// `tests/signature.py` records, per corpus document, what the *denoted graph* is — "what must not
// change when the document is rewritten while the denoted graph stays the same" — and it is the
// one comparison in the repository that is blind to identifiers: nodes, edges and per-primitive
// counts, a Weisfeiler-Lehman hash of the value graph over four refinement rounds, and five of the
// validator's counters with the multiset of derived instance keys beside them.
//
// Two implementations meet here, and both are this feature's business: the graph comes from
// `expand` (D1) and the counters from `analyse` (features 1.6a–1.6c), exactly as `signature()`
// takes `d1.emit` and `validate.analyse`. The hash itself is `signature.ts`, the Python ported for
// the test layer — sixteen hex digits of a SHA-256 over CPython's own `json.dumps`, which is
// unforgiving: one wrong separator, one unsorted key, one `1` written where the document says
// `1.0`, and every figure differs.
//
// The expectations are `tests/signatures/*.json` as the repository records them, copied by the
// oracle. They are not regenerated here: they are the recorded denotation, and a document whose
// signature moved is a change of meaning, which is what the suite exists to catch.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

/** One recorded signature, as `tests/signatures/<name>.json` writes it. */
interface Signature {
  nodes: number;
  edges: number;
  per_primitive: Record<string, number>;
  wl: string;
  parameter_slots: number;
  tensors: number;
  shared: number;
  state_slots: number;
  state_identities: number;
  instance_keys: Record<string, number>;
}

const schemas = repositorySchemas();
const source = nodeSource(repositoryRoot);

/**
 * The reference base, loaded once.
 *
 * `signature()` loads `data/primitive-library` itself rather than the bases the document declares
 * — every corpus document pins that one — so the suite reads it the same way.
 */
const reference: Library = loadLibrary(['data/primitive-library'], { schemas, source });

/**
 * The assignments the fixture was recorded under, from the `expansion` step's own records.
 *
 * The fixture writes them in the tagged encoding of feature 1.2, so `layers` comes back an integer
 * and `eps` a float — the distinction the signature's hash turns on.
 */
function assignments(): Map<string, PyRecord | null> {
  const text = readFileSync(join(oracleOut, 'expansion', 'index.json'), 'utf8');
  const recorded = toPython(parse(text)) as unknown as {
    documents: { name: string; path: string; assignment: PyRecord | null }[];
  };
  return new Map(
    recorded.documents
      .filter((one) => one.path.startsWith('data/models/'))
      .map((one) => [one.name, one.assignment === null ? null : decodeRecord(one.assignment)]),
  );
}

/** A counter of `analyse`'s, as a number: every one the signature reads is a whole number. */
function counter(analysis: Analysis, name: string): number {
  const held = analysis.stats.get(name);
  if (typeof held !== 'bigint') throw new TypeError(`${name} is not a count, but a ${typeof held}`);
  return Number(held);
}

/** `signature(model_path)`: the port's answer for one document. */
function signatureOf(path: string, assignment: PyRecord | null): Signature {
  const text = readFileSync(join(repositoryRoot, path), 'utf8');
  const options = assignment === null ? {} : { assignment };
  const graph = member(expandText(text, reference, options), 'd1') as PyValue;
  const analysis = analyseText(text, reference, options);
  // `raise ValueError(f"{name}: not valid, no signature: …")` — a signature is a valid document's.
  expect(formatSemanticProblems([...analysis.problems]), path).toEqual([]);
  const nodes = items(member(graph as PyRecord, 'nodes') as PyValue);
  return {
    nodes: nodes.length,
    edges: (member(graph as PyRecord, 'edges') as readonly PyValue[]).length,
    per_primitive: counted(
      nodes.map(
        ([, node]) =>
          member(member(node as PyRecord, 'primitive') as PyRecord, 'name') as string,
      ),
    ),
    wl: wlHash(graph),
    parameter_slots: counter(analysis, 'parameter_slots'),
    tensors: counter(analysis, 'tensors'),
    shared: counter(analysis, 'shared'),
    state_slots: counter(analysis, 'state_slots'),
    state_identities: counter(analysis, 'state_identities'),
    instance_keys: counted(
      [...analysis.bindings.instanceKeys.values()].map((key) =>
        key.map((axis) => axis as string).join('×'),
      ),
    ),
  };
}

describe('the signature suite', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('reproduces every recorded figure', () => {
    const manifest = readOracleManifest();
    const supplied = assignments();
    expect(manifest.signatures).toHaveLength(15);
    for (const file of manifest.signatures) {
      const name = file.slice('signatures/'.length, -'.json'.length);
      const expected = JSON.parse(readFileSync(join(oracleOut, file), 'utf8')) as Signature;
      const document = manifest.documents.find((one) => one.name === name);
      expect(document, name).toBeDefined();
      const path = (document as { path: string }).path;
      const assignment = supplied.get(name);
      expect(assignment, name).not.toBeUndefined();
      expect(signatureOf(path, assignment ?? null), name).toEqual(expected);
    }
  }, 600_000);

  it.skipIf(!generated)('reads the figures the suite is there to hold', () => {
    // The three the fixtures make easy to get wrong: the template is read under an assignment, the
    // composite's `instance_keys` are empty because an expanded template's are not merged into the
    // caller's (`analyse` merges the counters and not the keys), and `shared` is non-zero
    // somewhere — a tie is what a signature must see through a renaming.
    const figures = new Map(
      readOracleManifest().signatures.map((file) => [
        file.slice('signatures/'.length, -'.json'.length),
        JSON.parse(readFileSync(join(oracleOut, file), 'utf8')) as Signature,
      ]),
    );
    expect(assignments().get('decoder-causal-yarn@1.0.0')).not.toBeNull();
    expect(figures.get('shieldstral-3b-composite')?.instance_keys).toEqual({});
    expect(figures.get('shieldstral-3b-composite')?.state_slots).toBe(26);
    expect(figures.get('gemma3n-kvshare')?.shared).toBeGreaterThan(0);
    // Two documents denoting one graph: the composite built from the template and the flat one.
    const flat = figures.get('shieldstral-3b') as Signature;
    const composite = figures.get('shieldstral-3b-composite') as Signature;
    expect(composite.nodes).toBe(flat.nodes);
    expect(composite.edges).toBe(flat.edges);
    expect(composite.per_primitive).toEqual(flat.per_primitive);
  });
});
