import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  analyse,
  analyseText,
  formatSemanticProblems,
  keyOf,
  loadLibrary,
  toPython,
  whereOfSite,
  type Analysis,
  type EvaluatedLocation,
  type Library,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource } from '../library/source.js';
import { applyEdits, type Edit } from './edits.js';
import { decode, decodeRecord, encode } from './encoding.js';
import { raisedAs, type Raised } from './raised.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity of the bindings (feature 1.6c): the whole of `validate.analyse`, which feature 1.6b left
// open at the comment that opens the parameter bindings. What is compared here is what the tools
// answer *in full* — the refusals of V7, V9, V14, V15, V16, V17, V18 and V20 in their words and
// their order, the seventeen counters `--validate` prints, the advisories `--lint` prints, and the
// four answers the bindings add: the instance keys of §4.4, what is carried across the fragments of
// a stream (§5.3), the physical names the document binds, and the identity instances D3 and D4 read.
//
// The suite has two halves.
//
// The **documents** are the repository's own: the corpus, the template under its documented
// assignment, and the 73 of `tests/rejections/models/` — the last read under
// `data/primitive-library`, which is what `tests/run_rejections.py` hands `validate.semantic` (21
// of them declare `../primitive-library/`, which from `tests/rejections/models/` names the
// directory of rejection *bases* and gathers nothing: feature 1.6a's finding).
//
// The **cases** are edited documents, in the idiom features 1.3, 1.4 and 1.5 established: a
// pointer into a repository file with a few values changed, so both implementations read the same
// bytes and apply the same change. They exist because the repository reaches twenty-two of the
// thirty-five refusals these blocks carry and produces no advisory at all;
// `tests/oracle/binding_cases.py` says which branch each one is for, and which branches no edit of
// a corpus document can reach at all — those are the unit suite's, over a base built for them.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

/** A location as the fixture writes one: the same shape, its integers tagged. */
type RecordedLocation = PyValue;

/** One parameter or state identity instance, as the fixture writes one. */
interface RecordedIdentity {
  identity: string;
  rule: string;
  members: [string, string][];
  dtype: PyValue;
  location?: RecordedLocation;
  indices?: string[];
  writer?: [string, string] | null;
}

/** The facts every document and every case carries. */
interface RecordedFacts {
  read?: boolean;
  /**
   * Whether a line the fixture carries had the sentinel's address elided.
   *
   * `repr(expr.UNRESOLVED)` is `<object object at 0x…>`, and the address changes between two runs
   * of the *tools* — measured: three runs of `analyse` on `v3-streams-below-two.json` print three
   * addresses. One line of the bindings can be handed the sentinel (a slot whose multiplicity
   * reads an argument V3 refused), so the oracle elides the address and the port writes
   * `<object object>` (feature 1.1's `pyRepr`). The rest of the line is contract; the address is
   * a coin flip no implementation can be at parity with, as feature 1.6b's `next(iter(agree))` is.
   */
  elided_sentinel?: boolean;
  /**
   * Whether an instance of this document took an *arbitrary* own domain.
   *
   * `mine = next(iter(agree))` reads a Python `set` (feature 1.6b's finding), and at this stage
   * the coin flip reaches the *refusals*: measured on `v5-fusion-without-join.json`,
   * `PYTHONHASHSEED=0` prints 131 lines with 26 of V16 where seed 3 prints 105 with none, because
   * one reading puts the state on a fragmented stream and the other does not. The fixture records
   * no `errors`, no `advisories` and no `carried` for such a document; everything else it records
   * is independent of the choice.
   */
  arbitrary_own?: boolean;
  errors?: string[];
  stats?: Record<string, PyValue>;
  advisories?: string[];
  instance_keys?: [string, string[]][];
  carried?: [string, [string, string] | null][];
  physical?: {
    whole: [string, string][];
    slices: [string, [PyValue, PyValue, string][]][];
  };
  slots?: [string, string, string][];
  state_slots?: [string, string, string][];
  tensors?: RecordedIdentity[];
  states?: RecordedIdentity[];
}

/** One document of the repository, analysed in full. */
interface DocumentCase extends RecordedFacts {
  name: string;
  path: string;
  bases: string[];
  assignment: PyRecord | null;
  error: Raised | null;
}

/** One edited document: the source, the edits, and what the tools answered. */
interface EditedCase extends RecordedFacts {
  name: string;
  source: string;
  edits: Edit[];
  error: Raised | null;
}

/** The fixture, read as a document is read: `1e-05` a float, `4096` a whole number. */
function recorded(): { documents: DocumentCase[]; cases: EditedCase[] } {
  const text = readFileSync(join(oracleOut, 'bindings', 'index.json'), 'utf8');
  return toPython(parse(text)) as unknown as { documents: DocumentCase[]; cases: EditedCase[] };
}

const schemas = repositorySchemas();
const source = nodeSource(repositoryRoot);
const libraries = new Map<string, Library>();

function libraryFor(bases: readonly string[]): Library {
  const key = bases.join('|');
  const held = libraries.get(key);
  if (held !== undefined) return held;
  const library = loadLibrary(bases, { schemas, source });
  libraries.set(key, library);
  return library;
}

/** The reference base: what every edited case is read under, as the oracle reads them. */
const REFERENCE = ['data/primitive-library'];

/**
 * What the port answers for one recorded document.
 *
 * Read from its *text*, as the tools are given a path: a duplicate member name is inside the
 * refusal `analyse` catches, and is reported as V12 rather than raised (feature 0.3, feature 1.4).
 */
function analysedDocument(one: DocumentCase): Analysis {
  const text = readFileSync(join(repositoryRoot, one.path), 'utf8');
  return analyseText(text, libraryFor(one.bases), {
    ...(one.assignment === null ? {} : { assignment: decodeRecord(one.assignment) }),
  });
}

/** The same for an edited case: the source read, the fixture's own edits applied. */
function analysedCase(one: EditedCase): Analysis {
  const text = readFileSync(join(repositoryRoot, one.source), 'utf8');
  const document = toPython(parse(text));
  applyEdits(document, one.edits, (value) => decode(value as PyValue), one.name);
  return analyse(document, libraryFor(REFERENCE));
}

/** A location as the fixture writes one, from the port's own. */
function locationOf(location: EvaluatedLocation | undefined): unknown {
  if (location === undefined) return null;
  if ('tensor' in location) return { tensor: location.tensor };
  if ('slice' in location) {
    const part = location.slice;
    return {
      slice: {
        tensor: part.tensor,
        axis: part.axis,
        dim: encode(part.dim),
        offset: encode(part.offset),
        extent: encode(part.extent),
      },
    };
  }
  const form = 'stack' in location ? 'stack' : 'concat';
  const part = 'stack' in location ? location.stack : location.concat;
  return {
    [form]: {
      axis: part.axis,
      dim: encode(part.dim),
      parts: part.parts.map((one) => locationOf(one)),
    },
  };
}

/** A counter map as the fixture writes one. */
function statsOf(stats: ReadonlyMap<string, PyValue>): unknown {
  return Object.fromEntries([...stats].map(([name, one]) => [name, encode(one)]));
}

/**
 * A name the fixture writes bare: a slot, a state port, an axis, a stream, a kind.
 *
 * Every one of them is an identifier or an enum value of the grammar, so the fixture writes it as
 * text; a value that is not one is a defect this reading names rather than passes on.
 */
function label(value: PyValue): string {
  if (typeof value !== 'string') {
    throw new TypeError(`a slot, port, axis, kind or stream name is a string, not ${typeof value}`);
  }
  return value;
}

/** The whole answer in the fixture's shape. */
function answered(answer: Analysis, arbitrary = false): Record<string, unknown> {
  const bindings = answer.bindings;
  const settled = <T>(value: T): T | undefined => (arbitrary ? undefined : value);
  return {
    errors: settled(formatSemanticProblems([...answer.problems])),
    stats: statsOf(answer.stats),
    advisories: settled([...answer.advisories]),
    instance_keys: [...bindings.instanceKeys].map(([key, axes]) => [key, axes.map(label)]),
    carried: settled(
      [...bindings.carried].map(([rule, mine]) => [
        rule,
        mine === null ? null : [label(mine[0]), label(mine[1])],
      ]),
    ),
    physical: {
      whole: [...bindings.physical.whole].map(([name, identity]) => [name, identity]),
      slices: [...bindings.physical.slices].map(([name, regions]) => [
        name,
        regions.map((region) => [encode(region.offset), encode(region.extent), region.identity]),
      ]),
    },
    slots: [...bindings.slots.values()].map((slot) => [
      whereOfSite(slot.site),
      label(slot.name),
      slot.rule,
    ]),
    state_slots: [...bindings.stateSlots.values()].map((slot) => [
      whereOfSite(slot.site),
      label(slot.name),
      slot.rule,
    ]),
    tensors: bindings.tensorInstances.map((one) => ({
      identity: one.identity,
      rule: one.rule,
      members: one.members.map((member) => [whereOfSite(member.site), label(member.name)]),
      dtype: one.dtype,
      location: locationOf(one.location),
    })),
    states: bindings.stateInstances.map((one) => ({
      identity: one.identity,
      rule: one.rule,
      members: one.members.map((member) => [whereOfSite(member.site), label(member.name)]),
      dtype: one.dtype,
      indices: [...one.indices],
      writer:
        one.writer === null ? null : [whereOfSite(one.writer.site), label(one.writer.name)],
    })),
  };
}

/** The same reading of a recorded document or case, so the two are compared field by field. */
function expected(one: RecordedFacts): Record<string, unknown> {
  const arbitrary = one.arbitrary_own === true;
  const settled = <T>(value: T): T | undefined => (arbitrary ? undefined : value);
  return {
    errors: settled(one.errors ?? []),
    stats: one.stats ?? {},
    advisories: settled(one.advisories ?? []),
    instance_keys: (one.instance_keys ?? []).map((row) => [row[0], [...row[1]]]),
    carried: settled(
      (one.carried ?? []).map((row) => [row[0], row[1] === null ? null : [...row[1]]]),
    ),
    physical: {
      whole: (one.physical?.whole ?? []).map((row) => [...row]),
      slices: (one.physical?.slices ?? []).map((row) => [
        row[0],
        row[1].map((region) => [...region]),
      ]),
    },
    slots: (one.slots ?? []).map((row) => [...row]),
    state_slots: (one.state_slots ?? []).map((row) => [...row]),
    tensors: (one.tensors ?? []).map((identity) => ({
      identity: identity.identity,
      rule: identity.rule,
      members: identity.members.map((member) => [...member]),
      dtype: identity.dtype,
      location: identity.location ?? null,
    })),
    states: (one.states ?? []).map((identity) => ({
      identity: identity.identity,
      rule: identity.rule,
      members: identity.members.map((member) => [...member]),
      dtype: identity.dtype,
      indices: [...(identity.indices ?? [])],
      writer: identity.writer === undefined ? null : identity.writer,
    })),
  };
}

describe('the bindings against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('records every model document of the repository, and the edited cases', () => {
    const { documents, cases } = recorded();
    const manifest = readOracleManifest() as unknown as {
      bindings?: { documents: number; cases: number };
    };
    expect(manifest.bindings?.documents).toBe(documents.length);
    expect(manifest.bindings?.cases).toBe(cases.length);
    // The corpus and the template, and the 73 rejection documents.
    expect(documents).toHaveLength(15 + 73);
    // Every branch this feature decides is reached by some document or some case.
    const lines = [...documents, ...cases].flatMap((one) => one.errors ?? []);
    const reaches = (fragment: string): boolean => lines.some((line) => line.includes(fragment));
    for (const fragment of [
      'dtype selector is unknown',
      'dtype selector is not an enum',
      'outside the admissible set',
      'no location, while the document locates its weights',
      'already bound by',
      'is bound whole by',
      'overlap',
      'is not an axis of the slot',
      'slice inside a concat',
      'the offset does not resolve to a non-negative integer',
      'does not resolve in the physical name',
      'outside a stack over that axis',
      'unknown location form',
      'a weights_location_prefix on an instance that is not a',
      'a located template instance needs a weights_location_prefix',
      'has no parameter',
      'has no state port',
      'absent for these arguments',
      'declares a multiplicity that resolves to',
      'bound twice',
      'unbound parameter slot:',
      'unbound state port:',
      'is exclusive, it cannot be tied',
      'does not share with role',
      'incompatible shapes',
      'members with different payloads',
      'members under different derivation rules',
      'members indexed by different streams',
      'carried across fragments, but its stream',
      'writer(s) among',
      'reads across positions of the fragmented stream',
    ]) {
      expect(reaches(fragment), fragment).toBe(true);
    }
    // One document, and one only, reaches the place where the tools choose an instance's own
    // domain out of a Python set — a coin flip no implementation can be at parity with, and at
    // this stage it reaches the refusals themselves. A second one appearing is a finding.
    expect(
      [...documents, ...cases]
        .filter((one) => one.arbitrary_own === true)
        .map((one) => ('path' in one ? one.path : one.name)),
    ).toEqual(['tests/rejections/models/v5-fusion-without-join.json']);
    // One document, and one only, reaches the line that interpolates the sentinel: its address is
    // a coin flip no implementation can be at parity with. A second one appearing is a finding,
    // not a silent exemption.
    expect(
      [...documents, ...cases]
        .filter((one) => one.elided_sentinel === true)
        .map((one) => ('path' in one ? one.path : one.name)),
    ).toEqual(['tests/rejections/models/v3-streams-below-two.json']);
    // The advisory has no document in the repository at all; one edited case produces it.
    expect(documents.every((one) => (one.advisories ?? []).length === 0)).toBe(true);
    expect(cases.some((one) => (one.advisories ?? []).length > 0)).toBe(true);
    // A location of each of the four forms is evaluated somewhere.
    const locations = [...documents, ...cases]
      .flatMap((one) => one.tensors ?? [])
      .flatMap((one) => (one.location === undefined || one.location === null ? [] : [one.location]));
    for (const form of ['tensor', 'stack', 'concat', 'slice']) {
      expect(
        locations.some((one) => Object.prototype.hasOwnProperty.call(one, form)),
        form,
      ).toBe(true);
    }
  });

  it.skipIf(!generated)('answers every document as the tools answer it', () => {
    for (const one of recorded().documents) {
      let answer: Analysis | null = null;
      let raised: Raised | null = null;
      try {
        answer = analysedDocument(one);
      } catch (error) {
        raised = raisedAs(error);
      }
      expect(raised, one.path).toEqual(one.error);
      if (one.error !== null) continue;
      expect(
        answered(answer as Analysis, one.arbitrary_own === true),
        one.path,
      ).toEqual(expected(one));
    }
  }, 300_000);

  it.skipIf(!generated)('answers every edited case as the tools answer it', () => {
    for (const one of recorded().cases) {
      let answer: Analysis | null = null;
      let raised: Raised | null = null;
      try {
        answer = analysedCase(one);
      } catch (error) {
        raised = raisedAs(error);
      }
      expect(raised, one.name).toEqual(one.error);
      if (one.error !== null) continue;
      expect(
        answered(answer as Analysis, one.arbitrary_own === true),
        one.name,
      ).toEqual(expected(one));
    }
  }, 300_000);

  it.skipIf(!generated)('refuses nothing on the corpus, and counts what the tools count', () => {
    const corpus = recorded().documents.filter((one) => one.path.startsWith('data/models/'));
    expect(corpus).toHaveLength(15);
    for (const one of corpus) {
      expect(one.errors, one.path).toEqual([]);
      const answer = analysedDocument(one);
      expect(formatSemanticProblems([...answer.problems]), one.path).toEqual([]);
      expect(statsOf(answer.stats), one.path).toEqual(one.stats);
      // The seventeen counters `--validate` prints, in the order the tools write them: this
      // feature adds the last twelve to feature 1.6b's five.
      expect([...answer.stats.keys()], one.path).toEqual([
        'composite_primitives',
        'instances',
        'edges',
        'dag',
        'resolved_domains',
        'parameter_slots',
        'tensors',
        'shared',
        'located',
        'parameter_elements',
        'ops_per_element',
        'ops_per_cached_position',
        'ops_per_sequence',
        'ops_per_invocation',
        'precisions_checked',
        'state_slots',
        'state_identities',
      ]);
      // Every bound slot names a site of the graph, and the counters count what the maps hold.
      for (const slot of answer.bindings.slots.values()) {
        expect(answer.resolved.has(keyOf(slot.site)), one.path).toBe(true);
      }
      // The counters merge an expanded template's with the caller's (§4.6), so they equal the
      // maps only where nothing is expanded — and are never smaller.
      const slots = BigInt(answer.bindings.slots.size);
      const states = BigInt(answer.bindings.stateSlots.size);
      if (answer.subResults.size === 0) {
        expect(slots, one.path).toBe(answer.stats.get('parameter_slots'));
        expect(states, one.path).toBe(answer.stats.get('state_slots'));
      } else {
        expect(slots, one.path).toBeLessThan(answer.stats.get('parameter_slots') as bigint);
        expect(states, one.path).toBeLessThan(answer.stats.get('state_slots') as bigint);
      }
    }
  }, 300_000);

  it.skipIf(!generated)('reports the advisories `--lint` prints, and no others', () => {
    // `--lint`'s model findings are `analyse`'s advisories, printed as `W <file>: <line>`
    // (`lint.model_advisories`). The corpus produces none, which is what the oracle's `--lint`
    // output says in as many words — so the comparison is over the set, in both directions.
    const manifest = readOracleManifest();
    const corpus = recorded().documents.filter((one) => one.path.startsWith('data/models/'));
    const files = new Set(manifest.documents.map((one) => one.lint));
    const printed: string[] = [];
    for (const file of files) {
      printed.push(...readFileSync(join(oracleOut, file), 'utf8').split('\n'));
    }
    for (const one of corpus) {
      const answer = analysedDocument(one);
      const base = one.path.slice(one.path.lastIndexOf('/') + 1);
      for (const line of answer.advisories) {
        expect(printed, `${one.path}: ${line}`).toContain(`  W  ${base}: ${line}`);
      }
      expect(answer.advisories, one.path).toEqual(one.advisories ?? []);
    }
    // Nothing of the corpus is advised about, and `--lint` says so: the only findings its output
    // carries are `primitive_library`'s, which are feature 1.10's (uncalled primitives, axes and
    // roles no primitive cites) and not the validator's.
    expect(corpus.every((one) => (one.advisories ?? []).length === 0)).toBe(true);
    const findings = printed.filter((line) => line.trimStart().startsWith('W  '));
    expect(findings.every((line) => line.includes('W  primitive_library: '))).toBe(true);
  }, 300_000);

  it.skipIf(!generated)('reproduces the facts `tests/run_states.py` asserts', () => {
    const documents = recorded().documents;
    const of = (name: string): Analysis =>
      analysedDocument(documents.find((one) => one.name === name) as DocumentCase);

    // 1. "llama3-8b: 32 identities, each keyed (layer, session, branch)."
    const llama = of('llama3-8b');
    const keys = [...llama.bindings.instanceKeys.values()];
    expect(keys).toHaveLength(32);
    for (const key of keys) {
      expect(key).toEqual(['layer', 'instance.session', 'instance.branch']);
    }

    // 2. "gemma3n-kvshare: 20 identities from 30 slots; the two shared identities carry no layer
    //     index — sharing is several members under one identity, nothing else."
    const gemma = of('gemma3n-kvshare');
    expect(gemma.stats.get('state_slots')).toBe(30n);
    expect(gemma.stats.get('state_identities')).toBe(20n);
    const shared = [...gemma.bindings.instanceKeys]
      .filter(([, key]) => key.length === 2 && key[0] === 'instance.session')
      .map(([rule]) => rule)
      .sort();
    expect(shared).toEqual(['shared.full.kv', 'shared.sliding.kv']);
    // "the shared identities are written by layers 18 and 19; the readers own no key/value
    // projections (V20)"
    const writers = new Map(
      gemma.bindings.stateInstances
        .filter((one) => one.identity.startsWith('shared.'))
        .map((one) => [
          one.identity,
          one.writer === null ? null : `${whereOfSite(one.writer.site)}.${label(one.writer.name)}`,
        ]),
    );
    expect(writers.get('shared.sliding.kv')).toBe('decoder/attn[layer=18].kv');
    expect(writers.get('shared.full.kv')).toBe('decoder/attn_full[layer=19].kv');

    // 3. "voxtral: the encoder attention state, the front end's two histories and the decoder
    //     attention state are carried across the fragments of `audio` — the decoder's, at kind
    //     token, since the token input joins that stream (V18)."
    const voxtral = of('voxtral-realtime');
    expect(
      Object.fromEntries(
        [...voxtral.bindings.carried].map(([rule, mine]) => [
          rule,
          mine === null ? null : [label(mine[0]), label(mine[1])],
        ]),
      ),
    ).toEqual({
      'encoder.attn.kv': ['position', 'audio'],
      'conv_frontend.conv1_history': ['position', 'audio'],
      'conv_frontend.conv2_history': ['position', 'audio'],
      'decoder.attn.kv': ['token', 'audio'],
    });
    // "nothing else is carried, and no advisory: every self-indexed state on the fragmented
    // stream is carried"
    expect(voxtral.advisories).toEqual([]);
  }, 300_000);

  it.skipIf(!generated)('decides the V7, V14, V15, V16, V17, V18 and V20 cases', () => {
    const documents = recorded().documents;
    const suite = JSON.parse(
      readFileSync(join(oracleOut, 'rejections', 'models.json'), 'utf8'),
    ) as { cases: { document: string; expect: string; match: string }[] };
    // What this feature decides, case by case. V9 has no fixture in the suite at all — one of the
    // two rules of §6 the corpus and `tests/rejections` never exercise (V6 is feature 1.6b's).
    const mine = [
      'v14-inadmissible-dtype.json',
      'v15-multiplicity-mismatch.json',
      'v15-tying-exclusive.json',
      'v16-carried-without-fragmentation.json',
      'v17-instances-one-prefix.json',
      'v17-location-duplicate.json',
      'v17-location-partial.json',
      'v17-prefix-on-primitive.json',
      'v17-prefix-on-unlocated.json',
      'v17-slice-inside-concat.json',
      'v17-slice-overlap.json',
      'v17-stack-not-an-axis.json',
      'v17-template-located.json',
      'v18-fragmented-stateless-frontend.json',
      'v20-no-writer.json',
      'v20-two-writers.json',
      'v7-slot-bound-twice.json',
      'v7-unbound-state.json',
    ];
    for (const name of mine) {
      const rejection = suite.cases.find((one) => one.document === `models/${name}`);
      expect(rejection, name).toBeDefined();
      const path = join('tests', 'rejections', (rejection as { document: string }).document);
      const one = documents.find((each) => each.path === path);
      expect(one, path).toBeDefined();
      const answer = analysedDocument(one as DocumentCase);
      const printed = formatSemanticProblems([...answer.problems]);
      expect(printed, path).toEqual((one as DocumentCase).errors);
      // The runner's own reading (`tests/run_rejections.py`): a line with the expected code
      // carrying the expected words.
      const expectation = rejection as { expect: string; match: string };
      const decided = printed.filter((line) => line.startsWith(`[${expectation.expect}] `));
      expect(decided.length, path).toBeGreaterThan(0);
      expect(
        decided.some((line) => line.includes(expectation.match)),
        `${path}: ${expectation.match}`,
      ).toBe(true);
    }
    // Every model case of the suite carrying one of this feature's codes is one of these: the
    // codes are the partition, and it is stated here so that a case moving shows up.
    const codes = ['V9', 'V14', 'V15', 'V16', 'V17', 'V18', 'V20'];
    expect(
      suite.cases
        .filter((one) => codes.includes(one.expect))
        .map((one) => one.document.split('/').pop() as string)
        .sort(),
    ).toEqual(mine.filter((name) => !name.startsWith('v7-')).sort());
  }, 300_000);

  it.skipIf(!generated)('passes every model case of the rejection suite, 73 of 73', () => {
    // The feature's own criterion, read the way `tests/run_rejections.py` reads it: the grammar
    // first (feature 1.1's `structuralText`), then the meaning, and a line with the expected code
    // carrying the expected words. Every earlier feature's cases are in it, so this is the whole
    // model validator answering.
    const documents = recorded().documents;
    const suite = JSON.parse(
      readFileSync(join(oracleOut, 'rejections', 'models.json'), 'utf8'),
    ) as { cases: { document: string; expect: string; match: string }[] };
    expect(suite.cases).toHaveLength(73);
    for (const one of suite.cases) {
      const path = join('tests', 'rejections', one.document);
      const text = readFileSync(join(repositoryRoot, path), 'utf8');
      let lines = formatProblemLines(text);
      if (one.expect !== 'schema' && lines.length === 0) {
        const recordedCase = documents.find((each) => each.path === path) as DocumentCase;
        const answer = analysedDocument(recordedCase);
        lines = formatSemanticProblems([...answer.problems]).filter((line) =>
          line.startsWith(`[${one.expect}] `),
        );
      }
      expect(
        lines.some((line) => line.includes(one.match)),
        `${one.document} (${one.expect}): ${one.match}`,
      ).toBe(true);
    }
  }, 300_000);
});

/** The schema stage's lines, as `validate.structural` prints them (feature 1.1). */
function formatProblemLines(text: string): string[] {
  return schemas.structuralText(text, 'model').map((problem) => problem.message);
}
