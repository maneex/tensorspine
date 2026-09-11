import { describe, expect, it } from 'vitest';

import { d4, rootSite, type PyRecord, type PyValue } from '../../src/index.js';
import { PyKeyError, PyValueError } from '../../src/expr/errors.js';
import { library } from '../describe/source.js';
import { record, stateInstance, syntheticGraph, type SyntheticExtra } from './source.js';

// D4 (feature 1.8b), on the branches no corpus document reaches.
//
// The fifteen corpus documents and their 1 031 state identity instances are the parity suite's:
// every evolution, every access geometry, every sharing granularity, both branches of the
// carrying, a source-indexed state on a fragmented stream and one on a stream that is not. What
// they do not carry is here — an identity no member of which resolved, a port no rule applies to
// (V9 refuses it, the tools guard it all the same), a stream no domain answers, a payload whose
// extent does not resolve, a sub-byte payload, a flattened payload axis, a span of zero, the two
// figures R11 refuses, a writer no member writes, and the totals' arithmetic. Each is built as
// `_expand`'s own answer, since D4 is a function of that and of the library and of nothing else.

/** One payload component: 8 × 4 elements of `state.kv`, whose default dtype is `bf16`. */
const KV = `{"kv": {"role": "state.kv", "shape": {"axes": [
  {"name": "h", "axis": "attention.kv_heads", "nature": "feature", "extent": {"literal": 8}},
  {"name": "d", "axis": "attention.head_dim", "nature": "feature", "extent": {"literal": 4}}]}}}`;

/** An `append` rule indexed by the instance's own stream, the shape most of the base declares. */
const APPEND = `{"when": {"boolean": true}, "evolution": "append", "access": "logical_position",
  "sharing": "by_position", "indexed_by": {"self": true}}`;

/** A state port: the rules a case declares, and whatever else it needs. */
function statePort(
  rules: string,
  extra: { payload?: string; keyAxes?: string; operations?: string; carried?: string } = {},
): string {
  const operations = extra.operations ?? '{"r": {"effect": "read"}, "a": {"effect": "append"}}';
  const carried = extra.carried === undefined ? '' : `, "carried_across": {"when": ${extra.carried}}`;
  return `{"present_when": {"boolean": true}, "payload": ${extra.payload ?? KV},
    "key_axes": ${extra.keyAxes ?? '["instance.session"]'}, "operations": ${operations},
    "rules": ${rules}${carried}}`;
}

/** A primitive declaring one state port called `kv`. */
function definitionOf(port: string): PyValue {
  return record(`{"state_ports": {"kv": ${port}}}`);
}

/** D4 over one declaration, one port, one identity instance. */
function inventory(
  port: string,
  extra: SyntheticExtra = {},
  args: PyRecord = {},
  instance: Parameters<typeof stateInstance>[2] = {},
): PyRecord {
  const graph = syntheticGraph(
    [{ name: 'n', primitive: 'attention.dense', definition: definitionOf(port), args }],
    [],
    { ...extra, states: [stateInstance('s', [['n', 'kv']], instance)] },
  );
  return d4(graph, library);
}

/** The one row of a one-state graph. */
function only(answer: PyRecord): PyRecord {
  const states = answer['states'] as readonly PyValue[];
  expect(states).toHaveLength(1);
  return states[0] as PyRecord;
}

describe('one entry per state identity instance', () => {
  it('writes the port’s facts, its rule, its payload, its figures and its totals', () => {
    const answer = inventory(statePort(`[${APPEND}]`), { own: { n: ['token', 'tokens'] } });
    expect(only(answer)).toEqual({
      identity: 's',
      members: ['n.kv'],
      writer: 'n.kv',
      primitive: 'attention.dense',
      state: 'kv',
      evolution: 'append',
      access: 'logical_position',
      sharing: 'by_position',
      stream: { kind: 'token', stream: 'tokens' },
      indexed_by_source: false,
      indexed_by_port: null,
      instance_key: ['instance.session'],
      carried_across_fragments: false,
      span: null,
      stride: null,
      payload: [
        {
          component: 'kv',
          role: 'state.kv',
          dtype: 'bf16',
          shape: [
            { axis: 'attention.kv_heads', extent: 8n },
            { axis: 'attention.head_dim', extent: 4n },
          ],
          elements: 32n,
          bytes: 64n,
        },
      ],
      bytes_per_cached_position: 64n,
      bytes_bounded: null,
      operations: ['append', 'read'],
      visits: { write: 'once per new element of its stream', read: 'once per element produced' },
    });
    expect(answer['totals']).toEqual({
      identities: 1n,
      by_evolution: { append: 1n },
      append_bytes_per_cached_position: 64n,
      bounded_bytes: 0n,
      fixed_bytes: 0n,
      carried: [],
    });
  });

  it('skips an identity instance no member of which resolved', () => {
    const graph = syntheticGraph([], [], { states: [stateInstance('s', [])] });
    const answer = d4(graph, library);
    expect(answer['states']).toEqual([]);
    expect(answer['totals']).toEqual({
      identities: 0n,
      by_evolution: {},
      append_bytes_per_cached_position: 0n,
      bounded_bytes: 0n,
      fixed_bytes: 0n,
      carried: [],
    });
  });

  it('reads every fact from the first member, and lists the members it was given', () => {
    // "Of the first member; V15 makes the others compatible" is D3's line; V9 is D4's, and it
    // makes the members agree on the key axes, the payload, the rule and the stream — which is
    // the whole of what an entry says about a port.
    const port = statePort(`[${APPEND}]`);
    const other = statePort(
      `[{"when": {"boolean": true}, "evolution": "fixed", "access": "aggregate",
         "sharing": "at_fork_point", "indexed_by": {"self": true}}]`,
    );
    const graph = syntheticGraph(
      [
        { name: 'a', primitive: 'attention.dense', definition: definitionOf(port) },
        { name: 'b', primitive: 'attention.latent', definition: definitionOf(other) },
      ],
      [],
      { states: [stateInstance('shared.kv', [['a', 'kv'], ['b', 'kv']])] },
    );
    const row = only(d4(graph, library));
    expect(row['members']).toEqual(['a.kv', 'b.kv']);
    expect(row['primitive']).toBe('attention.dense');
    expect(row['evolution']).toBe('append');
    expect(row['writer']).toBe('a.kv');
  });

  it('names the writer the identity settled, whichever member it is', () => {
    const port = statePort(`[${APPEND}]`);
    const graph = syntheticGraph(
      [
        { name: 'a', definition: definitionOf(port) },
        { name: 'b', definition: definitionOf(port) },
      ],
      [],
      {
        states: [
          stateInstance('shared.kv', [['a', 'kv'], ['b', 'kv']], {
            writer: { site: rootSite('b'), name: 'kv' },
          }),
        ],
      },
    );
    expect(only(d4(graph, library))['writer']).toBe('b.kv');
  });

  it('writes a null writer for an identity nobody writes', () => {
    // V20 refuses such a document ("exactly one member writes"), and the derived schema's
    // `value_reference` is a string with no null beside it — so the tools' guard writes a value
    // their own schema would refuse. Unreachable, reproduced, and stated.
    const row = only(inventory(statePort(`[${APPEND}]`), {}, {}, { writer: null }));
    expect(row['writer']).toBeNull();
  });
});

describe('the rule that applies (§4.3)', () => {
  it('takes the first rule whose condition holds, not the last', () => {
    const rules = `[
      {"when": {"present": "window"}, "evolution": "window", "access": "ring",
       "sharing": "within_span", "indexed_by": {"self": true}, "span": {"argument": "window"}},
      ${APPEND}]`;
    const windowed = only(inventory(statePort(rules), {}, record('{"window": 4}')));
    expect(windowed['evolution']).toBe('window');
    expect(windowed['span']).toBe(4n);
    expect(only(inventory(statePort(rules)))['evolution']).toBe('append');
  });

  it('leaves every rule-derived fact blank when no rule applies', () => {
    // V9 refuses a present port no rule matches, so this cannot come out of a valid document; the
    // tools guard every field with `if rule` and the port reproduces the guard.
    const row = only(
      inventory(
        statePort(`[{"when": {"present": "window"}, "evolution": "window", "access": "ring",
          "sharing": "within_span", "indexed_by": {"self": true}, "span": {"literal": 2}}]`),
        { own: { n: ['token', 'tokens'] } },
      ),
    );
    expect(row['evolution']).toBeNull();
    expect(row['access']).toBeNull();
    expect(row['sharing']).toBeNull();
    expect(row['stream']).toBeNull();
    expect(row['indexed_by_source']).toBe(false);
    expect(row['indexed_by_port']).toBeNull();
    expect(row['span']).toBeNull();
    expect(row['stride']).toBeNull();
    expect(row['bytes_bounded']).toBeNull();
    // The payload is the port's, not the rule's, so it is written all the same.
    expect(row['bytes_per_cached_position']).toBe(64n);
    expect(row['visits']).toEqual({
      write: 'once per new element of its stream',
      read: 'once per element produced',
    });
  });

  it('counts an identity with no rule under a `null` evolution, as json.dumps writes the key', () => {
    const answer = inventory(
      statePort('[{"when": {"present": "window"}, "evolution": "window", "access": "ring", ' +
        '"sharing": "within_span", "indexed_by": {"self": true}, "span": {"literal": 2}}]'),
    );
    expect((answer['totals'] as PyRecord)['by_evolution']).toEqual({ null: 1n });
    // It feeds none of the three byte totals: each names one evolution and no other.
    expect((answer['totals'] as PyRecord)['append_bytes_per_cached_position']).toBe(0n);
    expect((answer['totals'] as PyRecord)['bounded_bytes']).toBe(0n);
    expect((answer['totals'] as PyRecord)['fixed_bytes']).toBe(0n);
  });

  it('raises on a rule that declares no indexing source, as the tools raise', () => {
    const rules = `[{"when": {"boolean": true}, "evolution": "append",
      "access": "logical_position", "sharing": "by_position"}]`;
    expect(() => inventory(statePort(rules))).toThrowError(PyKeyError);
    expect(() => inventory(statePort(rules))).toThrowError("'indexed_by'");
  });
});

describe('the stream the state grows along (§5.3)', () => {
  const bySource = `[{"when": {"boolean": true}, "evolution": "append", "access": "logical_position",
    "sharing": "by_source", "indexed_by": {"port": "source"}}]`;

  it('takes the instance’s own domain for a self-indexed state', () => {
    const row = only(inventory(statePort(`[${APPEND}]`), { own: { n: ['sequence', 'audio'] } }));
    expect(row['stream']).toEqual({ kind: 'sequence', stream: 'audio' });
    expect(row['indexed_by_source']).toBe(false);
  });

  it('takes the port’s domain for a source-indexed state, and names the port', () => {
    const row = only(
      inventory(statePort(bySource), {
        own: { n: ['token', 'tokens'] },
        domains: { 'n.source': ['sequence', 'audio'] },
      }),
    );
    expect(row['stream']).toEqual({ kind: 'sequence', stream: 'audio' });
    expect(row['indexed_by_source']).toBe(true);
    expect(row['indexed_by_port']).toBe('source');
    // "A state indexed by a source stream is written once per source element and frozen when the
    // source is complete" (§7) — the source-indexed wording, whatever the evolution.
    expect(row['visits']).toEqual({
      write: 'once per element of the source stream, until the source is complete',
      read: 'once per element produced',
    });
  });

  it('gives a source-indexed state the source’s visits whatever its evolution', () => {
    // `_visits` reads `source_indexed` before it reads the evolution, so a `fixed` state indexed
    // by a source stream is written once per source element and not once per element — "frozen
    // when the source is complete" (§7). No corpus document carries the pair.
    const fixedBySource = `[{"when": {"boolean": true}, "evolution": "fixed",
      "access": "aggregate", "sharing": "by_source", "indexed_by": {"port": "source"}}]`;
    const row = only(
      inventory(statePort(fixedBySource), { domains: { 'n.source': ['sequence', 'audio'] } }),
    );
    expect(row['evolution']).toBe('fixed');
    expect(row['visits']).toEqual({
      write: 'once per element of the source stream, until the source is complete',
      read: 'once per element produced',
    });
  });

  it('writes no stream where the graph answers none', () => {
    expect(only(inventory(statePort(`[${APPEND}]`)))['stream']).toBeNull();
    expect(only(inventory(statePort(bySource), { own: { n: ['token', 'tokens'] } }))['stream'])
      .toBeNull();
  });
});

describe('carrying across the fragments of a stream (§5.3, V16)', () => {
  const fragmented = record(
    '{"interfaces": {"inputs": {"audio": {"fragmented": true}, "tokens": {}}}}',
  );
  const bySource = `[{"when": {"boolean": true}, "evolution": "append", "access": "logical_position",
    "sharing": "by_source", "indexed_by": {"port": "source"}}]`;

  it('carries a state whose primitive’s condition holds, whatever its stream', () => {
    // D4 states the *primitive's* answer: V16 is what requires the stream to be fragmented, and a
    // document that got here has already passed it.
    const row = only(
      inventory(statePort(`[${APPEND}]`, { carried: '{"boolean": true}' }), {
        model: fragmented,
        own: { n: ['sequence', 'audio'] },
      }),
    );
    expect(row['carried_across_fragments']).toBe(true);
  });

  it('does not carry one whose condition does not hold', () => {
    const row = only(
      inventory(statePort(`[${APPEND}]`, { carried: '{"present": "chunk"}' }), {
        model: fragmented,
        own: { n: ['sequence', 'audio'] },
      }),
    );
    expect(row['carried_across_fragments']).toBe(false);
  });

  it('carries a source-indexed state on a fragmented stream by definition', () => {
    const row = only(
      inventory(statePort(bySource), {
        model: fragmented,
        domains: { 'n.source': ['sequence', 'audio'] },
      }),
    );
    expect(row['carried_across_fragments']).toBe(true);
  });

  it('does not carry a source-indexed state whose stream is not fragmented', () => {
    const row = only(
      inventory(statePort(bySource), {
        model: fragmented,
        domains: { 'n.source': ['token', 'tokens'] },
      }),
    );
    expect(row['carried_across_fragments']).toBe(false);
  });

  it('reads the stream a fragmented input declares, not the input’s name', () => {
    const renamed = record(
      '{"interfaces": {"inputs": {"waveform": {"fragmented": true, "stream": "audio"}}}}',
    );
    const row = only(
      inventory(statePort(bySource), {
        model: renamed,
        domains: { 'n.source': ['sequence', 'audio'] },
      }),
    );
    expect(row['carried_across_fragments']).toBe(true);
  });

  it('lists every carried identity in the totals, in the order the states were emitted', () => {
    const port = statePort(`[${APPEND}]`, { carried: '{"boolean": true}' });
    const plain = statePort(`[${APPEND}]`);
    const graph = syntheticGraph(
      [
        { name: 'a', definition: definitionOf(port) },
        { name: 'b', definition: definitionOf(plain) },
        { name: 'c', definition: definitionOf(port) },
      ],
      [],
      {
        model: fragmented,
        states: [
          stateInstance('one', [['a', 'kv']]),
          stateInstance('two', [['b', 'kv']]),
          stateInstance('three', [['c', 'kv']]),
        ],
      },
    );
    expect((d4(graph, library)['totals'] as PyRecord)['carried']).toEqual(['one', 'three']);
  });
});

describe('the payload (§4.3, O5.1)', () => {
  it('writes one row per component, in declaration order, each with its own role and dtype', () => {
    const payload = `{"keys": {"role": "state.kv", "shape": {"axes": [
        {"name": "d", "axis": "attention.head_dim", "nature": "feature", "extent": {"literal": 4}}]}},
      "history": {"role": "state.recurrent", "shape": {"axes": [
        {"name": "d", "axis": "attention.head_dim", "nature": "feature", "extent": {"literal": 2}}]}}}`;
    const row = only(inventory(statePort(`[${APPEND}]`, { payload })));
    expect((row['payload'] as readonly PyValue[]).map((one) => (one as PyRecord)['component']))
      .toEqual(['keys', 'history']);
    // `state.kv` defaults to `bf16` (2 bytes), `state.recurrent` to `f32` (4): 8 + 8.
    expect((row['payload'] as readonly PyRecord[]).map((one) => one['dtype']))
      .toEqual(['bf16', 'f32']);
    expect(row['bytes_per_cached_position']).toBe(16n);
  });

  it('takes the dtype the identity’s binding selects, for every component', () => {
    const row = only(inventory(statePort(`[${APPEND}]`), {}, {}, { dtype: 'f8e4m3' }));
    expect((row['payload'] as readonly PyRecord[])[0]?.['dtype']).toBe('f8e4m3');
    expect(row['bytes_per_cached_position']).toBe(32n);
  });

  it('takes a sub-byte dtype’s size as a real, and carries the real into the totals', () => {
    const answer = inventory(statePort(`[${APPEND}]`), {}, {}, { dtype: 'fp4' });
    expect((only(answer)['payload'] as readonly PyRecord[])[0]?.['bytes']).toBe(16);
    expect(only(answer)['bytes_per_cached_position']).toBe(16);
    expect((answer['totals'] as PyRecord)['append_bytes_per_cached_position']).toBe(16);
  });

  it('leaves the count and the size blank where an extent does not resolve', () => {
    const payload = `{"kv": {"role": "state.kv", "shape": {"axes": [
      {"name": "d", "axis": "attention.head_dim", "nature": "feature",
       "extent": {"argument": "absent"}}]}}}`;
    const row = only(inventory(statePort(`[${APPEND}]`, { payload })));
    expect((row['payload'] as readonly PyRecord[])[0]).toMatchObject({
      shape: [{ axis: 'attention.head_dim', extent: null }],
      elements: null,
      bytes: null,
    });
    // `sum(c['bytes'] or 0 …)`: a blank contributes the integer zero, never a blank total.
    expect(row['bytes_per_cached_position']).toBe(0n);
  });

  it('sums a real zero as the integer zero, which is what `or 0` reads', () => {
    // A zero is falsy in Python, so `c['bytes'] or 0` contributes the *integer* zero whether the
    // figure was `0` or `0.0` — and a payload of zero bytes therefore leaves the total an integer
    // rather than promoting it to a real, which the derived document's bytes record.
    const payload = `{"kv": {"role": "state.kv", "shape": {"axes": [
      {"name": "d", "axis": "attention.head_dim", "nature": "feature", "extent": {"literal": 0}}]}}}`;
    const row = only(inventory(statePort(`[${APPEND}]`, { payload }), {}, {}, { dtype: 'fp4' }));
    expect((row['payload'] as readonly PyRecord[])[0]?.['bytes']).toBe(0);
    expect(row['bytes_per_cached_position']).toBe(0n);
  });

  it('writes a flattened axis with the factors the primitive declared (O5.10)', () => {
    const payload = `{"kv": {"role": "state.kv", "shape": {"axes": [
      {"name": "f", "axis": "model.width", "nature": "feature",
       "extent": {"op": "multiply", "args": [{"literal": 8}, {"literal": 4}]},
       "factors": [
         {"name": "h", "axis": "attention.kv_heads", "nature": "feature", "extent": {"literal": 8}},
         {"name": "d", "axis": "attention.head_dim", "nature": "feature", "extent": {"literal": 4}}]}]}}}`;
    const row = only(inventory(statePort(`[${APPEND}]`, { payload })));
    expect((row['payload'] as readonly PyRecord[])[0]?.['shape']).toEqual([
      {
        axis: 'model.width',
        extent: 32n,
        factors: [
          { axis: 'attention.kv_heads', extent: 8n },
          { axis: 'attention.head_dim', extent: 4n },
        ],
      },
    ]);
  });
});

describe('the modulators and the bounded size (O5.8)', () => {
  function windowed(span: string, extra = ''): PyRecord {
    return only(
      inventory(
        statePort(`[{"when": {"boolean": true}, "evolution": "window", "access": "ring",
          "sharing": "within_span", "indexed_by": {"self": true}, "span": ${span}${extra}}]`),
      ),
    );
  }

  it('bounds a window at span × bytes per cached position', () => {
    const row = windowed('{"literal": 4}', ', "stride": {"literal": 2}');
    expect(row['span']).toBe(4n);
    expect(row['stride']).toBe(2n);
    expect(row['bytes_bounded']).toBe(256n);
  });

  it('bounds nothing where the span is zero, as it bounds nothing where it does not resolve', () => {
    // `(per_position * span) if span else None` reads the figure's *truth*: a span of zero is
    // falsy in Python, and the tools write no bound rather than a bound of zero.
    expect(windowed('{"literal": 0}')['bytes_bounded']).toBeNull();
    expect(windowed('{"argument": "absent"}')['bytes_bounded']).toBeNull();
    expect(windowed('{"argument": "absent"}')['span']).toBeNull();
  });

  it('sums a window that bounds nothing as the integer zero', () => {
    const answer = inventory(
      statePort(`[{"when": {"boolean": true}, "evolution": "window", "access": "ring",
        "sharing": "within_span", "indexed_by": {"self": true}, "span": {"literal": 0}}]`),
    );
    expect((answer['totals'] as PyRecord)['bounded_bytes']).toBe(0n);
  });
});

describe('the figures R11 refuses', () => {
  it('refuses a negative element count by name', () => {
    const payload = `{"kv": {"role": "state.kv", "shape": {"axes": [
      {"name": "d", "axis": "attention.head_dim", "nature": "feature", "extent": {"literal": -4}}]}}}`;
    const run = () => inventory(statePort(`[${APPEND}]`, { payload }));
    expect(run).toThrowError(PyValueError);
    expect(run).toThrowError(
      "s: derived state element count is -4 — negative or non-finite, which the validator's " +
        'domains should have refused (admitted upstream, a domain is missing)',
    );
  });

  it('refuses a non-finite byte size by name', () => {
    // The count is a real here (a real extent), so the width promotes it past the double range
    // where an integer count would have raised `OverflowError` at `elements` first.
    const payload = `{"kv": {"role": "state.kv", "shape": {"axes": [
      {"name": "d", "axis": "attention.head_dim", "nature": "feature", "extent": {"literal": 1e308}}]}}}`;
    const run = () => inventory(statePort(`[${APPEND}]`, { payload }));
    expect(run).toThrowError(PyValueError);
    expect(run).toThrowError('s: derived state byte size is inf — negative or non-finite');
  });

  it('refuses a negative span by name', () => {
    const run = () =>
      inventory(
        statePort(`[{"when": {"boolean": true}, "evolution": "window", "access": "ring",
          "sharing": "within_span", "indexed_by": {"self": true}, "span": {"literal": -2}}]`),
      );
    expect(run).toThrowError(PyValueError);
    expect(run).toThrowError('s: derived state span is -2 — negative or non-finite');
  });
});

describe('the instance key, the operations and the visits', () => {
  it('is the identity’s indices followed by the port’s key axes (O5.5)', () => {
    const row = only(
      inventory(
        statePort(`[${APPEND}]`, { keyAxes: '["instance.session", "instance.branch"]' }),
        {},
        {},
        { indices: ['layer'] },
      ),
    );
    expect(row['instance_key']).toEqual(['layer', 'instance.session', 'instance.branch']);
  });

  it('lists each effect the port admits once, sorted (O5.4)', () => {
    const operations = `{"evict": {"effect": "evict"}, "grow": {"effect": "append"},
      "lookup": {"effect": "read"}, "scan": {"effect": "read"}}`;
    expect(only(inventory(statePort(`[${APPEND}]`, { operations })))['operations'])
      .toEqual(['append', 'evict', 'read']);
  });

  it('gives a fixed state the per-element visits of §7', () => {
    const row = only(
      inventory(
        statePort(`[{"when": {"boolean": true}, "evolution": "fixed", "access": "aggregate",
          "sharing": "at_fork_point", "indexed_by": {"self": true}}]`),
      ),
    );
    expect(row['visits']).toEqual({ write: 'once per element', read: 'once per element' });
  });
});

describe('the totals', () => {
  it('counts each evolution in the order it was first seen, and feeds one total each', () => {
    const ports = {
      append: statePort(`[${APPEND}]`),
      fixed: statePort(`[{"when": {"boolean": true}, "evolution": "fixed", "access": "aggregate",
        "sharing": "at_fork_point", "indexed_by": {"self": true}}]`),
      window: statePort(`[{"when": {"boolean": true}, "evolution": "window", "access": "ring",
        "sharing": "within_span", "indexed_by": {"self": true}, "span": {"literal": 3}}]`),
    };
    const graph = syntheticGraph(
      [
        { name: 'w', definition: definitionOf(ports.window) },
        { name: 'a', definition: definitionOf(ports.append) },
        { name: 'f', definition: definitionOf(ports.fixed) },
        { name: 'a2', definition: definitionOf(ports.append) },
      ],
      [],
      {
        states: [
          stateInstance('one', [['w', 'kv']]),
          stateInstance('two', [['a', 'kv']]),
          stateInstance('three', [['f', 'kv']]),
          stateInstance('four', [['a2', 'kv']], { dtype: 'fp4' }),
        ],
      },
    );
    const totals = d4(graph, library)['totals'] as PyRecord;
    // A `Counter`'s keys keep their first-seen order, which is the states' order and is part of
    // the derived document's bytes.
    expect(Object.keys(totals['by_evolution'] as PyRecord)).toEqual(['window', 'append', 'fixed']);
    expect(totals['by_evolution']).toEqual({ window: 1n, append: 2n, fixed: 1n });
    // 64 + 16.0: the sum is Python's, term by term, and the real arrives with the second term.
    expect(totals['append_bytes_per_cached_position']).toBe(80);
    expect(totals['bounded_bytes']).toBe(192n);
    expect(totals['fixed_bytes']).toBe(64n);
    expect(totals['identities']).toBe(4n);
  });
});
