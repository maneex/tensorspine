import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  basesOf,
  derive,
  identityAt,
  identityDerived,
  identityReadings,
  inputValue,
  loadLibrary,
  outputValue,
  streamRows,
  toPython,
  valueOf,
  valueRows,
  type PyValue,
} from '../../src/index.js';
import { nodeSource } from '../library/source.js';
import { repositoryRoot } from '../json/repository.js';
import { repositorySchemas } from '../schema/repository.js';
import { corpus } from './source.js';

/**
 * What the sheets of plan §4.11 ask the core for: the identity a binding rule declares, the rows
 * the products carry for it, and the D2 value an edge, an input or an output shows.
 *
 * The reading under test is the join between a **place of the document** — which is what a
 * selection is (D1) — and the **name the products use**, and it is here rather than in the
 * interface because §5.2 rule 7 decides it: a scoped rule that declares no `tensor` has its
 * identity named `<composition>.<rule>` by the hoist, and the interface may not know that.
 */

const source = nodeSource(repositoryRoot);

/** One corpus document, derived, with the tree the readings are taken from. */
function open(name: string): { tree: PyValue; derived: PyValue } {
  const json = parse(corpus(name));
  const { bases, problem } = basesOf(`data/models/${name}.json`, toPython(json));
  if (problem !== null) throw new Error(`${name}: ${problem.message}`);
  const library = loadLibrary(bases, { schemas: repositorySchemas(), source });
  return {
    tree: toPython(json),
    derived: derive(json, { schemas: repositorySchemas(), library }),
  };
}

describe('the identities a document declares', () => {
  const { tree } = open('llama3-8b');
  const readings = identityReadings(tree);

  it('reads llama3-8b’s thirteen: three at the top level, nine scoped, one state', () => {
    expect(readings).toHaveLength(13);
    expect(readings.filter((one) => one.state)).toHaveLength(1);
    expect(readings.filter((one) => one.kind === 'constants')).toHaveLength(0);
  });

  it('keys a top-level rule by the place the author wrote, with its declared symbol', () => {
    const embed = identityAt(readings, '/bindings/parameters/embed.weight');
    expect(embed?.rule).toBe('embed.weight');
    expect(embed?.identity).toBe('embed.weight');
    expect(embed?.indices).toEqual([]);
    expect(embed?.state).toBe(false);
  });

  it('keys a scoped rule by its place inside the composition, and names it as §5.2 rule 7 does', () => {
    // The rule declares no `tensor` at all: the hoist names the identity `<composition>.<rule>`
    // and indexes it by the composition's own indices, which is what D3 then writes.
    const q = identityAt(readings, '/compositions/decoder/bindings/parameters/attn.q');
    expect(q?.rule).toBe('decoder.attn.q');
    expect(q?.identity).toBe('decoder.attn.q');
    expect(q?.indices).toEqual(['layer']);
    const kv = identityAt(readings, '/compositions/decoder/bindings/states/attn.kv');
    expect(kv?.state).toBe(true);
    expect(kv?.identity).toBe('decoder.attn.kv');
  });

  it('answers nothing for a place that declares no identity — a value rule is a label', () => {
    expect(identityAt(readings, '/bindings/values/decoder.entry')).toBeUndefined();
    expect(identityAt(readings, '/quantities/d')).toBeUndefined();
  });

  it('reads the symbol a rule declares, and the corpus always writes its own name', () => {
    // Measured over the fifteen documents: **no** rule of the corpus declares a `tensor` or an
    // `identity` whose name differs from the rule's own normalised name. The grammar admits the
    // difference — the symbol is a name of its own — so the two are read apart here and the
    // products are matched on the *identity*; the corpus simply never exercises it.
    const qwen = identityReadings(open('qwen3.5-4b-text').tree);
    const tie = identityAt(qwen, '/bindings/parameters/embed.weight');
    expect(tie?.identity).toBe('embed.weight');
    expect(qwen.filter((one) => one.identity !== one.rule)).toEqual([]);
  });
});

describe('what the products say about one identity', () => {
  const { tree, derived } = open('llama3-8b');
  const readings = identityReadings(tree);

  it('gives a scoped parameter identity its thirty-two instances and their sum', () => {
    const q = identityAt(readings, '/compositions/decoder/bindings/parameters/attn.q');
    const rows = identityDerived(derived, q?.identity ?? '', false);
    expect(rows.instances).toBe(32);
    expect(rows.tensors[0]?.identity).toBe('decoder.attn.q[layer=0]');
    expect(rows.tensors[0]?.slot).toBe('q');
    expect(rows.tensors[0]?.shape).toBe('[attention.heads=4096, model.width=4096]');
    // 32 MiB an instance — D3's own number, added thirty-two times and nothing else.
    expect(rows.tensors[0]?.bytes).toBe(33554432n);
    expect(rows.bytes).toBe(32n * 33554432n);
    expect(rows.states).toEqual([]);
  });

  it('gives a top-level identity its one instance', () => {
    const rows = identityDerived(derived, 'embed.weight', false);
    expect(rows.instances).toBe(1);
    expect(rows.bytes).toBe(1050673152n);
    expect(rows.tensors[0]?.located).toBe(true);
  });

  it('gives a state identity its D4 rows and their bytes a cached position', () => {
    const kv = identityAt(readings, '/compositions/decoder/bindings/states/attn.kv');
    const rows = identityDerived(derived, kv?.identity ?? '', true);
    expect(rows.instances).toBe(32);
    expect(rows.tensors).toEqual([]);
    expect(rows.states[0]?.identity).toBe('decoder.attn.kv[layer=0]');
    expect(rows.states[0]?.port).toBe('kv');
    expect(rows.states[0]?.evolution).toBe('append');
    expect(rows.bytesPerCachedPosition).toBe(32n * 4096n);
  });

  it('is empty for an identity the products do not carry', () => {
    expect(identityDerived(derived, 'decoder.attn.q2', false).instances).toBe(0);
    // The match is the instance's own name or that name with its indices, never a prefix.
    expect(identityDerived(derived, 'decoder.attn', false).instances).toBe(0);
  });
});

describe('what D2 says about a value, an input and an output', () => {
  const { derived } = open('llama3-8b');
  const values = valueRows(derived);

  it('finds the value an edge’s producing end carries', () => {
    const row = valueOf(values, 'embed.output');
    expect(row?.geometry).toBe('bf16[tokens, model.width=4096]');
    expect(row?.shape).toBe('[model.width=4096]');
    expect(row?.dtype).toBe('bf16');
    expect(row?.role).toBe('activation.hidden');
    expect(row?.to).toEqual(['decoder/attn_n[layer=0].input', 'decoder/attn_r[layer=0].a']);
    expect(row?.domain).toBe('kind token · stream tokens');
    expect(row?.stream).toBe('tokens');
    expect(row?.count).toBe('tokens 1.0');
  });

  it('finds the value a public input delivers, and what it is required for', () => {
    const row = inputValue(values, 'tokens');
    expect(row?.value).toBe('tokens');
    expect(row?.required).toBe(true);
    expect(row?.requiredFor).toEqual(['logits']);
    expect(row?.to).toEqual(['embed.tokens']);
  });

  it('finds the value a public output exposes — 501 KiB an element, as S15 writes it', () => {
    const row = outputValue(values, 'logits');
    expect(row?.value).toBe('lm_head.logits');
    expect(row?.bytesPerElement).toBe(513024);
    expect(row?.exposed).toEqual(['logits']);
  });

  it('reads the streams D2 reports, for the stream select of §4.15', () => {
    const streams = streamRows(derived);
    expect(streams.map((one) => one.name)).toEqual(['tokens']);
    expect(streams[0]?.kind).toBe('token');
    expect(streams[0]?.count).toBe('tokens 1.0');
    expect(streams[0]?.fragmentAlignment).toBeNull();
  });

  it('reads a fragmented stream’s alignment — voxtral-realtime’s `audio`, a multiple of 8', () => {
    const { derived: voxtral } = open('voxtral-realtime');
    const streams = streamRows(voxtral);
    expect(streams.map((one) => one.name)).toEqual(['audio', 'delay']);
    expect(streams[0]?.fragmentAlignment).toBe(8n);
    expect(streams[1]?.fragmentAlignment).toBeNull();
  });
});
