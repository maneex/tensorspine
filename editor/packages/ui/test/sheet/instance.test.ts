import { describe, expect, it } from 'vitest';

import { targetText } from '../../src/sheet/instance.js';
import { instanceOf } from './source.js';

/**
 * The instance sheet's sections — plan §4.11's first row, artboard S6's second and third columns.
 *
 * Ports, Parameters, Constants, States and Derived, on the selection the feature's block names.
 * Every figure below is the products' own number and every verdict is `describe`'s; what is tested
 * is that the sheet shows them where S6 shows them, and that an absent element is *listed* rather
 * than dropped — "a component that had to ask 'does this slot still exist?' by re-reading
 * `present_when` is the shortcut §7 forbids".
 */

const LLAMA = 'llama3-8b';
const ATTN = 'decoder/attn[layer=0]';

describe('Ports', () => {
  const sheet = instanceOf(LLAMA, ATTN);

  it('lists every declared port, present or not, with what joins it', () => {
    expect(sheet.inputs.map((row) => [row.name, row.present, row.fedBy])).toEqual([
      ['input', true, 'decoder.attn.norm_in'],
      ['source_values', false, null],
    ]);
    expect(sheet.outputs.map((row) => [row.name, row.present, row.consumed])).toEqual([
      ['output', true, true],
    ]);
  });

  it('shows the evaluated shape of a present port, and none for an absent one', () => {
    expect(sheet.inputs[0]?.shape).toBe('model.width=4096');
    expect(sheet.inputs[1]?.shape).toBe('');
  });
});

describe('Parameters', () => {
  const sheet = instanceOf(LLAMA, ATTN);

  it('lists the eleven the primitive declares and marks the four that are present', () => {
    expect(sheet.parameters).toHaveLength(11);
    expect(sheet.parameters.filter((row) => row.present).map((row) => row.name)).toEqual([
      'q',
      'k',
      'v',
      'out',
    ]);
  });

  it('names the identity each bound slot belongs to, with its stored shape', () => {
    const q = sheet.parameters.find((row) => row.name === 'q');
    expect(q?.identity).toBe('decoder.attn.q[layer=0]');
    expect(q?.boundBy).toBe('decoder.attn.q');
    expect(q?.role).toBe('attention.qkv_projection');
    expect(q?.shape).toBe('attention.heads=4096, model.width=4096');
  });

  it('says the primitive declares no constant slot', () => {
    expect(sheet.constants).toEqual([]);
  });
});

describe('States', () => {
  const sheet = instanceOf(LLAMA, ATTN);

  it('shows the applying rule of the ordered four — erratum E8’s rule 4', () => {
    const kv = sheet.states[0];
    expect(kv?.name).toBe('kv');
    expect(kv?.present).toBe(true);
    expect(kv?.rule).toBe(4);
    expect(kv?.rules).toBe(4);
    expect(kv?.evolution).toBe('append');
    expect(kv?.access).toBe('logical_position');
    expect(kv?.sharing).toBe('by_position');
    expect(kv?.indexedBy).toBe('self');
    expect(kv?.written).toBe(true);
    expect(kv?.identity).toBe('decoder.attn.kv[layer=0]');
  });

  it('shows the payload with each component’s evaluated shape', () => {
    expect(sheet.states[0]?.payload).toEqual([
      { name: 'k', shape: 'attention.kv_heads=8, attention.head_dim=128' },
      { name: 'v', shape: 'attention.kv_heads=8, attention.head_dim=128' },
    ]);
  });
});

describe('Derived', () => {
  const sheet = instanceOf(LLAMA, ATTN);

  it('carries D3’s rows for this site’s slots, with the products’ own figures', () => {
    expect(sheet.derived.tensors.map((row) => [row.slot, String(row.bytes)])).toEqual([
      ['q', '33554432'],
      ['k', '8388608'],
      ['v', '8388608'],
      ['out', '33554432'],
    ]);
    expect(sheet.derived.tensors[0]?.identity).toBe('decoder.attn.q[layer=0]');
    expect(sheet.derived.tensors[0]?.shape).toBe('[attention.heads=4096, model.width=4096]');
    expect(sheet.derived.tensors[0]?.located).toBe(true);
  });

  it('carries D4’s row for its state, with the bytes per cached position', () => {
    expect(sheet.derived.states.map((row) => [row.port, String(row.bytesPerCachedPosition)])).toEqual([
      ['kv', '4096'],
    ]);
  });

  it('carries D5’s applying corrections for this node', () => {
    expect(sheet.derived.corrections.map((row) => row.per)).toEqual(['cached_position']);
    expect(sheet.derived.corrections[0]?.status).toBe('exact');
  });

  it('carries D6’s applying partition options — erratum E7’s five', () => {
    expect(sheet.partitions).toHaveLength(5);
    expect(sheet.partitions.map((row) => row.target)).toEqual([
      'argument_axis attention.heads',
      'argument_axis attention.kv_heads',
      'instance_key_axis instance.session',
      'payload_axis kv.k.attention.kv_heads',
      'payload_axis kv.v.attention.kv_heads',
    ]);
    expect(sheet.partitions[0]?.communication).toEqual(['all_reduce']);
    expect(sheet.partitions[0]?.granularity).toBe('4');
  });

  it('counts the D1 nodes the declared site expands to, and its own flag', () => {
    expect(sheet.derived.nodes).toBe(32);
    expect(sheet.derived.acrossPositions).toBe(true);
  });
});

describe('a partition target', () => {
  it('is the tag and the names under it, never Python’s dict repr', () => {
    expect(targetText({ argument_axis: 'attention.heads' })).toBe('argument_axis attention.heads');
    expect(targetText('model.width')).toBe('model.width');
  });
});
