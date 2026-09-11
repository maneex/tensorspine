import { describe, expect, it } from 'vitest';

import {
  analyse,
  check,
  describe as describeDocument,
  formatSemanticProblems,
  generatedSite,
  parse,
  rootSite,
  toPython,
  type Candidate,
  type Description,
  type SiteKey,
  type Verdict,
} from '../../src/index.js';
import { corpus, describedCorpus, library, schemas } from './source.js';

// `check` (feature 1.6d): the verdict on one candidate edit, and its reason.
//
// The plan's Q5 decides what a verdict *is* for: "never refuse — the author wires first and fixes
// afterwards; the core's verdict is shown during the drag and lands in Problems on the drop, never
// blocks it". So a verdict is not a permission but a prediction: **the lines the document would
// then carry**, in the validator's own words. That wording is the parity contract (D2, §7 F1), and
// every line asserted below is pinned character for character — the rejection suite's own idiom.
//
// The feature's block names three: "V7 on an input already fed, V4 on a shape mismatch, V5 on a
// domain mismatch, each with the tools' wording". Beside them are the other rules a candidate can
// break — V1, V6, V15, V9, V17 — and the two readings of a member candidate, "Tie to…" (an
// identity by name) and a chip dragged onto a chip (a slot).

/** A site of a composition at one index value, as a selector names it. */
function at(composition: string, name: string, layer: number): SiteKey {
  return generatedSite(composition, name, [{ name: 'layer', value: BigInt(layer) }]);
}

/** The lines a verdict carries, as `--validate` would print them. */
function lines(verdict: Verdict): string[] {
  return formatSemanticProblems([...verdict.problems]);
}

/** `check` over one corpus document. */
function of(name: string, candidate: Candidate): Verdict {
  return check(describedCorpus(name), candidate);
}

/**
 * `llama3-8b` with `final_n.weight` written as a **slice** of `model.norm.weight` rather than the
 * whole of it — the one edit, as text, so that every other number keeps the float-ness its own
 * spelling gives it (D12).
 */
function slicedFinalNorm(): string {
  const text = corpus('llama3-8b');
  const whole = '"location": {\n          "tensor": [\n            "model.norm.weight"\n          ]\n        }';
  const slice =
    '"location": {\n          "slice": {\n            "tensor": [\n              "model.norm.weight"\n            ],\n' +
    '            "axis": "feature",\n            "offset": {\n              "literal": 0\n            }\n          }\n        }';
  if (!text.includes(whole)) throw new Error('the corpus no longer writes final_n.weight that way');
  return text.replace(whole, slice);
}

/** The same document with the candidate applied: `embed.weight` bound to that whole tensor. */
function applied(text: string): string {
  const before = '"location": {\n          "tensor": [\n            "model.embed_tokens.weight"\n          ]\n        }';
  const after = '"location": {\n          "tensor": [\n            "model.norm.weight"\n          ]\n        }';
  if (!text.includes(before)) throw new Error('the corpus no longer writes embed.weight that way');
  return text.replace(before, after);
}

/** One document of the corpus, edited, described as the editor would describe it. */
function describedText(text: string): Description {
  const tree = parse(text);
  return describeDocument(tree, { schemas, library });
}

describe('a candidate edge', () => {
  it('reports V7 on an input already fed, naming the binding that feeds it', () => {
    // `decoder/attn_n[layer=0].input` is fed by the top-level binding `decoder.entry`; §4.7 has
    // the older edge replaced on the drop, and this is the line the replacement reports.
    const verdict = of('llama3-8b', {
      edge: {
        from: { site: rootSite('embed'), port: 'output' },
        to: { site: at('decoder', 'attn_n', 0), port: 'input' },
        rule: 'attn_n.input',
      },
    });
    expect(lines(verdict)).toEqual([
      '[V7] input port fed twice: decoder/attn_n[layer=0].input by decoder.entry ' +
        'and attn_n.input',
    ]);
    expect(verdict.ok).toBe(false);
    expect(verdict.problems[0]?.path).toBe('/bindings/values/attn_n.input');
  });

  it('reports V7 in the interface block’s words when a public input feeds the port', () => {
    // The tools walk the value bindings before the public inputs, so the refusal is the input's
    // and it names the binding that got there first.
    const verdict = of('llama3-8b', {
      edge: {
        from: { site: rootSite('final_n'), port: 'output' },
        to: { site: rootSite('embed'), port: 'tokens' },
        rule: 'embed.tokens',
      },
    });
    expect(lines(verdict)).toContain(
      '[V7] input tokens: port embed.tokens also fed by embed.tokens',
    );
    expect(verdict.problems.find((one) => one.code === 'V7')?.path).toBe(
      '/interfaces/inputs/tokens',
    );
  });

  it('reports V4 on a shape mismatch, with both shapes as Python writes them', () => {
    const verdict = of('llama3-8b', {
      edge: {
        from: { site: rootSite('lm_head'), port: 'logits' },
        to: { site: rootSite('final_n'), port: 'input' },
        rule: 'final_n.input',
      },
    });
    expect(lines(verdict)).toContain(
      "[V4] final_n.input: shapes do not unify lm_head.logits[('model.vocabulary', 128256)] -> " +
        "norm.rms.input[('model.width', 4096)]",
    );
  });

  it('reports V5 when the port expects one kind and receives another', () => {
    // `lm_head.input` declares the kind `token`; the encoder of `whisper-large-v3` carries its
    // values on the `audio` stream at kind `position`.
    const verdict = of('whisper-large-v3', {
      edge: {
        from: { site: rootSite('enc_final_n'), port: 'output' },
        to: { site: rootSite('lm_head'), port: 'input' },
        rule: 'lm_head.input',
      },
    });
    expect(lines(verdict)).toContain(
      "[V5] lm_head@lm_head.input expects token, receives position (stream 'audio')",
    );
    expect(verdict.problems.find((one) => one.code === 'V5')?.path).toBe('/instances/lm_head');
  });

  it('reports V5 when the instance’s inputs would fall in different domains', () => {
    // `residual.add` takes both operands in one domain and declares no transform: feeding its `b`
    // from the encoder while `a` carries tokens is the refusal §5.3 states.
    const verdict = of('whisper-large-v3', {
      edge: {
        from: { site: rootSite('enc_final_n'), port: 'output' },
        to: { site: at('decoder', 'self_attn_r', 0), port: 'b' },
        rule: 'self_attn_r.b',
      },
    });
    expect(lines(verdict)).toContain(
      "[V5] residual.add@decoder/self_attn_r[layer=0]: inputs in different domains " +
        "[('position', 'audio'), ('token', 'tokens')], and no domain_transform declares it",
    );
  });

  it('reports V6 when the candidate closes a cycle in the value graph', () => {
    const verdict = of('llama3-8b', {
      edge: {
        from: { site: rootSite('final_n'), port: 'output' },
        to: { site: at('decoder', 'attn_n', 0), port: 'input' },
        rule: 'attn_n.input',
      },
    });
    const cycle = lines(verdict).filter((line) => line.startsWith('[V6]'));
    expect(cycle).toHaveLength(1);
    expect(cycle[0]).toMatch(/^\[V6\] value cycle: \d+ instance\(s\) in a cycle$/);
  });

  it('reports no cycle for an edge that closes none', () => {
    const verdict = of('llama3-8b', {
      edge: {
        from: { site: rootSite('embed'), port: 'output' },
        to: { site: rootSite('lm_head'), port: 'input' },
        rule: 'lm_head.input',
      },
    });
    expect(lines(verdict).filter((line) => line.startsWith('[V6]'))).toEqual([]);
  });

  it('reports V1 for an instance and for a port the document does not have', () => {
    expect(
      lines(
        of('llama3-8b', {
          edge: {
            from: { site: rootSite('nowhere'), port: 'output' },
            to: { site: rootSite('lm_head'), port: 'input' },
            rule: 'lm_head.input',
          },
        }),
      ),
    ).toEqual(['[V1] lm_head.input{}: from instance does not exist nowhere']);
    expect(
      lines(
        of('llama3-8b', {
          edge: {
            from: { site: rootSite('embed'), port: 'nowhere' },
            to: { site: rootSite('lm_head'), port: 'input' },
            rule: 'lm_head.input',
          },
        }),
      ),
    ).toEqual(["[V1] lm_head.input: embed has no output port 'nowhere'"]);
  });

  it('names the binding as §4.7 proposes it when the caller gives no name', () => {
    const verdict = of('llama3-8b', {
      edge: {
        from: { site: rootSite('embed'), port: 'output' },
        to: { site: at('decoder', 'attn_n', 0), port: 'input' },
      },
    });
    expect(lines(verdict)[0]).toContain('and attn_n.input');
  });
});

describe('a candidate member of a parameter identity', () => {
  it('admits `lm_head.weight` into the identity `embed.weight` on qwen3.5-4b-text', () => {
    // The two slots declare the role `embedding.table`, share with it, and have the same stored
    // shape: V15 admits the tie, which is what the corpus document itself writes.
    const verdict = of('qwen3.5-4b-text', {
      member: {
        kind: 'parameter',
        slot: { site: rootSite('lm_head'), name: 'weight' },
        into: { identity: 'embed.weight' },
      },
    });
    expect(verdict.ok).toBe(true);
    expect(verdict.problems).toEqual([]);
    expect(verdict.unknown).toBe(null);
  });

  it('refuses `embed.weight` into `final_n.weight`, with V15’s three reasons', () => {
    const verdict = of('qwen3.5-4b-text', {
      member: {
        kind: 'parameter',
        slot: { site: rootSite('embed'), name: 'weight' },
        into: { identity: 'final_n.weight' },
      },
    });
    expect(lines(verdict)).toEqual([
      '[V15] final_n.weight: norm.rms.weight is exclusive, it cannot be tied',
      "[V15] final_n.weight: norm.rms.weight does not share with role 'embedding.table'",
      "[V15] final_n.weight: incompatible shapes [('model.width', 2560)] vs " +
        "[('model.vocabulary', 248320), ('model.width', 2560)]",
      "[V15] final_n.weight: embed.weight does not share with role 'norm.scale'",
      "[V15] final_n.weight: incompatible shapes [('model.vocabulary', 248320), " +
        "('model.width', 2560)] vs [('model.width', 2560)]",
    ]);
    expect(verdict.problems.every((one) => one.path === '/bindings/parameters/final_n.weight')).toBe(
      true,
    );
  });

  it('reads a chip dragged onto another chip as the identity that one belongs to', () => {
    // §4.7: "Drag a slot chip onto another node's slot chip … creates or extends the identity."
    const verdict = of('llama3-8b', {
      member: {
        kind: 'parameter',
        slot: { site: rootSite('embed'), name: 'weight' },
        into: { slot: { site: rootSite('lm_head'), name: 'weight' } },
      },
    });
    expect(verdict.ok).toBe(true);
    expect(
      lines(
        of('llama3-8b', {
          member: {
            kind: 'parameter',
            slot: { site: rootSite('embed'), name: 'weight' },
            into: { slot: { site: rootSite('final_n'), name: 'weight' } },
          },
        }),
      ).length,
    ).toBe(5);
  });

  it('reports V7 about the slot itself before any question of compatibility', () => {
    expect(
      lines(
        of('llama3-8b', {
          member: {
            kind: 'parameter',
            slot: { site: at('decoder', 'attn', 0), name: 'q_gated' },
            into: { identity: 'embed.weight' },
          },
        }),
      ),
    ).toEqual(["[V7] parameter embed.weight: slot 'q_gated' absent for these arguments"]);
    expect(
      lines(
        of('llama3-8b', {
          member: {
            kind: 'parameter',
            slot: { site: rootSite('embed'), name: 'nowhere' },
            into: { identity: 'embed.weight' },
          },
        }),
      ),
    ).toEqual(["[V7] parameter embed.weight: embed has no parameter 'nowhere'"]);
    expect(
      lines(
        of('llama3-8b', {
          member: {
            kind: 'parameter',
            slot: { site: rootSite('nowhere'), name: 'weight' },
            into: { identity: 'embed.weight' },
          },
        }),
      ),
    ).toEqual(['[V1] parameter embed.weight: instance does not exist nowhere']);
  });

  it('judges the identity a slot would join, not the one it leaves', () => {
    // V7 makes every slot of a valid document bound exactly once, so a candidate that reported
    // the membership it leaves as a second binding would refuse every tie the list offers. What
    // it answers is V15 over the members the slot would stand beside.
    expect(
      lines(
        of('llama3-8b', {
          member: {
            kind: 'parameter',
            slot: { site: rootSite('final_n'), name: 'weight' },
            into: { identity: 'embed.weight' },
          },
        }),
      ).every((line) => line.startsWith('[V15]')),
    ).toBe(true);
  });

  it('states what it cannot decide rather than guessing a refusal', () => {
    const verdict = of('llama3-8b', {
      member: {
        kind: 'parameter',
        slot: { site: rootSite('embed'), name: 'weight' },
        into: { identity: 'no.such.identity' },
      },
    });
    expect(verdict.ok).toBe(false);
    expect(verdict.problems).toEqual([]);
    expect(verdict.unknown).toBe(
      "no parameter identity instance named 'no.such.identity' was derived from this document",
    );
  });
});

describe('a candidate member of a state identity', () => {
  it('admits a `kv` port into another layer’s identity', () => {
    const verdict = of('llama3-8b', {
      member: {
        kind: 'state',
        slot: { site: at('decoder', 'attn', 0), name: 'kv' },
        into: { identity: 'decoder.attn.kv[layer=1]' },
      },
    });
    expect(verdict.ok).toBe(true);
  });

  it('refuses a self-attention `kv` into a cross-attention identity, with V9’s words', () => {
    // The cross-attention state is indexed by its `source_values` port and grows along the audio
    // stream; the self-attention state grows along the token stream under another rule.
    const verdict = of('whisper-large-v3', {
      member: {
        kind: 'state',
        slot: { site: at('decoder', 'self_attn', 0), name: 'kv' },
        into: { identity: 'decoder.cross_attn.kv[layer=0]' },
      },
    });
    expect(lines(verdict)).toEqual([
      '[V9] state decoder.cross_attn.kv: members under different derivation rules',
      '[V9] state decoder.cross_attn.kv: members indexed by different streams',
    ]);
    expect(verdict.problems.every((one) => one.path === '/bindings/states/decoder.cross_attn.kv')).toBe(
      true,
    );
  });

  it('reports V7 and V1 about the port itself', () => {
    expect(
      lines(
        of('shieldstral-3b-composite', {
          member: {
            kind: 'state',
            slot: { site: at('vision', 'attn', 0), name: 'kv' },
            into: { identity: 'no.such.identity' },
          },
        }),
      ),
    ).toEqual([]);
    expect(
      lines(
        of('llama3-8b', {
          member: {
            kind: 'state',
            slot: { site: rootSite('embed'), name: 'kv' },
            into: { identity: 'decoder.attn.kv[layer=0]' },
          },
        }),
      ),
    ).toEqual(["[V1] state decoder.attn.kv: embed has no state port 'kv'"]);
  });
});

describe('a candidate location', () => {
  it('accepts a physical name nothing else binds', () => {
    const verdict = of('llama3-8b', {
      location: { identity: 'embed.weight', location: { tensor: ['a.brand.new.tensor'] } },
    });
    expect(verdict.ok).toBe(true);
  });

  it('reports V17 when the name is already bound by another identity', () => {
    const verdict = of('llama3-8b', {
      location: { identity: 'embed.weight', location: { tensor: ['model.norm.weight'] } },
    });
    expect(lines(verdict)).toEqual([
      "[V17] embed.weight: physical tensor 'model.norm.weight' already bound by final_n.weight",
    ]);
    expect(verdict.problems[0]?.path).toBe('/bindings/parameters/embed.weight/location');
  });

  it('reports V17 when a form names an axis the slot does not have', () => {
    const verdict = of('llama3-8b', {
      location: {
        identity: 'embed.weight',
        location: { stack: { axis: 'nowhere', part: { tensor: ['x'] } } },
      },
    });
    expect(lines(verdict)).toEqual([
      "[V17] embed.weight: stack: 'nowhere' is not an axis of the slot (axes: vocabulary, feature)",
    ]);
  });

  it('evaluates a name in the index environment the rule fired in', () => {
    // `decoder.attn.q` binds one identity per layer and its location prints the index; the
    // candidate is evaluated where that rule fires, so `{index: layer}` resolves.
    const verdict = of('llama3-8b', {
      location: {
        identity: 'decoder.attn.q[layer=3]',
        location: { tensor: ['model.layers.', { index: 'layer' }, '.self_attn.q_proj.weight'] },
      },
    });
    // The very name the document already binds for that identity: nothing else claims it, since
    // the identity's own binding is the one being replaced.
    expect(verdict.ok).toBe(true);
  });

  it('states what it cannot decide for an identity the document never derived', () => {
    const verdict = of('llama3-8b', {
      location: { identity: 'no.such.identity', location: { tensor: ['x'] } },
    });
    expect(verdict.problems).toEqual([]);
    expect(verdict.unknown).toContain("'no.such.identity'");
  });

  it('reports V17 when the candidate binds whole a name the document slices', () => {
    // The other side of the same rule, and the one a reading over the *slices* can miss: V17 is
    // computed by walking the slices and asking `whole` about each name, so a candidate that
    // binds a name whole is only seen if the document's slices of it are carried in.
    //
    // What it is held to is `analyse` itself: the lines the applied document carries, taken from
    // the validator rather than typed here, so this is a parity test between the two readings and
    // not a transcription of one of them.
    const sliced = slicedFinalNorm();
    const candidate: Candidate = {
      location: { identity: 'embed.weight', location: { tensor: ['model.norm.weight'] } },
    };
    const verdict = check(describedText(sliced), candidate);
    expect(lines(verdict)).toEqual(
      formatSemanticProblems([...analyse(toPython(parse(applied(sliced))) as never, library, {}).problems]),
    );
    expect(lines(verdict)).toEqual([
      "[V17] physical tensor 'model.norm.weight' is bound whole by embed.weight and sliced by final_n.weight",
    ]);
  });
});

describe('a document off the grammar', () => {
  it('decides nothing about a candidate', () => {
    const verdict = check(
      { conforms: false, structural: [], analysis: null, sites: new Map() },
      { edge: { from: { site: rootSite('a'), port: 'x' }, to: { site: rootSite('b'), port: 'y' } } },
    );
    expect(verdict.ok).toBe(false);
    expect(verdict.problems).toEqual([]);
    expect(verdict.unknown).toContain('off the grammar');
  });
});
