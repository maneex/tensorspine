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
import { fixture } from '../derive/source.js';

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

/** The lines `--validate`'s semantic stage prints for a document, from `analyse` itself. */
function analysedLines(text: string): string[] {
  return formatSemanticProblems([...analyse(toPython(parse(text)), library, {}).problems]);
}

const LAYERS_32 =
  '"layers": {\n      "type": {\n        "kind": "cardinality"\n      },\n      "source": {\n' +
  '        "kind": "literal",\n        "value": 32\n      }\n    },';
const STOP_32 = '"stop": {\n            "literal": 32\n          },';
const ATTN_K = '"attn.k": {\n            "members": [\n              {\n                "site": "attn",\n                "parameter": "k"\n              }\n            ],';

/**
 * `llama3-8b` cut to one layer, with the identity `attn.k` given the members named.
 *
 * One layer because the assertions are about one identity and the document repeats every line per
 * layer; the edits are textual, so every other number keeps the float-ness its own spelling gives
 * it (D12). `q_bias` is a slot of `attention.dense` that is **absent** unless the site's `q_bias`
 * argument is true, which this document's is not: binding it is a defect the document carries
 * (V7), and the question is what `check` then says about a candidate joining that identity.
 */
function oneLayerWithMembers(members: readonly string[]): string {
  const text = corpus('llama3-8b');
  for (const anchor of [LAYERS_32, STOP_32, ATTN_K]) {
    if (!text.includes(anchor)) throw new Error('the corpus no longer writes that block that way');
  }
  const rows = members
    .map(
      (name) =>
        `              {\n                "site": "attn",\n                "parameter": "${name}"\n              }`,
    )
    .join(',\n');
  return text
    .replace(LAYERS_32, LAYERS_32.replace('"value": 32', '"value": 1'))
    .replace(STOP_32, STOP_32.replace('"literal": 32', '"literal": 1'))
    .replace(ATTN_K, `"attn.k": {\n            "members": [\n${rows}\n            ],`);
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

  it('leaves an absent slot out of V15, as the validator does', () => {
    // `check_parameters` `continue`s past a slot that is absent under its instance's arguments —
    // "slot 'q_bias' absent for these arguments" — **before** its signature is collected, so the
    // tying is decided over the present members alone. A reading that collected it would judge
    // the candidate against a shape and a role the identity does not have.
    const held = oneLayerWithMembers(['k', 'q_bias']);
    const applied = oneLayerWithMembers(['k', 'q_bias', 'v']);
    const verdict = check(describedText(held), {
      member: {
        kind: 'parameter',
        slot: { site: at('decoder', 'attn', 0), name: 'v' },
        into: { identity: 'decoder.attn.k[layer=0]' },
      },
    });
    const v15 = (rows: readonly string[]): string[] => rows.filter((row) => row.startsWith('[V15]'));
    // The lines the validator prints for the document the drop would make, and no others.
    expect(v15(lines(verdict))).toEqual(v15(analysedLines(applied)));
    expect(lines(verdict).join('\n')).not.toContain('q_bias');
    // The document's own defect is still the document's: `check` speaks of the candidate.
    expect(analysedLines(held)).toContain(
      "[V7] parameter decoder.attn.k: slot 'q_bias' absent for these arguments",
    );
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

describe('a candidate onto a site the guards removed', () => {
  // §5.2 rule 3: "a guarded site that does not fire is not an instance, and is remembered as
  // absent: a binding naming it is not emitted". The validator skips such a binding in silence —
  // no V1, no line of any kind — because the site *is* declared; it is the document's own guard
  // that removed it. `check` must therefore say nothing about a gesture onto one, rather than
  // refusing a drop the document would accept without a word.
  //
  // The fixture is feature 1.13's `root-level-when`: `variant` is `small`, so `probe` does not
  // fire, and `probe.weight` already binds it — which is why the document validates clean.
  const text = fixture('root-level-when');
  const described = describeDocument(parse(text), { schemas, library });

  /** The lines `analyse` gives for the fixture, edited. */
  function linesOf(edited: string): string[] {
    return formatSemanticProblems([...analyse(toPython(parse(edited)), library, {}).problems]);
  }

  it('is the guard that removed it, so the document itself is clean', () => {
    expect(linesOf(text)).toEqual([]);
    expect(analyse(toPython(parse(text)), library, {}).absent.size).toBe(1);
  });

  it('says nothing about an edge drawn from one', () => {
    const verdict = check(described, {
      edge: {
        from: { site: rootSite('probe'), port: 'output' },
        to: { site: rootSite('exit'), port: 'input' },
        rule: 'exit.from_probe',
      },
    });
    expect(verdict.problems).toEqual([]);
    expect(verdict.unknown).toBeNull();
    expect(verdict.ok).toBe(true);
    // And the document the drop would make carries nothing either — not even the V7 about an
    // input already fed, because the binding is never emitted.
    expect(linesOf(withEdgeFromProbe(text))).toEqual([]);
  });

  it('says nothing about a member added from one', () => {
    const verdict = check(described, {
      member: {
        kind: 'parameter',
        slot: { site: rootSite('probe'), name: 'weight' },
        into: { identity: 'entry.weight' },
      },
    });
    expect(verdict.problems).toEqual([]);
    expect(verdict.unknown).toBeNull();
    // What the applied document carries is **not** a word about `probe`: the binding is skipped
    // whole, which is also what leaves `entry`'s own slot with nothing binding it. That last line
    // is a consequence of the *document*, counted over every slot once the bindings are read, and
    // no candidate-local reading produces it — `check` predicts the lines its own binding would
    // carry, and the skip means there are none.
    const applied = linesOf(withProbeInEntryWeight(text));
    expect(applied.filter((row) => row.includes('does not exist'))).toEqual([]);
    expect(applied).toEqual(['[V7] unbound parameter slot: norm.rms@entry.weight']);
  });
});

/** The fixture with a value binding drawn from the absent `probe` to `exit`. */
function withEdgeFromProbe(text: string): string {
  const anchor = '"values": {\n      ';
  const added =
    '"exit.from_probe": {\n        "from": {\n          "instance": {\n            "kind": "root",\n' +
    '            "instance": "probe"\n          },\n          "port": "output"\n        },\n' +
    '        "to": {\n          "instance": {\n            "kind": "root",\n            "instance": "exit"\n' +
    '          },\n          "port": "input"\n        }\n      },\n      ';
  if (!text.includes(anchor)) throw new Error('the fixture no longer writes its value bindings that way');
  return text.replace(anchor, anchor + added);
}

/** The fixture with the absent `probe`'s weight slot added to the identity `entry.weight`. */
function withProbeInEntryWeight(text: string): string {
  const anchor =
    '"members": [\n          {\n            "instance": {\n              "kind": "root",\n' +
    '              "instance": "entry"\n            },\n            "parameter": "weight"\n          }\n        ],';
  const added =
    '"members": [\n          {\n            "instance": {\n              "kind": "root",\n' +
    '              "instance": "entry"\n            },\n            "parameter": "weight"\n          },\n' +
    '          {\n            "instance": {\n              "kind": "root",\n' +
    '              "instance": "probe"\n            },\n            "parameter": "weight"\n          }\n        ],';
  if (!text.includes(anchor)) throw new Error('the fixture no longer writes entry.weight that way');
  return text.replace(anchor, added);
}

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
