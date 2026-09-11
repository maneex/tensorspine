import { describe, expect, it } from 'vitest';

import {
  checkCheckpoint,
  formatCheckpointProblem,
  squeeze,
  type CheckpointCheck,
  type CheckpointHeaders,
  type CheckpointProblem,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { PyIndexError, PyKeyError, PyValueError } from '../../src/expr/errors.js';
import {
  checkpointDirectory,
  d3Of,
  documentOf,
  entry,
  headersOf,
  readCheckpoint,
  withEntry,
  withoutEntry,
} from './source.js';

// V17 against a checkpoint (feature 1.9): the cases of `tests/run_artifact.py`, reproduced.
//
// The script is the repository's own suite for `tools/artifact.py`, and its cases are what this
// feature's block names: "clean, absent, shape wrong, dtype wrong, unit axes dropped, stack count,
// concat sum, slice fit and overlap … with the tools' wording". Each one below is the script's,
// built from the same material — llama3-8b's own D3, derived here by the core — and asserted on
// the tools' line character for character, the lines having been taken from `artifact.check` run
// on the same input.
//
// Two of the cases are not the script's and are named where they appear: the **warnings** §4.19
// asks for beside the tools' advisories, and the **overlap** half of V17, which is the
// *validator's* rule and not the checkpoint's — `artifact.py` never compares two slices, and
// `tests/rejections/models/v17-slice-overlap.json` is decided by `validate.analyse` (feature
// 1.6c). It is stated here so that the two halves of V17 cannot be confused for one.

/** `tests/run_artifact.py`'s `d3_of`: one identity, one location, one shape. */
function syntheticD3(location: PyValue, shape: readonly bigint[], dtype = 'bf16'): PyRecord {
  return {
    tensors: [
      {
        identity: 't',
        dtype,
        location,
        shape: shape.map((extent) => ({ axis: 'a', extent })),
      },
    ],
  };
}

/** The check over a hand-built D3. */
function checked(d3: PyRecord, headers: CheckpointHeaders): CheckpointCheck {
  return checkCheckpoint(documentOf(d3), headers);
}

/** The tools' `errors`: the messages, without the `[V17]` its printer adds. */
function messages(problems: readonly CheckpointProblem[]): string[] {
  return problems.map((problem) => problem.message);
}

/** Python's `==` over two lists of extents, for the proof below. */
function equalShapes(left: readonly PyValue[], right: readonly PyValue[]): boolean {
  return left.length === right.length && left.every((one, index) => one === right[index]);
}

const llama = () => d3Of('llama3-8b');

describe('llama3-8b against headers built from its own D3', () => {
  it('locates every one of its 291 tensors', () => {
    const tensors = (llama() as PyRecord)['tensors'] as readonly PyRecord[];
    expect(tensors).toHaveLength(291);
    expect(tensors.every((tensor) => 'location' in tensor)).toBe(true);
  });

  it('clean headers: no error, no advice', () => {
    const answer = checkCheckpoint(documentOf(llama()), headersOf(llama()));
    expect(answer.errors).toEqual([]);
    expect(answer.warnings).toEqual([]);
    expect(answer.stats).toEqual({ located: 291, physical: 291, unnamed: 0 });
  });

  it('an absent tensor is an error naming it', () => {
    const answer = checkCheckpoint(
      documentOf(llama()),
      withoutEntry(headersOf(llama()), 'model.layers.3.self_attn.q_proj.weight'),
    );
    expect(messages(answer.errors)).toEqual([
      "decoder.attn.q[layer=3]: physical tensor 'model.layers.3.self_attn.q_proj.weight' " +
        'is absent from the checkpoint',
    ]);
    expect(answer.errors[0]).toMatchObject({
      code: 'V17',
      severity: 'error',
      kind: 'absent',
      identity: 'decoder.attn.q[layer=3]',
      tensor: 'model.layers.3.self_attn.q_proj.weight',
    });
    expect(formatCheckpointProblem(answer.errors[0] as CheckpointProblem)).toMatch(/^\[V17] /);
  });

  it('a wrong shape is an error', () => {
    const answer = checkCheckpoint(
      documentOf(llama()),
      withEntry(headersOf(llama()), 'model.norm.weight', { shape: [4095n] }),
    );
    expect(messages(answer.errors)).toEqual([
      "final_n.weight: 'model.norm.weight' has shape [4095], the document says [4096]",
    ]);
    expect(answer.errors[0]?.kind).toBe('shape');
  });

  it('a wrong dtype is an error', () => {
    const answer = checkCheckpoint(
      documentOf(llama()),
      withEntry(headersOf(llama()), 'lm_head.weight', { dtype: 'f32' }),
    );
    expect(messages(answer.errors)).toEqual([
      "lm_head.weight: 'lm_head.weight' is f32, the document says bf16",
    ]);
    expect(answer.errors[0]?.kind).toBe('dtype');
  });

  it('unit axes the logical shape lacks are dropped', () => {
    // "A physical tensor may carry unit axes the logical shape lacks (`torch.nn.Conv1d` stores
    // `[C, 1, K]`); they are dropped before shapes are compared" — `artifact.py`'s own header.
    const answer = checkCheckpoint(
      documentOf(llama()),
      withEntry(headersOf(llama()), 'model.norm.weight', { shape: [1n, 4096n, 1n] }),
    );
    expect(answer.errors).toEqual([]);
    expect(squeeze([1n, 4096n, 1n])).toEqual([4096n]);
  });

  it('the second half of the shape test can never decide — a finding, ported as written', () => {
    // `if squeeze(h['shape']) != squeeze(logical) or len(h['shape']) < len(squeeze(logical))`.
    // The second clause is reached only when the first is false, and then it is unsatisfiable:
    // `squeeze(a) == squeeze(b)` gives `len(squeeze(a)) == len(squeeze(b))`, and a list is never
    // shorter than its own squeeze, so `len(h) >= len(squeeze(h)) == len(squeeze(logical))`. It is
    // dead code in the tools and is kept here as written; this is the proof, exhaustive over every
    // shape of rank 0 to 3 whose extents are 1 or 2.
    const shapes: bigint[][] = [[]];
    for (let rank = 1; rank <= 3; rank += 1) {
      for (const shorter of shapes.filter((one) => one.length === rank - 1)) {
        shapes.push([...shorter, 1n], [...shorter, 2n]);
      }
    }
    let reached = 0;
    for (const physical of shapes) {
      for (const logical of shapes) {
        if (!equalShapes(squeeze(physical), squeeze(logical))) continue;
        expect(physical.length >= squeeze(logical).length, `${String(physical)}`).toBe(true);
        reached += 1;
      }
    }
    expect(reached).toBeGreaterThan(0);
  });

  it('an unnamed physical tensor is a warning, not an error', () => {
    // Q7's "what must not be lost": a warning whenever the model does not use every tensor of the
    // checkpoint. `artifact.check` calls it an advisory; §4.19 calls it a warning.
    const headers: CheckpointHeaders = {
      ...headersOf(llama()),
      'model.rotary_emb.inv_freq': entry('f32', [64n]),
    };
    const answer = checkCheckpoint(documentOf(llama()), headers);
    expect(answer.errors).toEqual([]);
    expect(messages(answer.warnings)).toEqual([
      "physical tensor 'model.rotary_emb.inv_freq' (f32 [64]) is named by no location",
    ]);
    expect(answer.warnings[0]).toMatchObject({
      severity: 'warning',
      kind: 'unnamed',
      tensor: 'model.rotary_emb.inv_freq',
      file: 'x',
    });
    expect(answer.stats).toEqual({ located: 291, physical: 292, unnamed: 1 });
  });

  it('lists the unnamed tensors in Python’s own order', () => {
    // `sorted(set(headers) - used)` is a code-point sort. JavaScript's own sort compares UTF-16
    // code units, which puts a surrogate pair (U+10000 and above) *before* U+E000–U+FFFF — the
    // difference feature 1.1 found in `repr`'s name ordering. A physical name is a free string,
    // so a checkpoint can carry one; the two orders disagree here and Python's is the answer.
    const astral = 'w\u{10000}';
    const replacement = 'w�';
    expect([astral, replacement].sort()).toEqual([astral, replacement]);
    const headers: CheckpointHeaders = {
      ...headersOf(llama()),
      zz: entry('f32', [1n]),
      'Aa.weight': entry('f32', [1n]),
      'aa.weight': entry('f32', [1n]),
      [astral]: entry('f32', [1n]),
      [replacement]: entry('f32', [1n]),
    };
    const answer = checkCheckpoint(documentOf(llama()), headers);
    expect(answer.warnings.map((problem) => problem.tensor)).toEqual([
      'Aa.weight',
      'aa.weight',
      replacement,
      astral,
      'zz',
    ]);
  });
});

describe('the four location forms on a synthetic D3', () => {
  const stack = {
    stack: { axis: 'e', dim: 0n, parts: [0, 1, 2].map((i) => ({ tensor: `w.${String(i)}` })) },
  };
  const stackHeaders: CheckpointHeaders = {
    'w.0': entry('bf16', [4n]),
    'w.1': entry('bf16', [4n]),
    'w.2': entry('bf16', [4n]),
  };

  it('stack: three [4] make [3, 4]', () => {
    expect(checked(syntheticD3(stack, [3n, 4n]), stackHeaders).errors).toEqual([]);
  });

  it('stack: a missing part is an error', () => {
    const answer = checked(syntheticD3(stack, [3n, 4n]), withoutEntry(stackHeaders, 'w.1'));
    expect(messages(answer.errors)).toEqual([
      "t: physical tensor 'w.1' is absent from the checkpoint",
    ]);
  });

  it('stack: the parts fill the shape without the stacked axis, wherever that axis is', () => {
    // `inner = logical[:dim] + logical[dim + 1:]`. Every stack of the corpus is at dim 0 (the two
    // of `gemma3n-kvshare`, over the storage axis), so a port that dropped the leading extent
    // instead of the named one would pass every parity case.
    const second = {
      stack: { axis: 'e', dim: 1n, parts: [{ tensor: 'w.0' }, { tensor: 'w.1' }] },
    };
    const headers: CheckpointHeaders = {
      'w.0': entry('bf16', [3n, 4n]),
      'w.1': entry('bf16', [3n, 4n]),
    };
    expect(checked(syntheticD3(second, [3n, 2n, 4n]), headers).errors).toEqual([]);
    expect(messages(checked(syntheticD3(second, [2n, 3n, 4n]), headers).errors)).toEqual([
      "t: stack of 2 along 'e' of extent 3",
      "t: 'w.0' has shape [3, 4], the document says [2, 4]",
      "t: 'w.1' has shape [3, 4], the document says [2, 4]",
    ]);
  });

  it('stack: a count that is not the axis’s extent is an error', () => {
    const answer = checked(syntheticD3(stack, [4n, 4n]), {
      ...stackHeaders,
      'w.3': entry('bf16', [4n]),
    });
    expect(messages(answer.errors)).toEqual(["t: stack of 3 along 'e' of extent 4"]);
    expect(answer.errors[0]?.kind).toBe('stack');
    // The part that is there is still checked, and the fourth tensor is nobody's.
    expect(messages(answer.warnings)).toEqual([
      "physical tensor 'w.3' (bf16 [4]) is named by no location",
    ]);
  });

  const concat = { concat: { axis: 'r', dim: 0n, parts: [{ tensor: 'g' }, { tensor: 'u' }] } };
  const concatHeaders: CheckpointHeaders = {
    g: entry('bf16', [2n, 4n]),
    u: entry('bf16', [4n, 4n]),
  };

  it('concat: [2,4] and [4,4] make [6, 4]', () => {
    expect(checked(syntheticD3(concat, [6n, 4n]), concatHeaders).errors).toEqual([]);
  });

  it('concat: parts that do not sum are an error', () => {
    const answer = checked(syntheticD3(concat, [5n, 4n]), concatHeaders);
    expect(messages(answer.errors)).toEqual([
      "t: concat parts sum to 6 along 'r', the document says 5",
    ]);
    expect(answer.errors[0]?.kind).toBe('concat');
  });

  it('concat: an absent part reports the absence and stops the sum', () => {
    const answer = checked(syntheticD3(concat, [6n, 4n]), { g: concatHeaders['g'] as never });
    expect(messages(answer.errors)).toEqual([
      "t: physical tensor 'u' is absent from the checkpoint",
    ]);
  });

  it('concat: a part of the wrong rank is an error and stops the sum', () => {
    const answer = checked(syntheticD3(concat, [6n, 4n]), {
      ...concatHeaders,
      g: entry('bf16', [2n]),
    });
    expect(messages(answer.errors)).toEqual(["t: 'g' has shape [2], a part of [6, 4]"]);
  });

  const slice = { slice: { tensor: 'big', axis: 'r', dim: 0n, offset: 3n, extent: 4n } };
  const sliceHeaders: CheckpointHeaders = { big: entry('bf16', [10n, 4n]) };

  it('slice: [3, 7) of [10, 4] fits [4, 4]', () => {
    expect(checked(syntheticD3(slice, [4n, 4n]), sliceHeaders).errors).toEqual([]);
  });

  it('slice: [8, 12) of [10, 4] does not fit', () => {
    const answer = checked(
      syntheticD3({ slice: { ...slice.slice, offset: 8n } }, [4n, 4n]),
      sliceHeaders,
    );
    expect(messages(answer.errors)).toEqual([
      "t: 'big' has shape [10, 4]; the slice [8, 12) along 'r' does not fit [4, 4]",
    ]);
    expect(answer.errors[0]?.kind).toBe('slice');
  });

  it('slice: an axis beside the sliced one that differs does not fit either', () => {
    // `other` compares every axis but the sliced one; the extent along the sliced one may be
    // larger. A physical `[10, 5]` is long enough for `[3, 7)` and still holds the wrong rows.
    const answer = checked(syntheticD3(slice, [4n, 4n]), { big: entry('bf16', [10n, 5n]) });
    expect(messages(answer.errors)).toEqual([
      "t: 'big' has shape [10, 5]; the slice [3, 7) along 'r' does not fit [4, 4]",
    ]);
  });

  it('slice: an absent tensor, a rank that differs, a dtype that differs', () => {
    expect(messages(checked(syntheticD3(slice, [4n, 4n]), {}).errors)).toEqual([
      "t: physical tensor 'big' is absent from the checkpoint",
    ]);
    expect(
      messages(checked(syntheticD3(slice, [4n, 4n]), { big: entry('bf16', [10n]) }).errors),
    ).toEqual(["t: 'big' has shape [10], sliced for [4, 4]"]);
    expect(
      messages(checked(syntheticD3(slice, [4n, 4n]), { big: entry('f32', [10n, 4n]) }).errors),
    ).toEqual(["t: 'big' is f32, the document says bf16"]);
  });

  it('slice: the overlap of two slices is the validator’s rule, not the checkpoint’s', () => {
    // V17 has two halves. "The slices of one physical tensor do not overlap and do not coexist
    // with a whole binding of it" is decided on the *document*, by `validate.analyse`
    // (feature 1.6c, `tests/rejections/models/v17-slice-overlap.json`); `artifact.py` never
    // compares two slices with each other. Two identities slicing one tensor at overlapping
    // offsets therefore pass here — each slice fits — and are refused before any derivation.
    const two: PyRecord = {
      tensors: [
        {
          identity: 'a',
          dtype: 'bf16',
          location: { slice: { tensor: 'big', axis: 'r', dim: 0n, offset: 0n, extent: 6n } },
          shape: [{ axis: 'a', extent: 6n }, { axis: 'b', extent: 4n }],
        },
        {
          identity: 'b',
          dtype: 'bf16',
          location: { slice: { tensor: 'big', axis: 'r', dim: 0n, offset: 4n, extent: 6n } },
          shape: [{ axis: 'a', extent: 6n }, { axis: 'b', extent: 4n }],
        },
      ],
    };
    const answer = checked(two, sliceHeaders);
    expect(answer.errors).toEqual([]);
    expect(answer.stats).toEqual({ located: 2, physical: 1, unnamed: 0 });
  });
});

describe('a slot with a multiplicity: the storage axis first (§3.4, finding 30)', () => {
  const stacked = {
    stack: {
      axis: 'multiplicity',
      dim: 0n,
      parts: [0, 1, 2].map((i) => ({ tensor: `p.${String(i)}.weight` })),
    },
  };
  const copies: CheckpointHeaders = {
    'p.0.weight': entry('bf16', [2n, 2n]),
    'p.1.weight': entry('bf16', [2n, 2n]),
    'p.2.weight': entry('bf16', [2n, 2n]),
  };

  it('three [2, 2] under a stack over the storage axis make [3, 2, 2]', () => {
    expect(checked(syntheticD3(stacked, [3n, 2n, 2n]), copies).errors).toEqual([]);
  });

  it('a missing copy is an error naming it', () => {
    const answer = checked(
      syntheticD3(stacked, [3n, 2n, 2n]),
      withoutEntry(copies, 'p.2.weight'),
    );
    expect(messages(answer.errors)).toEqual([
      "t: physical tensor 'p.2.weight' is absent from the checkpoint",
    ]);
  });

  it('one fused [3, 2, 2] tensor is the copies whole', () => {
    const answer = checked(syntheticD3({ tensor: 'p.weight' }, [3n, 2n, 2n]), {
      'p.weight': entry('bf16', [3n, 2n, 2n]),
    });
    expect(answer.errors).toEqual([]);
  });

  it('a fused tensor with the wrong count is an error', () => {
    const answer = checked(syntheticD3({ tensor: 'p.weight' }, [3n, 2n, 2n]), {
      'p.weight': entry('bf16', [2n, 2n, 2n]),
    });
    expect(messages(answer.errors)).toEqual([
      "t: 'p.weight' has shape [2, 2, 2], the document says [3, 2, 2]",
    ]);
  });

  it('a declared count of one accepts the plain [2, 2] tensor for [1, 2, 2]', () => {
    const answer = checked(syntheticD3({ tensor: 'p.weight' }, [1n, 2n, 2n]), {
      'p.weight': entry('bf16', [2n, 2n]),
    });
    expect(answer.errors).toEqual([]);
  });
});

describe('the composite (§3.4): its instance’s tensors, prefixed, are the flat document’s names', () => {
  it('checks clean against the flat document’s headers', () => {
    const answer = checkCheckpoint(
      documentOf(d3Of('shieldstral-3b-composite')),
      headersOf(d3Of('shieldstral-3b')),
    );
    expect(answer.errors).toEqual([]);
    expect(answer.warnings).toEqual([]);
    expect(answer.stats).toEqual({ located: 458, physical: 458, unnamed: 0 });
  });
});

describe('the warnings §4.19 asks for beside the tools’ advisories', () => {
  it('a document that locates nothing has no unlocated warning: V17 makes it absent, not partial', () => {
    // "Locations are total or absent" (V17). `deepseek-v4-pro` locates none of its 1 771 tensors,
    // which is what `artifact.run` prints as "no location: nothing to check against"; every
    // physical tensor is then named by nobody, and that is the only thing to say.
    const answer = checkCheckpoint(documentOf(d3Of('deepseek-v4-pro')), {
      'model.embed_tokens.weight': entry('bf16', [4n]),
    });
    expect(answer.errors).toEqual([]);
    expect(answer.stats.located).toBe(0);
    expect(answer.warnings.map((problem) => problem.kind)).toEqual(['unnamed']);
  });

  it('an identity with no location while others have one is a warning', () => {
    // Unreachable from a valid document — V17's totality is decided before any derivation — and
    // emitted all the same, `checkCheckpoint` being a function of a derived document and a header
    // map. The wording is the editor's: the tools have none to reproduce.
    const partial: PyRecord = {
      tensors: [
        {
          identity: 'a',
          dtype: 'bf16',
          location: { tensor: 'w' },
          shape: [{ axis: 'a', extent: 4n }],
        },
        { identity: 'b', dtype: 'bf16', shape: [{ axis: 'a', extent: 4n }] },
      ],
    };
    const answer = checked(partial, { w: entry('bf16', [4n]) });
    expect(answer.errors).toEqual([]);
    expect(answer.warnings).toEqual([
      {
        code: 'V17',
        message: "identity 'b' has no location while others have one",
        severity: 'warning',
        kind: 'unlocated',
        identity: 'b',
      },
    ]);
    expect(answer.stats).toEqual({ located: 1, physical: 1, unnamed: 0 });
  });

  it('leads with the tools’ advisories, so the parity contract is a prefix of the warnings', () => {
    const partial: PyRecord = {
      tensors: [
        {
          identity: 'a',
          dtype: 'bf16',
          location: { tensor: 'w' },
          shape: [{ axis: 'a', extent: 4n }],
        },
        { identity: 'b', dtype: 'bf16', shape: [{ axis: 'a', extent: 4n }] },
      ],
    };
    const answer = checked(partial, { w: entry('bf16', [4n]), spare: entry('f32', [1n]) });
    expect(answer.warnings.map((problem) => problem.kind)).toEqual(['unnamed', 'unlocated']);
  });
});

describe('a dtype the reader has no mapping for', () => {
  // The decision `artifact/dtypes.ts` states: the name is carried through as the file spells it,
  // the tensor is still refused — a reader cannot certify a dtype it cannot name — and the kind
  // says why. `read_headers` would have answered `f8_e4m3fnuz` here, which is no dtype of the
  // language either, so both implementations refuse; only the spelling differs.
  it('is refused in the tools’ words, with the file’s own spelling and its own kind', () => {
    const answer = checked(syntheticD3({ tensor: 'w' }, [4n]), {
      w: { dtype: 'F8_E4M3FNUZ', known: false, shape: [4n], file: 'x' },
    });
    expect(messages(answer.errors)).toEqual([
      "t: 'w' is F8_E4M3FNUZ, the document says bf16",
    ]);
    expect(answer.errors[0]?.kind).toBe('dtype-unknown');
  });
});

describe('where the tools raise, this raises', () => {
  it('a location form that is none of the four: one line, then `_names`’ KeyError', () => {
    expect(() => checked(syntheticD3({ reshape: {} }, [2n]), {})).toThrowError(PyKeyError);
    expect(() => checked(syntheticD3({ reshape: {} }, [2n]), {})).toThrowError("'concat'");
  });

  it('a slice along an axis whose logical extent is 1', () => {
    // `[i for i, d in enumerate(logical) if d != 1].index(s['dim'])`, unguarded.
    expect(() =>
      checked(syntheticD3({ slice: { tensor: 'big', axis: 'r', dim: 0n, offset: 0n, extent: 1n } }, [
        1n,
        4n,
      ]), { big: entry('bf16', [1n, 4n]) }),
    ).toThrowError(new PyValueError('0 is not in list'));
  });

  it('a `dim` past the end of the shape', () => {
    expect(() =>
      checked(syntheticD3({ stack: { axis: 'e', dim: 5n, parts: [{ tensor: 'w' }] } }, [2n]), {
        w: entry('bf16', [2n]),
      }),
    ).toThrowError(PyIndexError);
  });

  it('a derived document with no D3 at all', () => {
    expect(() => checkCheckpoint({}, {})).toThrowError(new PyKeyError("'d3'"));
  });
});

describe('the real checkpoints, when they are on this machine', () => {
  // `tests/run_artifact.py` looks under `$TENSORSPINE_MODEL_ARTIFACTS/weights`, and says so
  // instead of searching when the variable is unset. The reading is the core's — the index, the
  // eight-byte prefix, exactly the header it announces — and the I/O is this test's.
  //
  // A skip is *printed*, as the script prints one: Vitest's default reporter keeps a passing
  // test's console quiet, so the note goes to the stream directly rather than being swallowed.
  const llamaCheckpoint = checkpointDirectory('Meta-Llama-3-8B');
  const shieldstral = checkpointDirectory('Shieldstral-1.0-3B');
  for (const [name, directory] of [
    ['Meta-Llama-3-8B', llamaCheckpoint],
    ['Shieldstral-1.0-3B', shieldstral],
  ] as const) {
    if (directory === null) {
      process.stdout.write(
        `  skip ${name} (not under $TENSORSPINE_MODEL_ARTIFACTS/weights on this machine)\n`,
      );
    }
  }

  it.skipIf(llamaCheckpoint === null)('llama3-8b against Meta-Llama-3-8B on disk', () => {
    const { headers, bytes } = readCheckpoint(llamaCheckpoint as string);
    const answer = checkCheckpoint(documentOf(llama()), headers);
    expect(answer.errors).toEqual([]);
    expect(answer.warnings).toEqual([]);
    expect(answer.stats).toEqual({ located: 291, physical: 291, unnamed: 0 });
    // Four shards, 16 GB on disk: the headers alone, which is §4.19's claim made good.
    expect(bytes).toBeLessThan(64 * 1024);
    expect(new Set(Object.values(headers).map((one) => one.file)).size).toBe(4);
    expect(Object.values(headers).every((one) => one.known)).toBe(true);
  });

  it.skipIf(shieldstral === null)(
    'shieldstral-3b and its composite against Shieldstral-1.0-3B, one file without an index',
    () => {
      const { headers } = readCheckpoint(shieldstral as string);
      const flat = checkCheckpoint(documentOf(d3Of('shieldstral-3b')), headers);
      expect(flat.errors).toEqual([]);
      expect(flat.warnings).toEqual([]);
      expect(flat.stats).toEqual({ located: 458, physical: 458, unnamed: 0 });
      const composite = checkCheckpoint(documentOf(d3Of('shieldstral-3b-composite')), headers);
      expect(composite.errors).toEqual([]);
      expect(composite.stats.located).toBe(458);
    },
  );
});
