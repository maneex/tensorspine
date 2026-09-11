/**
 * The safetensors dtype vocabulary, mapped onto the language's.
 *
 * This is the one table of the core whose *keys* are not a schema's vocabulary. The plan says so
 * in as many words (§1, "what the rule does not cover"): "The safetensors header format is the
 * file format's; its dtype vocabulary is a table in the core, mapped onto the schema's `dtype`
 * enum and audited against it like every other table." So the keys are the file format's names,
 * the values are values of `model.json#/$defs/dtype`, and
 * `editor/tests/audit/semantic-tables.test.ts` holds both halves to account: every value is a
 * dtype the language declares, every pair `tools/artifact.py` carries is here with the same
 * value, and every dtype of the enumeration that has **no** key is named in the audit with the
 * reason the format cannot spell it.
 *
 * ## What `tools/artifact.py` does, and what this does instead
 *
 * `read_headers` writes `DTYPES.get(entry['dtype'], entry['dtype'].lower())`. The dictionary maps
 * twelve names; the fallback lower-cases anything else. Feature 0.5 measured what that costs: the
 * vocabulary `@huggingface/hub` recognises is twenty-five names, and the fallback turns `U16` into
 * `u16` and `F8_E4M3FNUZ` into `f8_e4m3fnuz` — **neither of which is a value of the schema's
 * `dtype`** — while `u4`, `i4` and `f8e4m3fn` have no spelling in the table at all. V17 compares
 * that name with D3's, so an unmapped dtype reads today as a dtype *mismatch*, and the line it
 * prints claims the checkpoint holds a `u16` where the document wants a `bf16`.
 *
 * Feature 0.5 recorded the choice as feature 1.9's, between a refusal, a warning, and a
 * pass-through marked unknown. **The choice taken is the pass-through, marked** — and the
 * fallback is dropped, not reproduced:
 *
 * - *Refusing* is out. A dtype the table has not is a fact about the file, not an error of the
 *   author's, and §4.19's Weights panel must still list the tensor for the author to place; the
 *   plan's Q5 ("no gesture refused for a semantic reason") applies to opening a checkpoint as
 *   much as to drawing an edge.
 * - *Lower-casing* is out, because it is what makes the table unauditable. `name.lower()`'s range
 *   is every string, so "mapped onto the schema's `dtype` enum and audited against it" cannot be
 *   said of it at all — and because the name it invents (`u16`, `f8_e4m3fnuz`) *looks* like a
 *   dtype of the language while being none, which is the worst of the three readings for a reader
 *   of the Problems panel.
 * - So a name the table has not is carried through **exactly as the file spells it**, with
 *   {@link HeaderDtype.known} false. `checkCheckpoint` then still refuses the tensor — the reader
 *   cannot certify a dtype it cannot name — but the line it prints names the file's own spelling
 *   (`'w' is F8_E4M3FNUZ, the document says bf16`), which is true, and the problem it emits
 *   carries the kind `dtype-unknown` rather than `dtype`, so the panel can say that the reader has
 *   no mapping rather than that the checkpoint disagrees.
 *
 * The divergence from the tools is therefore confined to names outside the language's vocabulary,
 * where the tools' own answer could never have matched a D3 dtype either: for every one of the
 * thirteen names the table carries the two implementations answer the same string, and for every
 * other name they both refuse the tensor, differing only in the spelling they print. No document,
 * base or checkpoint of the repository reaches the difference — the corpus's checkpoints are
 * `bf16`, `f32`, `f16`, `i64` and `u8` throughout.
 *
 * ## The thirteenth key
 *
 * Twelve pairs are `tools/artifact.py`'s own, character for character. The thirteenth is
 * `FP4 → fp4`, which is **not** an invention: `'FP4'.lower()` is `'fp4'`, a value of the enum, so
 * the tools' fallback already answers exactly this and writing it down only closes the table.
 * Nothing else is added. In particular `F4` is left unknown although the language has `fp4`, and
 * `F8_E4M3FNUZ`/`F8_E5M2FNUZ` are left unknown although the language has `f8e4m3fn`: inventing
 * either pair would change an answer the tools give, and which encoding a lab means by which name
 * is a question for the language, not for a port (a finding, recorded in the ledger).
 */

/** The file format's name for a dtype, as a safetensors header writes it: `BF16`, `F8_E4M3`, … */
export type SafetensorsDtype = string;

/**
 * The safetensors names this core maps onto the language's `dtype` enumeration.
 *
 * Twelve pairs are `tools/artifact.py`'s `DTYPES`, and the audit requires them to stay equal to
 * it. The thirteenth, `FP4`, is the one name whose lower-case form is already a dtype of the
 * language, so writing it down closes the table without changing an answer.
 */
export const SAFETENSORS_DTYPES: Readonly<Record<SafetensorsDtype, string>> = {
  BF16: 'bf16',
  BOOL: 'bool',
  F16: 'f16',
  F32: 'f32',
  F64: 'f64',
  F8_E4M3: 'f8e4m3',
  F8_E5M2: 'f8e5m2',
  FP4: 'fp4',
  I16: 'i16',
  I32: 'i32',
  I64: 'i64',
  I8: 'i8',
  U8: 'u8',
};

/** A dtype read from a header: the language's name, or the file's own when there is none. */
export interface HeaderDtype {
  /** The language's `dtype` when {@link known}; the header's own spelling when not. */
  readonly dtype: string;
  /** Whether {@link dtype} is a value of the schema's `dtype` enumeration. */
  readonly known: boolean;
}

/**
 * One name of the file format's vocabulary, read as a dtype.
 *
 * The language's name when the table carries it, and otherwise the name as the file spells it,
 * marked unknown — never lower-cased into something that looks like a dtype and is not.
 */
export function dtypeOf(name: SafetensorsDtype): HeaderDtype {
  const mapped = Object.hasOwn(SAFETENSORS_DTYPES, name)
    ? SAFETENSORS_DTYPES[name]
    : undefined;
  return mapped === undefined ? { dtype: name, known: false } : { dtype: mapped, known: true };
}
