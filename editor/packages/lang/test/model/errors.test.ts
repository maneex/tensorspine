import { describe, expect, it } from 'vitest';

import { PyKeyError, PyTypeError } from '../../src/expr/index.js';
import { JsonParseError } from '../../src/json/index.js';
import { ModelError, loadModel, normalise } from '../../src/model/index.js';
import { readRepositoryFile } from '../json/repository.js';
import { document } from './document.js';

// The four refusals of `tools/model.py`, and the places it raises instead of refusing.
//
// Three of the four are in `tests/rejections/models.json`, which is the wording contract (plan §7
// F1): the manifest's own `match` string is read from the repository here rather than copied, so
// a wording that moves in the tools moves this test with it.
//
// The rule a refusal carries is `validate.analyse`'s line, not `model.py`'s — it catches the error
// and reads its own text, `V12` when it holds `duplicate` or `declared both` and `V1` otherwise —
// so the port of that line belongs to feature 1.6. What is checked here is that {@link
// ModelError.kind} says the same thing without a substring test, against the codes the rejection
// suite expects.

/** The rejection suite's model cases, as the repository writes them. */
interface RejectionCase {
  document: string;
  expect: string;
  match: string;
  note?: string;
}

const CASES = (
  JSON.parse(readRepositoryFile('tests/rejections/models.json')) as { cases: RejectionCase[] }
).cases;

/** The case for one rejection document, or a failure saying the suite no longer carries it. */
function rejectionCase(name: string): RejectionCase {
  const found = CASES.find((one) => one.document === `models/${name}`);
  expect(found, `tests/rejections/models.json no longer carries models/${name}`).toBeDefined();
  return found as RejectionCase;
}

/** The refusal of a rejection document, read from the repository as the tools read it. */
function refusalOf(name: string): ModelError {
  const text = readRepositoryFile(`tests/rejections/models/${name}`);
  try {
    loadModel(text);
  } catch (error) {
    if (error instanceof ModelError) return error;
    throw error;
  }
  throw new Error(`${name} was read; the rejection suite says it must not be`);
}

/** `validate.analyse`'s mapping, written here as the test's own statement of feature 1.6's line. */
function code(error: ModelError): string {
  return error.kind === 'duplicate' || error.kind === 'collision' ? 'V12' : 'V1';
}

/** The {@link ModelError} a text raises, or a failure. */
function refuses(text: string): ModelError {
  try {
    loadModel(text);
  } catch (error) {
    if (error instanceof ModelError) return error;
    throw error;
  }
  throw new Error('the document was read; it was expected to be refused');
}

describe('a scoped rule colliding with a top-level one (§5.2 rule 7, V12)', () => {
  it('is refused in the tools’ words, whichever kind of binding it is', () => {
    const rules = {
      values:
        '{"link": {"from": {"site": "a", "port": "out"}, "to": {"site": "b", "port": "in"}}}',
      parameters: '{"link": {"members": [{"site": "a", "parameter": "w"}]}}',
      states: '{"link": {"members": [{"site": "a", "state": "kv"}]}}',
      constants: '{"link": {"constant": "m", "members": [{"site": "a", "constant": "m"}]}}',
    };
    for (const [kind, scoped] of Object.entries(rules)) {
      const top =
        '{"values": {}, "parameters": {}, "constants": {}, "states": {}}'.replace(
          `"${kind}": {}`,
          `"${kind}": {"c.link": {}}`,
        );
      const error = refuses(document({ bindings: `{"${kind}": ${scoped}}` }, top));
      expect(error.kind).toBe('collision');
      expect(error.message).toBe(
        "binding 'c.link' is declared both in composition 'c' and at the top level",
      );
    }
  });

  it('reproduces the rejection suite’s case, and its code', () => {
    const expected = rejectionCase('v1-scoped-rule-collision.json');
    const error = refusalOf('v1-scoped-rule-collision.json');
    expect(error.message).toContain(expected.match);
    expect(error.message).toBe(
      "binding 'decoder.attn.norm_in' is declared both in composition 'decoder' and at the top level",
    );
    expect(code(error)).toBe(expected.expect);
    expect(expected.expect).toBe('V12');
  });
});

describe('an endpoint selecting no site of its composition', () => {
  it('names the composition, the rule and the site', () => {
    const from = refuses(
      document({
        bindings:
          '{"values": {"link": {"from": {"site": "nowhere", "port": "out"}, ' +
          '"to": {"site": "b", "port": "in"}}}}',
      }),
    );
    expect(from.kind).toBe('site');
    expect(from.message).toBe("composition 'c', binding 'link': no site named 'nowhere'");
    const to = refuses(
      document({
        bindings:
          '{"values": {"link": {"from": {"site": "a", "port": "out"}, ' +
          '"to": {"site": "nowhere", "port": "in"}}}}',
      }),
    );
    expect(to.message).toBe("composition 'c', binding 'link': no site named 'nowhere'");
    const member = refuses(
      document({
        bindings: '{"parameters": {"a.w": {"members": [{"site": "nowhere", "parameter": "w"}]}}}',
      }),
    );
    expect(member.message).toBe("composition 'c', binding 'a.w': no site named 'nowhere'");
  });

  it('reproduces the rejection suite’s case, and its code', () => {
    const expected = rejectionCase('v1-scoped-unknown-site.json');
    const error = refusalOf('v1-scoped-unknown-site.json');
    expect(error.message).toContain(expected.match);
    expect(error.message).toBe(
      "composition 'decoder', binding 'attn.norm_in': no site named 'attention'",
    );
    expect(code(error)).toBe(expected.expect);
    expect(expected.expect).toBe('V1');
  });
});

describe('an index override naming what the composition does not range over', () => {
  it('names the index, not the site', () => {
    const error = refuses(
      document({
        bindings:
          '{"values": {"link": {"from": {"site": "a", "indices": {"j": {"literal": 0}}, ' +
          '"port": "out"}, "to": {"site": "b", "port": "in"}}}}',
      }),
    );
    expect(error.kind).toBe('index');
    expect(error.message).toBe(
      "composition 'c', binding 'link': 'j' is not an index of the composition",
    );
  });

  it('has no case in the rejection suite, and is a V1 by analyse’s reading', () => {
    // `no site named` has one (`v1-scoped-unknown-site.json`) and this one does not; both are
    // reference failures of §5.2 rule 3, and `analyse` reads both as V1.
    const error = refuses(
      document({
        bindings: '{"states": {"a.kv": {"members": [{"site": "a", "indices": {"j": ' +
          '{"literal": 0}}, "state": "kv"}]}}}',
      }),
    );
    expect(code(error)).toBe('V1');
    expect(CASES.some((one) => one.match.includes('is not an index of the composition'))).toBe(false);
  });
});

describe('a duplicate member name (V12)', () => {
  it('is the parser’s refusal, raised as the ModelError the tools raise', () => {
    const error = refuses('{"schema": "tensorspine/2.0", "model": "a", "model": "b"}');
    expect(error.kind).toBe('duplicate');
    expect(error.message).toBe("duplicate member name 'model' (V12)");
    expect(code(error)).toBe('V12');
  });

  it('reproduces the rejection suite’s case', () => {
    const expected = rejectionCase('v12-duplicate-member-name.json');
    const error = refusalOf('v12-duplicate-member-name.json');
    expect(error.message).toContain(expected.match);
    expect(error.message).toBe("duplicate member name 'model' (V12)");
  });

  it('names the innermost duplicate, as the hook fires at an object’s close', () => {
    const error = refuses('{"a": {"x": 1, "x": 2}, "b": 1, "b": 2}');
    expect(error.message).toBe("duplicate member name 'x' (V12)");
  });
});

describe('a text that is not JSON at all', () => {
  it('is not a ModelError: model.load does not catch what CPython raises there', () => {
    // `validate.analyse` catches `ModelError` alone, so a `JSONDecodeError` leaves it uncaught —
    // the hole feature 1.1 states under V12 at the schema stage, where every reading of a document
    // crosses the grammar first.
    let raised: unknown;
    try {
      loadModel('{"schema": ');
    } catch (error) {
      raised = error;
    }
    expect(raised).toBeInstanceOf(JsonParseError);
    expect(raised).not.toBeInstanceOf(ModelError);
    expect((raised as JsonParseError).message).toBe(
      'Expecting value: line 1 column 12 (char 11)',
    );
  });
});

describe('where meaning assumes grammar, it raises rather than guessing (I7)', () => {
  it('raises a KeyError for a member the tools read with []', () => {
    const missing = [
      // `model['bindings'][kind]`, read where a scoped rule of that kind exists.
      {
        text: document(
          { bindings: '{"states": {"a.kv": {"members": [{"site": "a", "state": "kv"}]}}}' },
          '{"values": {}, "parameters": {}, "constants": {}}',
        ),
        name: "'states'",
      },
      // `comp['indices']`, read by `_current` and by `_hoist`'s `for_each`.
      {
        text: document({
          indices: 'null',
          bindings: '{"values": {"link": {"from": {"site": "a", "port": "out"}, "to": {"site": "b", "port": "in"}}}}',
        }).replace('"indices": null,\n', ''),
        name: "'indices'",
      },
      // `rule['members']`, `m[slot]` and `rule[side]['port']`.
      { text: document({ bindings: '{"parameters": {"a.w": {}}}' }), name: "'members'" },
      {
        text: document({ bindings: '{"parameters": {"a.w": {"members": [{"site": "a"}]}}}' }),
        name: "'parameter'",
      },
      {
        text: document({
          bindings: '{"values": {"link": {"from": {"site": "a"}, "to": {"site": "b", "port": "in"}}}}',
        }),
        name: "'port'",
      },
    ];
    for (const { text, name } of missing) {
      expect(() => loadModel(text), name).toThrowError(PyKeyError);
      try {
        loadModel(text);
      } catch (error) {
        expect((error as PyKeyError).message).toBe(name);
      }
    }
  });

  it('raises a TypeError where Python’s own container would', () => {
    expect(() => normalise('a document')).toThrowError(PyTypeError);
    expect(() => normalise(42n)).toThrowError(PyTypeError);
    const unhashable = document({
      bindings: '{"values": {"link": {"from": {"site": ["a"], "port": "out"}, "to": {"site": "b", "port": "in"}}}}',
    });
    expect(() => loadModel(unhashable)).toThrowError(PyTypeError);
    try {
      loadModel(unhashable);
    } catch (error) {
      expect((error as PyTypeError).message).toBe("unhashable type: 'list'");
    }
  });

  it('refuses a site that is a value no instance map can be keyed by', () => {
    const error = refuses(
      document({
        bindings: '{"values": {"link": {"from": {"site": 3, "port": "out"}, "to": {"site": "b", "port": "in"}}}}',
      }),
    );
    expect(error.message).toBe("composition 'c', binding 'link': no site named '3'");
  });
});
