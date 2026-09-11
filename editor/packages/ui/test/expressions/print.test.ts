import { describe, expect, it } from 'vitest';

import { readdirSync } from 'node:fs';
import { join } from 'node:path';

import { parse, type JsonValue, type SchemaRegistry } from '@tensorspine/lang';
import { SchemaShapes } from '@tensorspine/store';

import { presentation } from '../../src/presentation/index.js';
import { printAt, type PrintContext } from '../../src/expressions/index.js';
import { readRepositoryFile, registry as repositorySchemas, repositoryRoot } from '../presentation/source.js';

/**
 * The text form of §4.13, held to artboard S7 and to every expression the corpus writes.
 *
 * What is asserted is the whole of what makes the form honest: the symbols, the forms and the
 * precedences are `presentation.json`'s, an operator with no symbol prints `name(args)`, an index
 * prints with the `$` its place binds, and nothing at all raises on an expression the corpus
 * carries.
 */

const registry: SchemaRegistry = repositorySchemas();
const context: PrintContext = {
  registry,
  shapes: new SchemaShapes(registry),
  bindings: presentation(),
};

const MODEL = 'https://tensorspine.dev/schema/2.0/model.json';
const EXPRESSION = `${MODEL}#/$defs/scalar_expression`;
const CONDITION = `${MODEL}#/$defs/condition`;

function expression(text: string): string {
  return printAt(context, EXPRESSION, parse(text));
}

function condition(text: string): string {
  return printAt(context, CONDITION, parse(text));
}

describe('the text form of an expression', () => {
  it('prints a quantity bare and an index with its prefix (S7)', () => {
    expect(expression('{"quantity": "d"}')).toBe('d');
    expect(expression('{"index": "layer"}')).toBe('$layer');
  });

  it('prints a literal as the document writes it, lexeme and all', () => {
    expect(expression('{"literal": 4096}')).toBe('4096');
    expect(expression('{"literal": 1e-05}')).toBe('1e-05');
    expect(expression('{"literal": 1.0}')).toBe('1.0');
    expect(expression('{"literal": "causal"}')).toBe('causal');
    expect(expression('{"literal": true}')).toBe('true');
  });

  it('prints S7’s own line: d div heads', () => {
    expect(expression('{"op": "floor_divide", "args": [{"quantity": "d"}, {"quantity": "heads"}]}')).toBe(
      'd div heads',
    );
  });

  it('prints the index expression llama3-8b writes for its last layer', () => {
    expect(expression('{"op": "subtract", "args": [{"quantity": "layers"}, {"literal": 1}]}')).toBe(
      'layers - 1',
    );
  });

  it('prints an operator with no symbol as name(args) — §4.13’s own fallback', () => {
    expect(expression('{"op": "min", "args": [{"quantity": "a"}, {"quantity": "b"}]}')).toBe(
      'min(a, b)',
    );
    expect(expression('{"op": "max", "args": [{"literal": 1}, {"literal": 2}]}')).toBe('max(1, 2)');
  });

  it('prints a prefix operator before its argument and a function operator around it', () => {
    expect(expression('{"op": "negate", "args": [{"quantity": "d"}]}')).toBe('-d');
    expect(expression('{"op": "absolute", "args": [{"quantity": "d"}]}')).toBe('abs(d)');
  });

  it('parenthesises where precedence asks and nowhere else', () => {
    expect(
      expression(
        '{"op": "multiply", "args": [{"op": "add", "args": [{"quantity": "a"}, {"quantity": "b"}]}, {"quantity": "c"}]}',
      ),
    ).toBe('(a + b) * c');
    expect(
      expression(
        '{"op": "add", "args": [{"op": "multiply", "args": [{"quantity": "a"}, {"quantity": "b"}]}, {"quantity": "c"}]}',
      ),
    ).toBe('a * b + c');
  });
});

describe('the text form of a condition', () => {
  it('prints S7’s own line, with no parentheses at all', () => {
    expect(
      condition(
        `{"all": [
           {"compare": {"operator": "equal",
             "left": {"op": "modulo", "args": [{"index": "layer"}, {"literal": 5}]},
             "right": {"literal": 4}}},
           {"compare": {"operator": "greater_or_equal",
             "left": {"index": "layer"}, "right": {"literal": 4}}}]}`,
      ),
    ).toBe('$layer mod 5 = 4 and $layer >= 4');
  });

  it('prints the invariant S7 draws on the unit side', () => {
    expect(
      condition(
        '{"compare": {"operator": "equal", "left": {"op": "modulo", "args": [{"quantity": "heads"}, {"quantity": "kv_heads"}]}, "right": {"literal": 0}}}',
      ),
    ).toBe('heads mod kv_heads = 0');
  });

  it('prints the guard qwen3.5-35b-a3b writes, as S2’s badge shows it', () => {
    expect(
      condition(
        '{"compare": {"operator": "equal", "left": {"op": "modulo", "args": [{"index": "layer"}, {"literal": 4}]}, "right": {"literal": 3}}}',
      ),
    ).toBe('$layer mod 4 = 3');
  });

  it('prints not and or', () => {
    expect(condition('{"not": {"boolean": true}}')).toBe('not true');
    expect(
      condition('{"any": [{"boolean": true}, {"boolean": false}]}'),
    ).toBe('true or false');
  });

  it('parenthesises an or inside an and', () => {
    expect(
      condition('{"all": [{"any": [{"boolean": true}, {"boolean": false}]}, {"boolean": true}]}'),
    ).toBe('(true or false) and true');
  });
});

describe('every expression the repository writes', () => {
  it('prints without raising, and never as the empty string', () => {
    let printed = 0;
    for (const [anchor, values] of expressionsOfCorpus()) {
      for (const value of values) {
        const text = printAt(context, anchor, value);
        expect(text.length).toBeGreaterThan(0);
        printed += 1;
      }
    }
    expect(printed).toBeGreaterThan(1000);
  });
});

/** Every `scalar_expression` and `condition` of the corpus, found by walking the documents. */
function expressionsOfCorpus(): Map<string, JsonValue[]> {
  const found = new Map<string, JsonValue[]>([
    [EXPRESSION, []],
    [CONDITION, []],
  ]);
  for (const text of corpusTexts()) {
    const tree = parse(text);
    collect(tree, found);
  }
  return found;
}

/** The members that hold an expression or a condition, as the two shapes they take. */
function collect(value: JsonValue, into: Map<string, JsonValue[]>): void {
  if (Array.isArray(value)) {
    for (const one of value as readonly JsonValue[]) collect(one, into);
    return;
  }
  if (typeof value !== 'object' || value === null || !('members' in value)) return;
  const node = value as { members: readonly { name: string; value: JsonValue }[] };
  const names = node.members.map((one) => one.name);
  if (names.length >= 1 && EXPRESSION_TAGS.some((tag) => names.includes(tag))) {
    into.get(EXPRESSION)?.push(value);
  }
  if (names.length === 1 && CONDITION_TAGS.includes(names[0] as string)) {
    into.get(CONDITION)?.push(value);
  }
  for (const one of node.members) collect(one.value, into);
}

// The tags are the schemas' own and this is a test, which is where §1 (b) admits them.
const EXPRESSION_TAGS = ['literal', 'quantity', 'index', 'op'];
const CONDITION_TAGS = ['boolean', 'not', 'all', 'any', 'compare'];

/** The corpus documents, read at the source (plan D14). */
function corpusTexts(): string[] {
  return readdirSync(join(repositoryRoot, 'data/models'))
    .filter((name) => name.endsWith('.json'))
    .map((name) => readRepositoryFile(`data/models/${name}`));
}


