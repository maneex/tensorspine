import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { alternativeLabel, loadSchemas, parse } from '@tensorspine/lang';
import { describe, expect, it } from 'vitest';

import { alternationAt } from '../../src/forms/alternatives.js';
import { auditPresentation } from '../../src/presentation/audit.js';
import { formLines, noteLines } from '../../src/forms/render.js';
import { formContext, formOf } from '../../src/forms/walk.js';
import { readPresentation } from '../../src/presentation/load.js';
import { presentation } from '../../src/presentation/index.js';
import {
  CHOOSER,
  JSON_EDITOR,
  MAP,
  SECTION,
  TEXT,
} from '../../src/forms/widget.js';
import type { FormRow } from '../../src/forms/types.js';
import { context, MODEL, UNIT, repositoryRoot } from './source.js';

// The four cases plan §2.3 names, and the one thing every one of them is about: the walker reads
// the schema and never remembers it.
//
//   * `location` — a four-way `oneOf` whose alternatives are told apart by the key each requires,
//     and which contains itself (`stack.part` is a `location`);
//   * `quantity_definition` — `domain` is declared always and required only when the source is
//     external, which the schema says with `if`/`then` and nothing else;
//   * `argument_value` — a union of a union, recursive through `record`, which is where plan
//     §4.12's source modes come from;
//   * a map — `additionalProperties` with `propertyNames`, whose names are an `identifier` and
//     whose entries are whatever the document wrote.
//
// And the fifth: a construct the walker has no reading of renders as the generic JSON row and is
// listed, "never fails" (§1).

const row = (rows: readonly FormRow[], path: string): FormRow => {
  const found = rows.find((one) => one.path === path);
  if (found === undefined) throw new Error(`no row at '${path}' among ${String(rows.length)}`);
  return found;
};

const tags = (rows: readonly FormRow[], path: string): readonly string[] =>
  (row(rows, path).modes ?? []).map((mode) => mode.tag);

describe('a `oneOf` is a chooser labelled by what tells its alternatives apart', () => {
  it('offers `location`’s four forms by the key each requires', () => {
    const form = formOf(context(), { anchor: `${MODEL}#/$defs/location`, label: 'location' });
    expect(form.rows).toHaveLength(1);
    const chooser = row(form.rows, '');
    expect(chooser.widget).toBe(CHOOSER);
    // Not written here: the four names are the `required` keys of the four alternatives, read
    // from the schema. A fifth form added to the grammar appears without a line changing.
    expect(chooser.modes?.map((mode) => mode.tag)).toEqual(['tensor', 'stack', 'concat', 'slice']);
    expect(chooser.modes?.every((mode) => mode.widget === SECTION)).toBe(true);
    expect(chooser.mode).toBeUndefined();
    expect(form.notes).toEqual([]);
  });

  it('descends into the form the value is, and into no other', () => {
    const value = parse(
      '{"stack": {"axis": "head", "part": {"tensor": ["layers.", {"index": "layer"}, ".w"]}}}',
    );
    const form = formOf(context(), { anchor: `${MODEL}#/$defs/location`, value, label: 'location' });
    expect(formLines(form)).toEqual([
      'chooser location  {tensor:section | *stack:section | concat:section | slice:section}',
      '  section stack  required',
      '    text axis  required  is head  matches ^[A-Za-z_][A-Za-z0-9_-]*$',
      '    chooser part  required  {*tensor:section | stack:section | concat:section | slice:section}',
      '      token-list tensor  required  items>=1',
    ]);
    // A `location` inside a `location`: the recursion is the value's and stops where it stops.
    expect(row(form.rows, '/stack/part').mode).toBe('tensor');
    // The physical name is the token editor's (`presentation.json`), so the walker does not
    // descend into the three item forms — the widget owns its subtree.
    expect(row(form.rows, '/stack/part/tensor').widget).toBe('token-list');
  });

  it('labels by a member’s `const` when the required keys do not tell them apart', () => {
    // `quantity_type`'s five alternatives all require `kind` and nothing else; what separates
    // them is the constant each one's `kind` is fixed to. Plan §1 says "the alternative's
    // discriminating required key **or `const`**", and this is the `const` half.
    expect(tags(formOf(context(), { anchor: `${MODEL}#/$defs/quantity_type` }).rows, '')).toEqual([
      'cardinality',
      'real',
      'physical',
      'enum',
      'boolean',
    ]);
    expect(tags(formOf(context(), { anchor: `${MODEL}#/$defs/quantity_source` }).rows, '')).toEqual([
      'literal',
      'external',
      'derived',
    ]);
  });
});

describe('a `required` that appears under a condition', () => {
  const anchor = `${MODEL}#/$defs/quantity_definition`;

  it('is a condition on the row, not a requirement of the place', () => {
    const form = formOf(context(), { anchor, label: 'quantity_definition' });
    expect(row(form.rows, '/type').required).toBe(true);
    expect(row(form.rows, '/source').required).toBe(true);
    const domain = row(form.rows, '/domain');
    expect(domain.required).toBe(false);
    expect(domain.conditions).toEqual([
      {
        branch: `${anchor}/then`,
        test: `${anchor}/if`,
        holds: null,
        requires: true,
        declares: false,
      },
    ]);
  });

  it('says whether the condition holds, once there is a value to read it against', () => {
    const external = formOf(context(), {
      anchor,
      value: parse('{"type": {"kind": "cardinality"}, "source": {"kind": "external"}}'),
    });
    expect(row(external.rows, '/domain').conditions?.[0]?.holds).toBe(true);
    expect(row(external.rows, '/domain').present).toBe(false);

    const literal = formOf(context(), {
      anchor,
      value: parse('{"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 32}}'),
    });
    expect(row(literal.rows, '/domain').conditions?.[0]?.holds).toBe(false);
  });

  it('reads what a branch adds and not what one forbids, and says so', () => {
    // `evolution_rule` has two branches on the same member: a `window` evolution *requires*
    // `span`, and an `append` or `fixed` one writes `not: {required: ["span"]}` — it forbids it.
    // The walker reads the first and not the second: the row for `span` carries the requirement
    // and its condition, and the prohibition is Ajv's to report on the row when a document
    // writes one anyway (D5, Q5 — the gesture is never refused, the verdict is shown). The one
    // `not` of the four schemas is that one.
    const anchor = `${UNIT}#/$defs/evolution_rule`;
    const span = row(formOf(context(), { anchor }).rows, '/span');
    expect(span.conditions?.map((one) => `${one.test} ${String(one.requires)}`)).toEqual([
      `${anchor}/allOf/0/if true`,
    ]);
    const forbidden = formOf(context(), {
      anchor,
      value: parse('{"evolution": "append", "access": "logical_position", "span": {"literal": 1}}'),
    });
    expect(row(forbidden.rows, '/span').present).toBe(true);
    expect(row(forbidden.rows, '/span').conditions?.[0]?.holds).toBe(false);
  });

  it('shows a member only a branch declares, with the branch that declares it', () => {
    // `argument_type` declares `kind` always and `unit` only when `kind` is `physical`; the
    // schema says so in an `allOf` of `if`/`then`, and the row carries the branch.
    const form = formOf(context(), { anchor: `${UNIT}#/$defs/argument_type`, label: 'argument_type' });
    expect(row(form.rows, '/kind').required).toBe(true);
    const unit = row(form.rows, '/unit');
    expect(unit.required).toBe(false);
    expect(unit.conditions?.[0]?.requires).toBe(true);
    expect(unit.options?.map((one) => one.label)).toEqual([
      'bytes',
      'elements',
      'tokens',
      'seconds',
      'operations',
    ]);
  });
});

describe('`argument_value`: a union of a union, and the source modes of §4.12', () => {
  const anchor = `${MODEL}#/$defs/argument_value`;

  it('flattens to the modes the argument sheet draws', () => {
    const form = formOf(context(), { anchor, label: 'argument_value' });
    expect(tags(form.rows, '')).toEqual(['literal', 'quantity', 'index', 'expression', 'record']);
    const modes = row(form.rows, '').modes ?? [];
    // Three of §4.12's four modes are written as one scalar member and are edited on the row
    // itself; the fourth is the expression editor's, and `record` nests.
    expect(modes.filter((one) => one.inline).map((one) => one.member)).toEqual([
      'literal',
      'quantity',
      'index',
    ]);
    // The quantity and index modes name something; the binding says which, the schema cannot.
    expect(modes.map((one) => one.referent ?? '')).toEqual(['', 'quantity', 'index', '', '']);
    // The expression mode covers the four shaped forms the vocabulary cannot tell apart.
    expect(modes.find((one) => one.tag === 'expression')?.anchors).toHaveLength(4);
    expect(form.notes).toEqual([]);
  });

  it('says which mode a written value is, by the grammar’s verdict and not by its keys', () => {
    const mode = (text: string): string | undefined =>
      formOf(context(), { anchor, value: parse(text) }).rows[0]?.mode;
    expect(mode('{"literal": 32}')).toBe('literal');
    expect(mode('{"quantity": "d"}')).toBe('quantity');
    expect(mode('{"index": "layer"}')).toBe('index');
    // `op` and `args` are required by three alternatives at once; only the operator each admits
    // separates them, which is why the mode is Ajv's answer and not a tag match.
    expect(mode('{"op": "negate", "args": [{"literal": 1}]}')).toBe('expression');
    expect(mode('{"op": "add", "args": [{"literal": 1}, {"quantity": "d"}]}')).toBe('expression');
    expect(mode('{"record": {}}')).toBe('record');
    // Nothing is written: no mode at all, which is what an absent value is.
    expect(formOf(context(), { anchor }).rows[0]?.mode).toBeUndefined();
  });

  it('shows the mode a refused value resembles, so the rows that repair it are drawn', () => {
    // Feature 2.13's own defect, found by its own case: a chooser writes the **blank** of the
    // alternative it is set to, and a blank identifier is no `stack` to Ajv — so a form that drew
    // only what Ajv accepts would hide the very rows that repair what it had just written.
    //
    // Which alternative a value *is* stays Ajv's answer (the three operation forms above); which
    // alternative's rows are *drawn* is that answer, or — where there is none — the alternative
    // whose discriminating keys the value carries, with the grammar's refusal on the row that
    // breaks it (§4.17, and Q5: nothing is refused, everything is reported).
    const form = formOf(context(), { anchor, value: parse('{"quantity": 4}') });
    expect(form.rows[0]?.mode).toBe('quantity');
    expect(form.rows[0]?.written).toBe(4);
    // And a value that resembles nothing at all still has no mode.
    expect(formOf(context(), { anchor, value: parse('{"nothing": 1}') }).rows[0]?.mode).toBeUndefined();
  });

  it('recurses through a record, one level of indent per level of the value', () => {
    const value = parse(
      '{"record": {"span": {"literal": 4096}, "beta": {"record": {"low": {"quantity": "d"}}}}}',
    );
    const form = formOf(context(), { anchor, value });
    expect(formLines(form)).toEqual([
      'chooser argument_value  {literal:scalar | quantity:text→quantity | index:text→index | expression:expression | *record:section}',
      '  map record  required  keys #/$defs/identifier',
      '    chooser span  {*literal:scalar | quantity:text→quantity | index:text→index | expression:expression | record:section}  is 4096',
      '    chooser beta  {literal:scalar | quantity:text→quantity | index:text→index | expression:expression | *record:section}',
      '      map record  required  keys #/$defs/identifier',
      '        chooser low  {literal:scalar | *quantity:text→quantity | index:text→index | expression:expression | record:section}  is d  names a quantity',
    ]);
    expect(row(form.rows, '/record/beta/record/low').depth).toBe(4);
  });
});

describe('a map: `additionalProperties` with `propertyNames`', () => {
  it('says what its names must be, and takes its entries from the value', () => {
    const form = formOf(context(), {
      anchor: `${MODEL}#/$defs/argument_map`,
      value: parse('{"heads": {"literal": 32}, "kv_heads": {"quantity": "kv"}}'),
      label: 'arguments',
    });
    const map = row(form.rows, '');
    expect(map.widget).toBe(MAP);
    expect(map.keys).toEqual({
      anchor: `${MODEL}#/$defs/identifier`,
      pattern: '^[A-Za-z_][A-Za-z0-9_-]*$',
    });
    expect(form.rows.map((one) => one.label)).toEqual(['arguments', 'heads', 'kv_heads']);
    expect(row(form.rows, '/kv_heads').mode).toBe('quantity');
  });

  it('says the fewest members a map may have, where the schema states one', () => {
    const form = formOf(context(), {
      anchor: `${MODEL}#/$defs/composition_definition/properties/instances`,
      label: 'instances',
    });
    expect(row(form.rows, '').bounds).toEqual({ minProperties: 1 });
    expect(row(form.rows, '').keys?.anchor).toBe(`${MODEL}#/$defs/identifier`);
  });
});

describe('an unknown construct', () => {
  const identity = 'https://example.test/schema/probe.json';
  const probe = loadSchemas(
    [
      {
        path: 'probe.json',
        text: JSON.stringify({
          $schema: 'https://json-schema.org/draft/2020-12/schema',
          $id: identity,
          type: 'object',
          additionalProperties: false,
          required: ['known'],
          properties: {
            known: { type: 'string' },
            opaque: { $comment: 'a place the schema says nothing about' },
            untagged: { oneOf: [{ $comment: 'one' }, { $comment: 'the other' }] },
            shared: {
              oneOf: [
                { type: 'object', required: ['a'], properties: { a: { type: 'string' } } },
                { type: 'object', required: ['a'], properties: { a: { type: 'integer' } } },
              ],
            },
          },
        }),
      },
    ],
    { origin: 'probe' },
  );
  const shaped = formContext(probe, presentation());
  const form = formOf(shaped, { anchor: `${identity}#`, label: 'probe' });

  it('renders the generic JSON row where the schema states nothing', () => {
    expect(row(form.rows, '/opaque').widget).toBe(JSON_EDITOR);
    expect(row(form.rows, '/opaque').generic).toBe('unreadable');
  });

  it('renders a chooser it cannot label, and says which half it cannot do', () => {
    expect(row(form.rows, '/untagged').generic).toBe('untagged');
    expect(row(form.rows, '/shared').generic).toBe('undiscriminated');
    expect(tags(form.rows, '/shared')).toEqual(['a', 'a']);
  });

  it('lists every one of them, once per construct', () => {
    expect(noteLines(form)).toEqual([
      'form: /opaque: the schema states nothing here, so the place gets the generic JSON row',
      'form: /untagged: 2 of 2 alternatives carry no required key and no type, so a chooser has nothing to label them by',
      'form: /shared: its alternatives share their required keys and no member fixes a constant, so no chooser tells them apart',
    ]);
  });

  it('renders a member the schema does not declare, rather than dropping it', () => {
    const written = formOf(shaped, {
      anchor: `${identity}#`,
      value: parse('{"known": "a", "invented": {"anything": 1}}'),
      label: 'probe',
    });
    expect(row(written.rows, '/invented')).toMatchObject({
      widget: JSON_EDITOR,
      generic: 'undeclared',
      present: true,
      anchors: [],
    });
    expect(noteLines(written)).toContain(
      "form: /invented: 'invented' is written here and the schema declares no such member",
    );
  });

  it('is what the four schemas of the repository never reach today', () => {
    // The snapshot records it too (`test/snapshots/forms.test.ts`), and this is the sentence that
    // says what the empty list means: every construct the grammar writes has a reading, so the
    // generic path is exercised by a schema written for it and by nothing of the repository.
    expect(formOf(context(), { role: 'model' }).notes).toEqual([]);
    expect(formOf(context(), { role: 'primitive-library-unit' }).notes).toEqual([]);
  });
});

describe('what a presentation binding is doing, shown by taking one away', () => {
  it('leaves a chooser that cannot tell three operation forms apart', () => {
    // `unary`, `binary` and `nary` all require `op` and `args` (feature 1.1's finding); what
    // makes them one `expression` mode is the editor bound to `scalar_expression`. Without it the
    // walker reports exactly the gap §1's "unknown constructs get the generic widget" is for.
    const anchor = `${MODEL}#/$defs/scalar_expression`;
    const source = JSON.parse(
      readFileSync(join(repositoryRoot, 'editor/packages/ui/src/presentation.json'), 'utf8'),
    ) as Record<string, unknown>;
    const rest = Object.fromEntries(Object.entries(source).filter(([key]) => key !== anchor));
    const without = formContext(
      formContextRegistry(),
      readPresentation(rest),
    );
    const form = formOf(without, { anchor, label: 'scalar_expression' });
    expect(row(form.rows, '').generic).toBe('undiscriminated');
    expect(tags(form.rows, '')).toEqual([
      'literal',
      'quantity',
      'index',
      'op | args',
      'op | args',
      'op | args',
      'if | then | else',
    ]);
    // And with it, the four modes of §4.12.
    expect(tags(formOf(context(), { anchor }).rows, '')).toEqual([
      'literal',
      'quantity',
      'index',
      'expression',
    ]);
  });
});

/** The repository's registry, as the suite's shared context holds it. */
function formContextRegistry(): Parameters<typeof formContext>[0] {
  return context().registry;
}

describe('a problem the core raised lands on the row at its pointer', () => {
  it('is the core’s own pointer, and its own words', () => {
    // The core emits `{code, message, path}` natively (plan §3), the store keys a place the same
    // way, and a row is keyed by that pointer — so binding a problem to a row is a lookup and not
    // a second addressing scheme. Nothing of the schema side is needed for it, which is why the
    // schema-side anchor feature 1.1 left open stayed open.
    const text = readFileSync(join(repositoryRoot, 'data', 'models', 'llama3-8b.json'), 'utf8');
    const broken = parse(text.replace('"tensorspine/2.0"', '"tensorspine/9.9"'));
    const problems = context().registry.structural(broken, 'model');
    expect(problems.map((one) => one.path)).toEqual(['/schema']);
    const form = formOf(context(), { role: 'model', value: broken, problems, label: 'llama3-8b' });
    expect(row(form.rows, '/schema').error).toBe(problems[0]?.message);
    expect(form.rows.filter((one) => one.error !== undefined)).toHaveLength(1);
  });

  it('leaves a problem whose place the form has no row for to the Problems panel', () => {
    const form = formOf(context(), {
      anchor: `${MODEL}#/$defs/quantity_definition`,
      problems: [{ path: '/nowhere/at/all', message: 'a place this form has no row for' }],
    });
    expect(form.rows.filter((one) => one.error !== undefined)).toEqual([]);
  });
});

describe('what a row calls generic agrees with what the presentation audit lists', () => {
  it('marks a mode bound exactly where a binding names its alternative', () => {
    // Plan §1 (a)'s second half already lists "every enum value and `oneOf` member of the loaded
    // schemas that no binding names" (feature 2.2, snapshotted). A chooser must not invent a
    // second list: a mode it calls unbound is one that list carries, and one it calls bound is one
    // that list does not. Both directions, over every union of every loaded schema.
    const audit = auditPresentation(context().registry, presentation());
    const listed = new Set(
      audit.generic.filter((one) => one.kind === 'alternative').map((one) => one.anchor),
    );
    const vocabulary = context().registry.vocabulary();
    let checked = 0;
    for (const union of vocabulary.unions) {
      const alternation = alternationAt(vocabulary, context().shapes, union.pointer);
      if (alternation === undefined) continue;
      for (const mode of formOf(context(), { anchor: union.pointer }).rows[0]?.modes ?? []) {
        for (const anchor of mode.anchors) {
          const alternative = alternation.alternatives.find((one) => one.anchor === anchor);
          if (alternative === undefined) continue;
          expect(mode.unbound.includes(anchor), `${union.pointer} ${alternative.place}`).toBe(
            listed.has(alternative.place),
          );
          checked += 1;
        }
      }
    }
    expect(checked).toBeGreaterThan(80);
  });

  it('labels five unions more finely than that list does, which is the `const` half of §1', () => {
    // The two lists agree about *which* alternatives render generically; the name can differ, and
    // where it does it differs by being more specific. The audit reads the vocabulary, which sees
    // an alternative's own `const` and not a member's; the chooser reads the member's, which is
    // what plan §1 asks for ("the alternative's discriminating required key **or `const`**"). The
    // five below are every place in the loaded schemas where that changes anything — and the
    // model schema's own selectors are among them, which is worth the rule on its own.
    const vocabulary = context().registry.vocabulary();
    const finer: string[] = [];
    for (const union of vocabulary.unions) {
      const alternation = alternationAt(vocabulary, context().shapes, union.pointer);
      if (alternation?.discriminator == null) continue;
      const audited = union.alternatives.map((one) => alternativeLabel(one)).join(', ');
      const chooser = alternation.alternatives.map((one) => one.label).join(', ');
      if (audited !== chooser) {
        finer.push(`${union.pointer.replace(/^.*#/, '#')}: ${audited} -> ${chooser}`);
      }
    }
    expect(finer).toEqual([
      '#/$defs/argument_domain: kind, values -> interval, set',
      '#/$defs/quantity_type: kind, kind, unit, values, kind -> cardinality, real, physical, enum, boolean',
      '#/$defs/quantity_domain: kind, values -> interval, set',
      '#/$defs/instance_selector: kind | instance, composition | indices -> root, generated',
      '#/$defs/quantity_source: value, kind, expression -> literal, external, derived',
    ]);
  });
});

describe('every corpus document renders', () => {
  /** The fourteen models and the template, by the path each is read at. */
  function corpus(): string[] {
    const root = join(repositoryRoot, 'data', 'models');
    const found: string[] = [];
    const walk = (directory: string, prefix: string): void => {
      for (const entry of readdirSync(directory, { withFileTypes: true }).sort((left, right) =>
        left.name < right.name ? -1 : 1,
      )) {
        if (entry.isDirectory()) walk(join(directory, entry.name), `${prefix}${entry.name}/`);
        else if (entry.name.endsWith('.json')) found.push(`${prefix}${entry.name}`);
      }
    };
    walk(root, '');
    return found;
  }

  const documents = corpus();

  it('walks all fifteen with nothing the walker has no reading for', () => {
    expect(documents).toHaveLength(15);
    for (const name of documents) {
      const text = readFileSync(join(repositoryRoot, 'data', 'models', name), 'utf8');
      const form = formOf(context(), { role: 'model', value: parse(text), label: name });
      expect(noteLines(form), name).toEqual([]);
      expect(form.rows.filter((one) => one.generic !== undefined), name).toEqual([]);
      expect(form.rows.length, name).toBeGreaterThan(100);
    }
  });

  it('renders every argument of every document in one of §4.12’s modes', () => {
    // D7's own catching rule: "the corpus test (every argument of every corpus document renders
    // in some mode)". A row whose place is an `argument_value` and whose value is written must
    // say which alternative it is; one that said nothing would be a value the sheet could not
    // show, and there are none.
    const modes = new Set<string>();
    for (const name of documents) {
      const text = readFileSync(join(repositoryRoot, 'data', 'models', name), 'utf8');
      const form = formOf(context(), { role: 'model', value: parse(text), label: name });
      for (const one of form.rows) {
        if (!one.anchors.includes(`${MODEL}#/$defs/argument_value`)) continue;
        expect(one.present, `${name} ${one.path}`).toBe(true);
        expect(one.mode, `${name} ${one.path}`).toBeDefined();
        modes.add(one.mode ?? '');
      }
    }
    expect([...modes].sort()).toEqual(['expression', 'index', 'literal', 'quantity', 'record']);
  });

  it('reads a name row as the definition the schema binds it to', () => {
    const text = readFileSync(join(repositoryRoot, 'data', 'models', 'llama3-8b.json'), 'utf8');
    const form = formOf(context(), { role: 'model', value: parse(text), label: 'llama3-8b' });
    const primitive = row(form.rows, '/instances/embed/primitive/name');
    expect(primitive.widget).toBe(TEXT);
    expect(primitive.written).toBe('embed');
    expect(row(form.rows, '/schema').constant).toBe('tensorspine/2.0');
  });
});
