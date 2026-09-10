import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/parse.js';
import { serialize } from '../../src/json/serialize.js';
import { isJsonArray, isJsonNumber, isJsonObject, toPlain, type JsonValue } from '../../src/json/tree.js';
import { interchangeFiles, readRepositoryFile } from './repository.js';

// The criterion of the feature and of D12: a document the editor read and did not edit is
// written back byte for byte. The corpus and the reference base are the files the editor loads
// and stores, so the whole of `data/models/` and `data/primitive-library/` is the test — the
// fourteen models, the `decoder-causal-yarn` template, the base's manifest, its 36 primitive
// versions, its axes and its precision roles.

const files = interchangeFiles();

describe('every file of the corpus and of the reference base', () => {
  it('is a file this test knows how to read', () => {
    expect(files.length).toBeGreaterThan(140);
    expect(files.filter((path) => !path.endsWith('.json'))).toEqual([]);
  });

  for (const path of files) {
    it(`round-trips ${path}`, () => {
      const text = readRepositoryFile(path);
      expect(serialize(parse(text))).toBe(text);
    });
  }
});

describe('what the round-trip rests on', () => {
  it('reads the same values a plain JSON parser reads', () => {
    for (const path of files) {
      const text = readRepositoryFile(path);
      expect(toPlain(parse(text))).toEqual(JSON.parse(text) as unknown);
    }
  });

  it('finds in the corpus the lexemes the plan counted', () => {
    // Plan Appendix A: the number lexemes the serializer must reproduce, with their counts —
    // every one of them a text a plain writer would change.
    const counted = new Map<string, number>();
    const walk = (value: JsonValue): void => {
      if (isJsonNumber(value)) {
        const lexeme = value.lexeme ?? '';
        if (value.real) counted.set(lexeme, (counted.get(lexeme) ?? 0) + 1);
        return;
      }
      if (isJsonArray(value)) {
        for (const item of value) walk(item);
        return;
      }
      if (isJsonObject(value)) for (const member of value.members) walk(member.value);
    };
    for (const path of files) walk(parse(readRepositoryFile(path)));

    expect(counted.get('1e-05')).toBe(6);
    expect(counted.get('1e-06')).toBe(7);
    expect(counted.get('1e-12')).toBe(1);
    expect(counted.get('1.0')).toBe(2);
    expect(counted.get('0.25')).toBe(5);
    expect(counted.get('1.2772588722239782')).toBe(3);
    // Every real of the corpus is written the way the rule of §7 F7 states, so a lexeme that the
    // formatter would write differently would be a divergence from the corpus, not a preference.
    for (const [lexeme] of counted) expect(lexeme).not.toBe('');
  });

  it('holds the corpus to the rendering rule the formatter implements', () => {
    // Every number of every file, re-formatted from its value and its float-ness, comes back as
    // the file writes it: the corpus is already in the canonical form, so an editor that writes
    // a new number writes it as the corpus would have.
    const offenders: string[] = [];
    const walk = (value: JsonValue, path: string, file: string): void => {
      if (isJsonNumber(value)) {
        const lexeme = value.lexeme ?? '';
        const written = serialize({ kind: 'number', value: value.value, real: value.real }).trimEnd();
        if (written !== lexeme) offenders.push(`${file}${path}: ${lexeme} formats as ${written}`);
        return;
      }
      if (isJsonArray(value)) {
        value.forEach((item, position) => {
          walk(item, `${path}/${String(position)}`, file);
        });
        return;
      }
      if (isJsonObject(value)) {
        for (const member of value.members) walk(member.value, `${path}/${member.name}`, file);
      }
    };
    for (const path of files) walk(parse(readRepositoryFile(path)), '', path);
    expect(offenders).toEqual([]);
  });
});
