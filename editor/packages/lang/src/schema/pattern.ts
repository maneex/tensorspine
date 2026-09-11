/**
 * A schema `pattern` as Python's `re` reads it.
 *
 * `jsonschema` decides `pattern` and `patternProperties` with `re.search`, and Python's `$` —
 * without `re.MULTILINE` — matches at the end of the string **and just before a single trailing
 * newline**. JavaScript's `$` matches only at the end. So `"embed\n"` satisfies
 * `^[A-Za-z_][A-Za-z0-9_-]*$` for the tools and not for a reader that compiled the pattern as
 * written, and the editor would refuse a document at the schema stage that `--validate` passes —
 * a divergence neither reader of the editor could catch on its own, because Ajv and the walk
 * would agree with each other and both be wrong.
 *
 * The translation is one rule: every `$` that is neither escaped nor inside a character class
 * becomes `(?=\n?$)`, which is that position exactly — the end, or one newline before it. A `$`
 * in the middle of a pattern gets it too, because Python's does not care where it stands;
 * `\$` and `[$]` are the literal character and are left as they are. Nothing else is rewritten:
 * on everything the five schemas' fourteen patterns use, the two engines already agree.
 *
 * The class reading is the simple one — `[` opens and the next unescaped `]` closes — which is
 * what every pattern of the schemas is written in. A `]` as a class's first character is literal
 * to Python and empty-then-literal to JavaScript, and that disagreement is the regex dialects'
 * own, not this translation's.
 */
export function pythonPattern(pattern: string): string {
  let out = '';
  let inClass = false;
  for (let index = 0; index < pattern.length; index += 1) {
    const character = pattern[index] as string;
    if (character === '\\') {
      out += character + (pattern[index + 1] ?? '');
      index += 1;
      continue;
    }
    if (inClass) {
      out += character;
      if (character === ']') inClass = false;
      continue;
    }
    if (character === '[') {
      inClass = true;
      out += character;
      continue;
    }
    out += character === '$' ? '(?=\\n?$)' : character;
  }
  return out;
}

/** A schema pattern compiled the way both readers of the core compile it. */
export function pythonRegExp(pattern: string, flags = 'u'): RegExp {
  return new RegExp(pythonPattern(pattern), flags);
}
