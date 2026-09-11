/**
 * The interface's strings — plan §4.21's i18n bullet.
 *
 * > every GUI string in one dictionary (English first; the Alcyone convention — English source
 * > string as the key). Schema and library text is shown as written (English).
 *
 * So the key **is** the English sentence, and {@link text} answers the key itself until a
 * dictionary carries a translation for it. Three consequences worth writing down, because they
 * are what make the convention worth having:
 *
 *  - a string is written where it is shown, in English, and reads as English in the source;
 *  - nothing has to be renamed to translate it — a locale is a map from those sentences;
 *  - a string that never passes through {@link text} is invisible to a translator, which is the
 *    one mistake this file exists to make findable ({@link keysUsed} is what a suite reads).
 *
 * What is **not** here: anything the schemas, the library units or the core say. A label read from
 * a `title`, a tooltip read from a `description`, a message written by `validate` and a symbol
 * bound in `presentation.json` are shown as they stand (§1, §4.21) — translating them would be
 * translating the language.
 */

/** One locale's translations, keyed by the English source string. */
export type Dictionary = Readonly<Record<string, string>>;

/** The locale in force and the dictionary behind it. */
interface Locale {
  readonly tag: string;
  readonly entries: Dictionary;
}

/** English needs no dictionary: every key is already its own translation. */
const ENGLISH: Locale = { tag: 'en', entries: {} };

let locale: Locale = ENGLISH;

const used = new Set<string>();

/**
 * The string to show for an English source string.
 *
 * A key the dictionary does not carry answers itself — an untranslated interface is an English
 * one, never a blank one or a bracketed identifier.
 */
export function text(key: string): string {
  used.add(key);
  return locale.entries[key] ?? key;
}

/**
 * A sentence with one value in it, for the few places a label is not a constant.
 *
 * The key carries `{}` where the value goes, so a translator moves it: `text` is asked for the
 * key first, so a dictionary can reorder the sentence around it.
 */
export function textWith(key: string, value: string): string {
  const pattern = text(key);
  return pattern.includes('{}') ? pattern.replace('{}', value) : `${pattern} ${value}`;
}

/** Install a locale. `en` (or an empty dictionary) restores the English source strings. */
export function setLocale(tag: string, entries: Dictionary = {}): void {
  locale = tag === ENGLISH.tag ? ENGLISH : { tag, entries };
}

/** The locale in force. */
export function currentLocale(): string {
  return locale.tag;
}

/**
 * Every key {@link text} has been asked for since the module loaded.
 *
 * For the suites: rendering the shell and reading this is how a string that never reached the
 * dictionary is found, and it is what a translation file would be generated from.
 */
export function keysUsed(): string[] {
  return [...used].sort();
}

/** Forget what was asked for — a suite's, between cases. */
export function forgetKeysUsed(): void {
  used.clear();
}
