#!/usr/bin/env node
/**
 * Vendor the specification's anchors — plan §1, feature 2.8 (§4.17's "a problem's code links to
 * its anchor in the specification").
 *
 * > Help text is the schema's. Tooltips are `description`s (schemas, declarations,
 * > `value_descriptions`); links to the specification are generated from the anchors
 * > `docs/SPECIFICATION.md` carries (a build step extracts `V1`–`V20`, `O`- and `I`-identifiers to
 * > their anchors).
 *
 * This is that build step. It reads the specification and writes
 * `packages/ui/src/shell/anchors.ts` — the same arrangement `scripts/logo.ts` uses for the
 * wordmark, and for the same reason: the interface needs the fact synchronously, in every
 * deployment, including the one CI builds against a stub with no network; and
 * `tests/audit/anchors.test.ts` reads the specification again and requires the committed module to
 * equal what this would write, so a rule renumbered or a section re-anchored fails the audit
 * rather than shipping as a dead link.
 *
 * **What an identifier is, here.** A row of a table whose first cell is the identifier in bold and
 * nothing else — `| **V1** | Every quantity, … |`. That is how §6, §9.1, §9.2 and Appendix A state
 * them, and it is what distinguishes a *statement* from the citations the prose is full of ("a
 * base that does not exist is a rejection (V1)"). An identifier stated twice, or stated under a
 * section whose anchor is ambiguous, is refused rather than guessed at.
 *
 * **What it carries.** The section's anchor and the section's heading, and not the requirement's
 * own text: the text belongs to the specification, one click away, and a copy of it in the
 * interface's source would be a second place to keep in step for a tooltip. The heading is what a
 * link's title says — where the rule is stated.
 *
 * Run it with `pnpm anchors`, from `editor/`.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** `editor/`, this script's own workspace. */
export const editorRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/** The repository the editor is part of (D14): the specification is read from it. */
export const repositoryRoot = resolve(editorRoot, '..');

/** Where the identifiers are stated. */
export const SPECIFICATION = 'docs/SPECIFICATION.md';

/** Where the map lives. */
export const MODULE = 'packages/ui/src/shell/anchors.ts';

/** One identifier, and the section that states it. */
export interface Anchor {
  /** The section's anchor, as `docs/SPECIFICATION.md` writes it and the built page carries it. */
  readonly anchor: string;
  /** The section's heading, word for word. */
  readonly section: string;
}

/** A table row stating an identifier: the identifier in bold, alone in the first cell. */
const STATEMENT = /^\|\s*\*\*([VOIN]\d+(?:\.\d+)?)\*\*\s*\|/;

/** An explicit anchor. The specification writes one before every section it names. */
const ANCHOR = /<a id="([^"]+)"><\/a>/g;

/** A heading, at any level below the document's own title. */
const HEADING = /^(#{2,6})\s+(.*?)\s*$/;

/**
 * Every identifier the specification states, in the order it states them.
 *
 * It refuses rather than guesses three ways: an identifier stated before any anchor, an identifier
 * stated under a line carrying several anchors (which one is the section's would be a choice), and
 * an identifier stated twice.
 */
export function anchorsOf(text: string): Map<string, Anchor> {
  const found = new Map<string, Anchor>();
  let anchor: { ids: string[]; line: number } | null = null;
  let section: string | null = null;
  const lines = text.split('\n');
  for (const [index, line] of lines.entries()) {
    const ids = [...line.matchAll(ANCHOR)].map((match) => match[1] ?? '');
    if (ids.length > 0) {
      anchor = { ids, line: index + 1 };
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading !== null) {
      section = heading[2] ?? '';
      continue;
    }
    const stated = STATEMENT.exec(line);
    if (stated === null) continue;
    const id = stated[1] ?? '';
    if (anchor === null || section === null) {
      throw new Error(`${SPECIFICATION}:${String(index + 1)}: ${id} is stated before any anchored section`);
    }
    if (anchor.ids.length !== 1) {
      throw new Error(
        `${SPECIFICATION}:${String(anchor.line)}: the section stating ${id} carries ` +
          `${String(anchor.ids.length)} anchors, so which one to link to is a guess`,
      );
    }
    if (found.has(id)) {
      throw new Error(`${SPECIFICATION}:${String(index + 1)}: ${id} is stated twice`);
    }
    found.set(id, { anchor: anchor.ids[0] ?? '', section });
  }
  if (found.size === 0) {
    throw new Error(`${SPECIFICATION}: no identifier is stated — the extraction has stopped working`);
  }
  return found;
}

/** The module written beside the shell. */
export function moduleFor(anchors: ReadonlyMap<string, Anchor>): string {
  const entries = [...anchors]
    .map(([id, one]) => `  ${quoted(id)}: { anchor: ${quoted(one.anchor)}, section: ${quoted(one.section)} },`)
    .join('\n');
  const series = [...new Set([...anchors.keys()].map((id) => id[0] ?? ''))].join(', ');
  return `/**
 * Where the specification states each of its identifiers — plan §1, §4.17.
 *
 * **Generated by \`editor/scripts/anchors.ts\` (\`pnpm anchors\`) from \`${SPECIFICATION}\`. Do not
 * edit.** \`tests/audit/anchors.test.ts\` reads the same source and requires this file to equal
 * what the script would write, so a rule renumbered or a section re-anchored fails the audit
 * rather than shipping as a link that lands nowhere.
 *
 * The ${series} series, ${String(anchors.size)} identifiers. Each carries the anchor of the section
 * that **states** it — not of the many places that cite it — and that section's heading, which is
 * what a link says it is taking the reader to.
 */

/** One identifier, and the section that states it. */
export interface SpecificationAnchor {
  /** The section's anchor, as \`${SPECIFICATION}\` writes it and the built page carries it. */
  readonly anchor: string;
  /** The section's heading, word for word. */
  readonly section: string;
}

/** Every identifier the specification states, in the order it states them. */
export const SPECIFICATION_ANCHORS: Readonly<Record<string, SpecificationAnchor>> = {
${entries}
};
`;
}

/** A string as TypeScript source: single quotes, the two escapes a heading can need. */
function quoted(text: string): string {
  return `'${text.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
}

/** Read the specification, write the module. */
export function vendorAnchors(): { bytes: number; anchors: number } {
  const text = readFileSync(join(repositoryRoot, SPECIFICATION), 'utf8');
  const anchors = anchorsOf(text);
  const module = moduleFor(anchors);
  writeFileSync(join(editorRoot, MODULE), module);
  return { bytes: module.length, anchors: anchors.size };
}

const invoked = process.argv[1];
if (invoked !== undefined && resolve(invoked) === fileURLToPath(import.meta.url)) {
  const { bytes, anchors } = vendorAnchors();
  process.stdout.write(`${MODULE}  ${String(anchors)} identifiers, ${(bytes / 1024).toFixed(1)} KiB\n`);
}
