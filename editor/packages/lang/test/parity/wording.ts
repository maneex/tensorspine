/**
 * How a moved message reads when the parity job fails (feature 1.12).
 *
 * The wording of the tools' refusals *is* the parity contract: the rejection suite matches on
 * substrings such as `required argument missing 'mask'`, and the core prints the tools' messages
 * so that those substrings are found (D2). The plan's finding F1 records the cost and the
 * compensation in one line — "a wording change in `tools/` breaks the core's parity job without
 * any change of meaning … until then, the core mirrors the wording; **the parity job names the
 * message that moved**" — and this is that naming.
 *
 * A `.some(line => line.includes(match))` assertion that fails says only `false is not true`, and
 * the reader has to run the core by hand to learn what it answered instead. What is wanted, and
 * what this writes, is both sides at once: the substring nobody carried, and the lines that were
 * there. The wording that moved is then visible without a second run — which is the difference
 * between a failure a reviewer reads and a failure a reviewer reproduces.
 */

/** How many lines of the answer a failure shows before it stops. */
const SHOWN = 6;

/**
 * The failure text for a substring no line of the core's answer carries.
 *
 * @param where the case: a fixture document, a base, a unit — whatever names the input.
 * @param match the substring the wording contract asks for, as the fixture writes it.
 * @param lines what the core answered, in its own order.
 */
export function movedMessage(where: string, match: string, lines: readonly string[]): string {
  const shown = lines.slice(0, SHOWN).map((line) => `    ${line}`);
  if (lines.length > SHOWN) shown.push(`    … ${String(lines.length - SHOWN)} more`);
  const answered = lines.length === 0 ? ['    (nothing)'] : shown;
  return [
    `${where}: the wording moved`,
    `  the contract asks for a line carrying: ${match}`,
    '  the core answered:',
    ...answered,
  ].join('\n');
}
