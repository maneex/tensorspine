/**
 * Expressions and conditions as text — plan §4.13, artboard S7.
 *
 * The printing half, which the folded canvas of §4.7 needs for its guard badges, its composition
 * headers, its structural summaries and its boundary handles. Feature 2.11 adds the editors —
 * the tree view, the strict parser, the reference pickers — over the same conventions: the
 * symbols, the forms, the precedences and the prefixes are `presentation.json`'s, read here and
 * read there, so that text and tree cannot drift apart.
 */
export { printAt, printValue, UNPRINTABLE, type PrintContext } from './print.js';
