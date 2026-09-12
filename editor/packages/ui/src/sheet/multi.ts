/**
 * §4.11's multi-selection: *the intersection of the editable rows*.
 *
 * > several selected instances show the intersection of their editable rows (primitive, families,
 * > a common argument) with "multiple values" where they differ.
 *
 * Features 2.10 and 2.12 both left this waiting on "a selection that can hold more than one
 * place"; feature 2.14's rubber band (§4.4's Shift+drag) is that selection, so this is where the
 * sheet is answered.
 *
 * **What "the intersection" is, stated.** A row is in it when *every* selected site has a row of
 * that argument path — which is the core's answer per site, since which arguments a site has
 * depends on its primitive and on its own arguments (`describe`, §4.12). Two sites of one
 * primitive intersect in all of them; two sites of different primitives intersect in whatever
 * their declarations happen to share, which is the honest reading of "a common argument" and
 * needs no rule about primitives being the same.
 *
 * **What is compared, and what is not.** Two rows agree when they were **written** the same way:
 * the same source mode and the same text. A row defaulted in both is the same row; a row written
 * `8` in one and defaulted to `8` in the other is *not*, and the sheet says so rather than hiding
 * a difference that the file carries. That is the same rule §4.12's own `source` column states.
 */
import { pyStr, type SiteDescription } from '@tensorspine/lang';

import type { ArgumentRow, ArgumentSheet } from './arguments.js';

/** One row of the intersection: the row of each site, and whether they agree. */
export interface MultiRow {
  /** The argument path, which is what the rows are matched on. */
  readonly path: string;
  /** The row of the first selected site — what the sheet draws. */
  readonly row: ArgumentRow;
  /** The same row at every other selected site, in the selection's order. */
  readonly also: readonly ArgumentRow[];
  /** Whether the sites write it differently — §4.11's "multiple values". */
  readonly differs: boolean;
}

/** What the sheet of several selected sites shows. */
export interface MultiSheet {
  /** How many sites are selected. */
  readonly count: number;
  /** The primitive they all pin, or `null` where they do not all pin one. */
  readonly primitive: string | null;
  /** The version they all pin, or `null`. */
  readonly version: string | null;
  /** The families every one of them carries, in the first site's order. */
  readonly families: readonly string[];
  /** Whether the families differ from site to site. */
  readonly familiesDiffer: boolean;
  /** The names of the selected sites, in the selection's order — the sheet's own title line. */
  readonly names: readonly string[];
  /** The rows every site has, in the first site's order. */
  readonly rows: readonly MultiRow[];
  /** How many rows one site has that the others do not — what the intersection left out. */
  readonly dropped: number;
}

/** The intersection of several sites' sheets. */
export function multiSheet(
  sites: readonly { readonly site: SiteDescription; readonly sheet: ArgumentSheet }[],
): MultiSheet {
  const first = sites[0];
  const rest = sites.slice(1);
  if (first === undefined) {
    return {
      count: 0,
      primitive: null,
      version: null,
      families: [],
      familiesDiffer: false,
      names: [],
      rows: [],
      dropped: 0,
    };
  }
  const byPath = rest.map((one) => new Map(one.sheet.rows.map((row) => [row.path, row])));
  const rows: MultiRow[] = [];
  let dropped = 0;
  for (const row of first.sheet.rows) {
    const found = byPath.map((held) => held.get(row.path));
    if (found.some((one) => one === undefined)) {
      dropped += 1;
      continue;
    }
    const also = found as ArgumentRow[];
    rows.push({ path: row.path, row, also, differs: also.some((one) => !agree(row, one)) });
  }
  for (const held of byPath) dropped += [...held.keys()].filter((path) => !rows.some((row) => row.path === path)).length;

  const primitives = sites.map((one) => pyStr(one.site.primitive));
  const versions = sites.map((one) => pyStr(one.site.version));
  const families = sites.map((one) => one.site.families.map((family) => pyStr(family)));
  const shared = (families[0] ?? []).filter((family) =>
    families.every((each) => each.includes(family)),
  );
  return {
    count: sites.length,
    primitive: primitives.every((one) => one === primitives[0]) ? (primitives[0] ?? null) : null,
    version: versions.every((one) => one === versions[0]) ? (versions[0] ?? null) : null,
    families: shared,
    familiesDiffer: families.some((each) => each.length !== shared.length),
    names: sites.map((one) => pyStr(one.site.key.name)),
    rows,
    dropped,
  };
}

/** Whether two rows of the same argument were written the same way (§4.12's own `source`). */
function agree(one: ArgumentRow, other: ArgumentRow): boolean {
  if (one.source !== other.source) return false;
  if ((one.mode ?? '') !== (other.mode ?? '')) return false;
  return (one.expression ?? '') === (other.expression ?? '');
}
