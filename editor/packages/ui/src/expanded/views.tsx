/**
 * The view feature 2.16 gives the shell — §4.9's read-only tab over D1, by the `kind` its tab
 * carries (§4.2's "one per open model, one per drill-in, one per JSON source view").
 */
import type { JSX } from 'react';

import { EXPANDED_TAB } from '../documents/index.js';
import type { Tab } from '../shell/index.js';

import { ExpandedView } from './Expanded.js';

/** The views this feature gives the shell, by the `kind` their tabs carry. */
export const EXPANDED_VIEWS: Readonly<Record<string, (tab: Tab) => JSX.Element>> = {
  [EXPANDED_TAB]: (tab) => <ExpandedView tab={tab} />,
};
