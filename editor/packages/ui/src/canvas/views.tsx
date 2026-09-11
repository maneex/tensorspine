/**
 * The views feature 2.9 gives the shell — §4.7's canvas as the body of a model's own tab.
 *
 * > The default editor of a model: the *folded* document — root instances, compositions as group
 * > boxes, interfaces as terminal nodes — the reading `--view` gives, made editable.
 *
 * The **source pane** feature 2.6 put in that tab moves to a tab of its own, which is where §4.2
 * puts it ("one per JSON source view") and what S1's own strip draws beside `llama3-8b` and
 * `llama3-8b › decoder`. `View ▸ JSON Source` (Ctrl+Shift+J, "opens beside") is what opens it, and
 * feature 2.17 replaces the pane with Monaco and the schema attached.
 */
import type { JSX } from 'react';

import { DOCUMENT_TAB, DocumentView, SOURCE_TAB, SourcePane } from '../documents/index.js';
import type { Tab } from '../shell/index.js';

import { Canvas } from './Canvas.js';

/** The views this feature gives the shell, by the `kind` their tabs carry. */
export const CANVAS_VIEWS: Readonly<Record<string, (tab: Tab) => JSX.Element>> = {
  [DOCUMENT_TAB]: (tab) => <DocumentView tab={tab} body={(one) => <Canvas one={one} />} />,
  [SOURCE_TAB]: (tab) => <DocumentView tab={tab} body={(one) => <SourcePane one={one} />} />,
};
