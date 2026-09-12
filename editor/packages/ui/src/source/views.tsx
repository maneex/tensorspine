/**
 * The view feature 2.17 gives the shell — §4.2's "one per JSON source view", with Monaco in it.
 *
 * It is composed by the application beside the canvas's and the expanded graph's
 * (`apps/web/src/app.tsx`), and it is an entry point of its own for the reason the layout is one:
 * what a module imports says what it depends on, and a text editor must not reach the shell's
 * first chunk. Nothing here imports Monaco — `./Source.tsx` does, with `await import(…)`, on the
 * first drawing.
 */
import type { JSX } from 'react';

import { DocumentView, SOURCE_TAB } from '../documents/index.js';
import type { Tab } from '../shell/index.js';

import { SourceEditor } from './Source.js';

/** The views this feature gives the shell, by the `kind` their tabs carry. */
export const SOURCE_VIEWS: Readonly<Record<string, (tab: Tab) => JSX.Element>> = {
  [SOURCE_TAB]: (tab) => <DocumentView tab={tab} body={(one) => <SourceEditor one={one} />} />,
};
