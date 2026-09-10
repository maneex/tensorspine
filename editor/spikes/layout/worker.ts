import { parentPort } from 'node:worker_threads';

import { layout, type Graph, type Layout, type LayoutSettings } from '@tensorspine/ui/layout';

/**
 * The layout, off the main thread.
 *
 * §4.9 and §5.6 put ELK in a worker so that laying out several hundred nodes never blocks the
 * interface. A Node worker thread is not the browser's Web Worker, but it is the same shape of
 * boundary — the graph is structured-cloned in, the layout structured-cloned out — so running
 * the spike through one measures what the application will pay, and proves the module carries
 * nothing across that a clone cannot.
 */

/** What the script sends. `settings.engine` cannot cross a clone, so the worker uses its own. */
export interface LayoutRequest {
  readonly id: string;
  readonly graph: Graph;
  readonly settings: Omit<LayoutSettings, 'engine'>;
}

/** What comes back: the layout, or the message of whatever refused it. */
export type LayoutResponse =
  | { readonly id: string; readonly layout: Layout }
  | { readonly id: string; readonly failure: string };

const port = parentPort;
if (port === null) throw new Error('this module is a worker entry point');

port.on('message', (request: LayoutRequest) => {
  layout(request.graph, request.settings).then(
    (result) => {
      port.postMessage({ id: request.id, layout: result } satisfies LayoutResponse);
    },
    (error: unknown) => {
      port.postMessage({
        id: request.id,
        failure: error instanceof Error ? error.message : String(error),
      } satisfies LayoutResponse);
    },
  );
});
