import { connectLang, type Lang } from '@tensorspine/lang/api';

/**
 * The language core, in a worker, for the static application (D11, §5.3).
 *
 * "It runs in a Web Worker, so validation and derivation never block the UI, and in the same
 * process on the desktop and on the web." The worker is created here rather than in
 * `packages/ui`, because `new Worker(new URL(…), { type: 'module' })` is a **build**
 * instruction — the bundler follows it and emits the core as its own chunk — and the interface
 * package must import no platform. What the interface receives is the {@link Lang} interface, and
 * it cannot tell which side of a boundary the answers came from.
 *
 * The whole core is in the worker's chunk and none of it in the page's: `connectLang` carries the
 * protocol and the codec, which is a few kilobytes, and the schema registry, the library loader,
 * the validator and the derivation are behind the message port. Feature 0.4's lesson, in its
 * third instance.
 */
export function startLang(): Lang {
  const worker = new Worker(new URL('./worker.ts', import.meta.url), {
    type: 'module',
    name: 'tensorspine-lang',
  });
  return connectLang(worker, {
    onClose: () => {
      worker.terminate();
    },
  });
}
