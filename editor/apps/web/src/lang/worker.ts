import { serveLang } from '@tensorspine/lang/api/engine';

/**
 * The language core's Web Worker entry point.
 *
 * Two lines, and it is the only place a platform is named. `packages/lang` names none — it runs
 * "unchanged in a Web Worker, in Node and in a test" (§5.1) — so serving its API over a
 * particular transport is the application's, exactly as `Platform` is (§5.2). The desktop shell's
 * entry will be the same two lines over its own port.
 *
 * A worker's global scope *is* its port: it posts and it listens, which is the whole of
 * `LangPort`. That structural minimum is what keeps `packages/lang` free of a DOM type while a
 * `DedicatedWorkerGlobalScope` satisfies it as it stands.
 */
serveLang(globalThis);
