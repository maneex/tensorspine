/**
 * The transport-ready half of a call's options: what is left once the functions are taken out.
 *
 * `AbortSignal` and `onProgress` are the caller's and stay on the caller's side — a function is a
 * `DataCloneError`, so sending the options whole would fail the call rather than the callback.
 * Both the in-process proxy and the client strip them the same way, here, so that the session
 * receives one shape whichever drove it.
 */
import type { PyRecord } from '../expr/value.js';

import type { CallOptions, DescribeOptions, LibraryHandle, ValidateOptions } from './types.js';

/** What the session is given for a call that reads a document. */
export interface StrippedOptions {
  readonly library: LibraryHandle;
  readonly assignment?: PyRecord;
  readonly revision?: number;
  readonly lint?: ValidateOptions['lint'];
}

/** {@link StrippedOptions} with the sites `describe` was asked for. */
export interface StrippedDescribeOptions extends StrippedOptions {
  readonly only?: readonly string[];
  readonly folded?: boolean;
  readonly compatibility?: boolean;
  readonly identity?: DescribeOptions['identity'];
}

/** The options as the session takes them. */
export function stripped(options: ValidateOptions | CallOptions): StrippedOptions {
  const lint = 'lint' in options ? options.lint : undefined;
  return {
    library: options.library,
    ...(options.assignment === undefined ? {} : { assignment: options.assignment }),
    ...(options.revision === undefined ? {} : { revision: options.revision }),
    ...(lint === undefined ? {} : { lint }),
  };
}

/** The same for `describe`, which names the sites it wants. */
export function strippedDescribe(options: DescribeOptions): StrippedDescribeOptions {
  return {
    ...stripped(options),
    ...(options.only === undefined ? {} : { only: options.only }),
    ...(options.folded === undefined ? {} : { folded: options.folded }),
    ...(options.compatibility === undefined ? {} : { compatibility: options.compatibility }),
    ...(options.identity === undefined ? {} : { identity: options.identity }),
  };
}
