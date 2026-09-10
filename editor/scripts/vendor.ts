#!/usr/bin/env node
/**
 * `pnpm vendor` — what the static application ships beside its own code.
 *
 * The editor has no server and no interpreter: the schemas it reads, the reference base and
 * corpus it opens as the Examples workspace, and the artifacts the tools generate are copied
 * into `apps/web/public/vendor/` at build time, from the repository this editor is part of
 * (the editor plan's §1 "the schemas are the repository's files", D11 and D14 "vendoring them
 * at build", F5 "generated artifacts are consumed, never regenerated").
 *
 * What lands under the vendor root:
 *
 *     vendor.json                        the manifest: commit, file list, sha256 per file
 *     schemas/                           the repository's `schemas/`, byte for byte
 *     data/models/                       the corpus, the Examples workspace's models
 *     data/primitive-library/            the reference base
 *     generated/primitive-schema/        `--document primitive-schema`, one file per version
 *     generated/primitive-library.md     `--document primitive-library`
 *
 * The layout under `data/` is the repository's on purpose: a document resolves its bases
 * relative to itself (`"base": "../primitive-library/"`) and the base manifest resolves its
 * templates the same way (`"templates": "../models/"`), so models and base must stay siblings
 * for the Examples workspace to open exactly as `data/` does.
 *
 * Three rules this script keeps:
 *
 *  - It describes; it does not judge. Every fact in the manifest is read from a file that was
 *    copied or from an artifact the tools wrote. Nothing about the language is restated here.
 *  - It refuses without the tools, and a refusal is inert: the requirements are checked before
 *    anything is removed or written, so a failed run leaves the previous output untouched.
 *    Any failure after that removes the output rather than leave a half vendor behind.
 *  - The manifest is written last. A vendor directory without `vendor.json` is not one.
 *
 * The tools decide between one file and one file per unit by asking whether `-o` names an
 * existing directory, so every output directory is made before they are called.
 *
 * Python runs here, in a build script, and never at runtime (the implementation plan's §0.2):
 * `tools/` is the generator of these artifacts and is never ported to JavaScript (F5).
 *
 * Usage:  pnpm vendor
 *         node scripts/vendor.ts [--out DIR] [--repository DIR] [--python EXE] [--quiet]
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** `editor/`, the workspace root: this file lives in `editor/scripts/`. */
const editorRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/** Where the static application looks for its vendored material: Vite serves `public/` as is. */
export const defaultVendorRoot = join(editorRoot, 'apps', 'web', 'public', 'vendor');

/** A refusal with its cause, in the tools' voice: what is missing and what to do about it. */
export class VendorError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'VendorError';
  }
}

export interface VendorOptions {
  /** The checkout to vendor from (default: the repository `editor/` is part of). */
  repositoryRoot?: string | undefined;
  /** Where to write (default: `editor/apps/web/public/vendor`). */
  out?: string | undefined;
  /** The interpreter that runs the tools (default: `$PYTHON`, else `python3`). */
  python?: string | undefined;
  /** Say nothing on success. */
  quiet?: boolean | undefined;
}

/** One vendored file, named relative to the vendor root, with its digest. */
export interface VendorFile {
  path: string;
  bytes: number;
  sha256: string;
}

/** One vendored schema and the `$id` it declares — the key the registry indexes it by. */
export interface VendorSchema {
  path: string;
  id: string | null;
}

/** One group of vendored files: where it landed and where it came from. */
export interface VendorSet {
  name: string;
  path: string;
  source: string;
  files: number;
}

export interface VendorManifest {
  generated_by: string;
  repository_commit: string | null;
  repository_dirty: boolean | null;
  tools: { command: string; python: string; jsonschema: string };
  examples: { root: string; models: string; primitive_library: string };
  schemas: VendorSchema[];
  primitive_schemas: Record<string, string>;
  primitive_library_reference: string;
  sets: VendorSet[];
  files: VendorFile[];
}

export interface VendorResult {
  /** The directory written. */
  out: string;
  /** The manifest, as it was written to `vendor.json`. */
  manifest: VendorManifest;
}

/** The repository directories and files this script copies or runs, in the order it needs them. */
const TOOL = 'tools/tensorspine';
const SCHEMAS = 'schemas';
const MODELS = 'data/models';
const PRIMITIVE_LIBRARY = 'data/primitive-library';

/** Where each of them lands under the vendor root. */
const VENDORED_SCHEMAS = 'schemas';
const VENDORED_EXAMPLES = 'data';
const VENDORED_MODELS = 'data/models';
const VENDORED_PRIMITIVE_LIBRARY = 'data/primitive-library';
const VENDORED_PRIMITIVE_SCHEMAS = 'generated/primitive-schema';
const VENDORED_REFERENCE = 'generated/primitive-library.md';

/** The `$id` the tools give a generated argument schema: `…/primitive/<name>/<version>.json`. */
const PRIMITIVE_SCHEMA_ID = /^[a-z]+:\/\/[^/]+\/primitive\/(.+)\/([^/]+)\.json$/;

/** Every regular file below `root`, as `/`-separated paths, sorted, directories descended. */
function walk(root: string, at = ''): string[] {
  const here = at === '' ? root : join(root, at);
  const entries = readdirSync(here, { withFileTypes: true });
  entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const found: string[] = [];
  for (const entry of entries) {
    const path = at === '' ? entry.name : `${at}/${entry.name}`;
    if (entry.isDirectory()) found.push(...walk(root, path));
    else if (entry.isFile()) found.push(path);
    else throw new VendorError(`${join(root, path)} is neither a file nor a directory — the vendor copies regular files only`);
  }
  return found;
}

function record(path: string, bytes: Buffer): VendorFile {
  return { path, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
}

function write(path: string, bytes: Buffer): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, bytes);
}

/** Copy a repository tree under the vendor root, recording every file's digest as it goes. */
function copyTree(from: string, out: string, prefix: string, files: VendorFile[]): number {
  const names = walk(from);
  for (const name of names) {
    const bytes = readFileSync(join(from, name));
    write(join(out, prefix, name), bytes);
    files.push(record(`${prefix}/${name}`, bytes));
  }
  return names.length;
}

/** One invocation of the tools, from the repository being vendored. Its refusal is ours. */
function tools(python: string, root: string, args: string[]): void {
  const result = spawnSync(python, [join(root, TOOL), ...args], { cwd: root, encoding: 'utf8' });
  if (result.error) {
    throw new VendorError(`${python} could not run the tools (${result.error.message})`);
  }
  if (result.status !== 0) {
    const output = `${result.stdout ?? ''}${result.stderr ?? ''}`.trimEnd();
    throw new VendorError(
      `\`tensorspine ${args.join(' ')}\` exited ${String(result.status)}` +
        (output === '' ? '' : `\n${output}`),
    );
  }
}

/** The version of the interpreter and of `jsonschema`, and the proof that both are there. */
function interpreter(python: string, root: string): { python: string; jsonschema: string } {
  const probe =
    'import json, sys, importlib.metadata as m; ' +
    "print(json.dumps({'python': '.'.join(str(n) for n in sys.version_info[:3]), " +
    "'jsonschema': m.version('jsonschema')}))";
  const result = spawnSync(python, ['-c', probe], { cwd: root, encoding: 'utf8' });
  if (result.error) {
    throw new VendorError(
      `${python} does not run (${result.error.message}) — the tools are Python; ` +
        'set PYTHON to the interpreter to use',
    );
  }
  if (result.status !== 0) {
    const output = `${result.stdout ?? ''}${result.stderr ?? ''}`.trimEnd();
    throw new VendorError(
      "the tools need jsonschema: python3 -m pip install 'jsonschema==4.25.0'" +
        (output === '' ? '' : `\n${output}`),
    );
  }
  return JSON.parse(result.stdout) as { python: string; jsonschema: string };
}

/**
 * Everything the run needs, checked before anything is touched: without the tools there is no
 * vendor, and a refusal must leave whatever was there alone.
 */
function requirements(root: string, python: string): { python: string; jsonschema: string } {
  if (!existsSync(join(root, TOOL))) {
    throw new VendorError(
      `the tools are missing: ${TOOL} is not in ${root} — the vendored artifacts are generated ` +
        'by them, and this script never falls back on stale files',
    );
  }
  for (const path of [SCHEMAS, MODELS, PRIMITIVE_LIBRARY]) {
    if (!existsSync(join(root, path))) {
      throw new VendorError(`${path}/ is missing from ${root} — run this from a checkout of the repository`);
    }
  }
  return interpreter(python, root);
}

/** The commit the vendored bytes come from, and whether the working tree still matches it. */
function head(root: string): { commit: string | null; dirty: boolean | null } {
  const rev = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' });
  if (rev.error || rev.status !== 0) return { commit: null, dirty: null };
  const status = spawnSync('git', ['status', '--porcelain'], { cwd: root, encoding: 'utf8' });
  const dirty = status.error || status.status !== 0 ? null : status.stdout.trim() !== '';
  return { commit: rev.stdout.trim(), dirty };
}

/** The `$id` a JSON document declares, or null when it declares none. */
function identity(bytes: Buffer): string | null {
  const document = JSON.parse(bytes.toString('utf8')) as { $id?: unknown };
  return typeof document.$id === 'string' ? document.$id : null;
}

/**
 * Vendor the repository's schemas, corpus, reference base and generated artifacts.
 *
 * Throws a `VendorError` — never writing a partial result — when the tools are absent or refuse.
 */
export function vendor(options: VendorOptions = {}): VendorResult {
  const root = resolve(options.repositoryRoot ?? resolve(editorRoot, '..'));
  const out = resolve(options.out ?? defaultVendorRoot);
  const python = options.python ?? process.env['PYTHON'] ?? 'python3';

  const versions = requirements(root, python);

  rmSync(out, { recursive: true, force: true });
  try {
    const files: VendorFile[] = [];
    const sets: VendorSet[] = [
      {
        name: 'schemas',
        path: VENDORED_SCHEMAS,
        source: `${SCHEMAS}/`,
        files: copyTree(join(root, SCHEMAS), out, VENDORED_SCHEMAS, files),
      },
      {
        name: 'models',
        path: VENDORED_MODELS,
        source: `${MODELS}/`,
        files: copyTree(join(root, MODELS), out, VENDORED_MODELS, files),
      },
      {
        name: 'primitive-library',
        path: VENDORED_PRIMITIVE_LIBRARY,
        source: `${PRIMITIVE_LIBRARY}/`,
        files: copyTree(join(root, PRIMITIVE_LIBRARY), out, VENDORED_PRIMITIVE_LIBRARY, files),
      },
    ];

    const schemas: VendorSchema[] = walk(join(out, VENDORED_SCHEMAS))
      .filter((name) => name.endsWith('.json'))
      .map((name) => ({
        path: `${VENDORED_SCHEMAS}/${name}`,
        id: identity(readFileSync(join(out, VENDORED_SCHEMAS, name))),
      }));

    // The generated artifacts, as the tools build them (F5). `-o` must name an existing
    // directory for the per-primitive form, and the parent must exist for the reference.
    const schemaDirectory = join(out, VENDORED_PRIMITIVE_SCHEMAS);
    mkdirSync(schemaDirectory, { recursive: true });
    tools(python, root, [
      '--document', 'primitive-schema',
      '--primitive-library', join(root, PRIMITIVE_LIBRARY),
      '--schemas', join(root, SCHEMAS),
      '-o', schemaDirectory,
    ]);

    const primitiveSchemas: Record<string, string> = {};
    for (const name of walk(schemaDirectory)) {
      const path = `${VENDORED_PRIMITIVE_SCHEMAS}/${name}`;
      const bytes = readFileSync(join(schemaDirectory, name));
      const id = identity(bytes);
      const parsed = id === null ? null : PRIMITIVE_SCHEMA_ID.exec(id);
      if (parsed === null) {
        throw new VendorError(
          `${path} declares no primitive $id (${id ?? 'none'}) — the argument-schema map cannot be named`,
        );
      }
      primitiveSchemas[`${parsed[1] ?? ''}@${parsed[2] ?? ''}`] = path;
      files.push(record(path, bytes));
    }
    sets.push({
      name: 'primitive-schema',
      path: VENDORED_PRIMITIVE_SCHEMAS,
      source: `${TOOL} --document primitive-schema`,
      files: Object.keys(primitiveSchemas).length,
    });

    // `--link-base ROOT` leaves the units' repository-relative URLs (`docs/SPECIFICATION.md`)
    // as their authors wrote them: the vendored reference states where a link points in the
    // repository, and what that becomes in a deployment is the deployment's business.
    const reference = join(out, VENDORED_REFERENCE);
    mkdirSync(dirname(reference), { recursive: true });
    tools(python, root, [
      '--document', 'primitive-library',
      '--primitive-library', join(root, PRIMITIVE_LIBRARY),
      '--schemas', join(root, SCHEMAS),
      '--link-base', root,
      '-o', reference,
      join(root, MODELS),
    ]);
    files.push(record(VENDORED_REFERENCE, readFileSync(reference)));
    sets.push({
      name: 'primitive-library-reference',
      path: VENDORED_REFERENCE,
      source: `${TOOL} --document primitive-library --link-base .`,
      files: 1,
    });

    files.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
    const { commit, dirty } = head(root);
    const manifest: VendorManifest = {
      generated_by: 'editor/scripts/vendor.ts',
      repository_commit: commit,
      repository_dirty: dirty,
      tools: { command: TOOL, python: versions.python, jsonschema: versions.jsonschema },
      examples: {
        root: VENDORED_EXAMPLES,
        models: VENDORED_MODELS,
        primitive_library: VENDORED_PRIMITIVE_LIBRARY,
      },
      schemas,
      primitive_schemas: primitiveSchemas,
      primitive_library_reference: VENDORED_REFERENCE,
      sets,
      files,
    };
    // Last, so that a directory without it is not mistaken for a vendor.
    writeFileSync(join(out, 'vendor.json'), `${JSON.stringify(manifest, null, 2)}\n`);

    if (options.quiet !== true) {
      console.log(`vendor: ${String(files.length)} file(s) -> ${out}`);
      for (const set of sets) {
        console.log(`  ${set.path.padEnd(30)} ${String(set.files).padStart(4)}  ${set.source}`);
      }
    }
    return { out, manifest };
  } catch (error) {
    // No half vendor: what cannot be finished is removed rather than left to be shipped.
    rmSync(out, { recursive: true, force: true });
    throw error;
  }
}

function main(argv: string[]): number {
  const options: VendorOptions = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (flag === '--quiet') {
      options.quiet = true;
      continue;
    }
    if (flag !== '--out' && flag !== '--repository' && flag !== '--python') {
      console.error(`vendor: unknown option ${String(flag)}`);
      console.error('usage: node scripts/vendor.ts [--out DIR] [--repository DIR] [--python EXE] [--quiet]');
      return 2;
    }
    if (value === undefined) {
      console.error(`vendor: ${flag} needs a value`);
      return 2;
    }
    if (flag === '--out') options.out = value;
    else if (flag === '--repository') options.repositoryRoot = value;
    else options.python = value;
    index += 1;
  }
  try {
    vendor(options);
    return 0;
  } catch (error) {
    if (error instanceof VendorError) {
      console.error(`vendor: ${error.message}`);
      return 2;
    }
    throw error;
  }
}

const invoked = process.argv[1];
if (invoked !== undefined && resolve(invoked) === fileURLToPath(import.meta.url)) {
  process.exit(main(process.argv.slice(2)));
}
