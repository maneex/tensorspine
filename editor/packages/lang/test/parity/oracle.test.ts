import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  editorRoot,
  oracleGenerated,
  oracleGenerator,
  oracleOut,
  readOracleManifest,
  repositoryRoot,
} from './oracle.js';

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';

// The parity layer of the implementation plan's §0.3 compares the core with the repository's
// tools over the oracle's material (§0.5). The core is not ported yet; what the layer can
// already hold to account is its own wiring — the generator's place, the output directory's
// exclusion from the tree, and the manifest's agreement with the corpus once it is generated.
describe('the parity oracle', () => {
  it('has its generator where the suites look for it', () => {
    expect(existsSync(oracleGenerator)).toBe(true);
  });

  it('writes into a directory the repository does not carry', () => {
    const ignored = readFileSync(join(editorRoot, '.gitignore'), 'utf8');
    expect(ignored.split(/\r?\n/)).toContain('tests/oracle/out/');
  });

  it.runIf(inCI)('is generated before the suites in CI', () => {
    expect(oracleGenerated()).toBe(true);
  });

  it.skipIf(!oracleGenerated())('names every product it wrote, and wrote every one it names', () => {
    const manifest = readOracleManifest();
    const documents = manifest.documents.map((d) => d.name).sort();

    expect(documents).toContain('llama3-8b');
    expect(documents).toContain('decoder-causal-yarn@1.0.0');
    expect(new Set(documents).size).toBe(documents.length);

    for (const document of manifest.documents) {
      expect(existsSync(join(repositoryRoot, document.path)), document.path).toBe(true);
      for (const product of [document.d1, document.derived, document.validate, document.lint]) {
        expect(existsSync(join(oracleOut, product)), product).toBe(true);
      }
    }

    for (const file of manifest.primitive_schemas.files) {
      expect(existsSync(join(oracleOut, manifest.primitive_schemas.directory, file)), file).toBe(
        true,
      );
    }

    for (const copied of [manifest.rejections.models, manifest.rejections.primitive_library]) {
      expect(existsSync(join(oracleOut, copied)), copied).toBe(true);
    }
    for (const fixture of manifest.rejections.documents) {
      expect(existsSync(join(repositoryRoot, 'tests', 'rejections', fixture)), fixture).toBe(true);
    }
    for (const signature of manifest.signatures) {
      expect(existsSync(join(oracleOut, signature)), signature).toBe(true);
    }
  });
});
