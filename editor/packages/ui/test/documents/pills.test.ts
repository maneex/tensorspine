import { describe, expect, it } from 'vitest';

import { derivationRow } from '@tensorspine/lang/api';
import type { Verdict } from '@tensorspine/lang/api';

import { derivationPill, validationPill } from '../../src/documents/Pills.js';
import { countsOf, noReading, type Reading } from '../../src/documents/pipeline.js';

// The bar's two pills and the first two of the status bar's eight fields — "the core's validation
// and derivation state" (component inventory §2). Every word of them is a reading of what the core
// answered, and the states a document passes through are what this asks about.

/** A verdict with the rows a case wants, and nothing else of one. */
function verdict(problems: { severity: 'error' | 'warning' | 'notice' }[]): Verdict {
  return {
    stagesRun: ['schema', 'library', 'assignment', 'semantic'],
    problems: problems.map((one, index) => ({
      code: '',
      message: `row ${String(index)}`,
      path: '',
      severity: one.severity,
      source: 'semantic' as const,
    })),
    stats: new Map(),
  };
}

/** A reading in the state a case wants. */
function reading(over: Partial<Reading>): Reading {
  return { ...noReading(3), ...over };
}

describe('the validation state', () => {
  it('says nothing has been asked before the first answer', () => {
    expect(validationPill(noReading(1)).text).toBe('not checked');
  });

  it('says it is checking from the gesture, not from the answer', () => {
    expect(validationPill(reading({ checking: true })).tone).toBe('stale');
    expect(validationPill(reading({ checking: true })).text).toBe('checking…');
  });

  it('says there is nothing to report where the core reported nothing', () => {
    const pill = validationPill(reading({ verdict: verdict([]) }));
    expect(pill).toEqual({ text: 'no problems', tone: 'ok' });
  });

  it('counts every row where one of them is a refusal', () => {
    const pill = validationPill(
      reading({ verdict: verdict([{ severity: 'error' }, { severity: 'warning' }]) }),
    );
    expect(pill).toEqual({ text: '2 problems', tone: 'bad' });
  });

  it('counts the warnings apart where none of them refuses', () => {
    const pill = validationPill(reading({ verdict: verdict([{ severity: 'warning' }]) }));
    expect(pill).toEqual({ text: '1 warnings', tone: 'der' });
  });

  it('is counted from the verdict’s own severities and from nothing else', () => {
    expect(countsOf(null)).toEqual({ errors: 0, warnings: 0 });
    expect(countsOf(verdict([{ severity: 'error' }, { severity: 'notice' }]))).toEqual({
      errors: 1,
      warnings: 0,
    });
  });
});

describe('the derivation state (§4.18’s fresh / stale / failed)', () => {
  it('says nothing has been derived before anything has', () => {
    expect(derivationPill(noReading(1)).text).toBe('not derived');
    // And a document being re-checked with nothing derived yet is still not derived: "stale" is
    // a word about a figure that exists, and there is none.
    expect(derivationPill(reading({ derivation: 'stale', derived: null })).text).toBe('not derived');
  });

  it('says fresh only where the derivation was for the revision the document is at', () => {
    const fresh = reading({ derived: {}, derivedAt: 3, derivation: 'fresh' });
    expect(derivationPill(fresh)).toEqual({ text: 'derived · fresh', tone: 'der' });
    expect(derivationPill({ ...fresh, revision: 4 }).text).toBe('derived · stale');
  });

  it('says a refusal is a refusal, whichever revision it was for', () => {
    const refusal = derivationRow(new Error('not valid, no products'), 'x.json');
    expect(derivationPill(reading({ derivation: 'failed', failure: refusal })).tone).toBe('bad');
    expect(
      derivationPill(reading({ derivation: 'failed', derived: {}, derivedAt: 1, revision: 2 })).text,
    ).toBe('derivation failed');
  });

  it('says a document the core skipped is not derived, which is not a failure', () => {
    // A document with a refusal, and a template with no assignment: the tools call both *skipped*
    // rather than failed, and the bar says the same.
    expect(derivationPill(reading({ derivation: 'skipped' })).text).toBe('not derived');
    expect(derivationPill(reading({ derivation: 'skipped' })).tone).toBe('stale');
  });
});
