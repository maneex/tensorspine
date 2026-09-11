import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { crc32, dosStamp, zipOf, ZipError } from '../../src/documents/zip.js';

// `Download Workspace as Zip` (§4.4) writes an archive, and an archive nobody can open is worse
// than no command. Three layers of that claim, from the cheapest to the one that matters:
//
//   1. the checksum, against the standard's own check value;
//   2. the record layout, read back out of the bytes this module wrote;
//   3. **a reader that is not this module** — `zipfile`, the one on every machine that runs the
//      oracle (§0.5), spawned from a *test*, which is where the plan admits Python and the only
//      place it ever runs. A self-consistent round trip would prove nothing about the format.

/** The signatures the format fixes, read back out of the bytes. */
const LOCAL = 0x0403_4b50;
const CENTRAL = 0x0201_4b50;
const END = 0x0605_4b50;

function u16(bytes: Uint8Array, at: number): number {
  return (bytes[at] as number) | ((bytes[at + 1] as number) << 8);
}

function u32(bytes: Uint8Array, at: number): number {
  return (
    ((bytes[at] as number) |
      ((bytes[at + 1] as number) << 8) |
      ((bytes[at + 2] as number) << 16) |
      ((bytes[at + 3] as number) << 24)) >>>
    0
  );
}

describe('the checksum every entry carries', () => {
  it('is CRC-32/ISO-HDLC, at the standard’s own check value', () => {
    const encoder = new TextEncoder();
    expect(crc32(encoder.encode(''))).toBe(0);
    // The check value the specification of CRC-32/ISO-HDLC states for "123456789".
    expect(crc32(encoder.encode('123456789'))).toBe(0xcbf4_3926);
    expect(crc32(encoder.encode('a'))).toBe(0xe8b7_be43);
  });
});

describe('the MS-DOS stamp', () => {
  it('counts years from 1980 and seconds in twos', () => {
    // Local time, because the stamp is: MS-DOS recorded the clock on the wall, not UTC.
    expect(dosStamp(new Date(1980, 0, 1, 0, 0, 0))).toEqual({ date: (1 << 5) | 1, time: 0 });
    const stamp = dosStamp(new Date(2026, 8, 11, 13, 45, 31));
    expect(stamp.date).toBe(((2026 - 1980) << 9) | (9 << 5) | 11);
    expect(stamp.time).toBe((13 << 11) | (45 << 5) | 15);
  });

  it('writes anything before 1980 as its start, which is the earliest the format has', () => {
    expect(dosStamp(new Date(0)).date >> 9).toBe(0);
  });
});

describe('an archive of a workspace', () => {
  const entries = [
    { path: 'models/llama3-8b.json', content: '{"model": "llama3-8b"}\n' },
    { path: 'primitive-library/primitive-library.json', content: '{"kind": "base"}\n' },
  ];

  it('writes one local header per file, a directory of the same, and an end record', () => {
    const archive = zipOf(entries);
    expect(u32(archive, 0)).toBe(LOCAL);
    // The end record is the last 22 bytes, there being no comment.
    const end = archive.length - 22;
    expect(u32(archive, end)).toBe(END);
    expect(u16(archive, end + 8)).toBe(entries.length);
    expect(u16(archive, end + 10)).toBe(entries.length);
    const directory = u32(archive, end + 16);
    expect(u32(archive, directory)).toBe(CENTRAL);
    expect(u32(archive, end + 12)).toBe(end - directory);
    // Stored, with the UTF-8 flag set: the two things every reader is told about an entry.
    expect(u16(archive, 8)).toBe(0);
    expect(u16(archive, 6) & 0x0800).toBe(0x0800);
    // The checksum and the two sizes of the first entry are the file's own.
    const bytes = new TextEncoder().encode(entries[0]?.content ?? '');
    expect(u32(archive, 14)).toBe(crc32(bytes));
    expect(u32(archive, 18)).toBe(bytes.length);
    expect(u32(archive, 22)).toBe(bytes.length);
  });

  it('refuses a path written twice, and one written not at all', () => {
    expect(() => zipOf([...entries, entries[0] as (typeof entries)[number]])).toThrow(ZipError);
    expect(() => zipOf([{ path: '', content: '' }])).toThrow(ZipError);
  });

  it('takes bytes as well as text, which is what a workspace of anything but JSON needs', () => {
    const archive = zipOf([{ path: 'a.bin', content: new Uint8Array([0, 1, 2, 255]) }]);
    expect(u32(archive, 18)).toBe(4);
  });
});

/** Whether this machine has the interpreter the oracle needs (§0.5). */
function hasPython(): boolean {
  try {
    execFileSync('python3', ['-c', 'import zipfile'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

describe('an archive read back by a reader that is not this one', () => {
  const available = hasPython();

  it.runIf(available)('opens, lists and decompresses to the texts that went in', () => {
    const files: Record<string, string> = {
      'models/llama3-8b.json': '{"model": "llama3-8b"}\n',
      'models/decoder-causal-yarn/1.0.0.json': '{"version": "1.0.0"}\n',
      'primitive-library/primitive-library.json': '{"kind": "base"}\n',
      // Non-ASCII, as the library units carry it (`ensure_ascii=False`, D12).
      'notes/é.json': '{"note": "é中文"}\n',
    };
    const archive = zipOf(
      Object.entries(files).map(([path, content]) => ({ path, content })),
    );
    const directory = mkdtempSync(join(tmpdir(), 'tensorspine-zip-'));
    try {
      const path = join(directory, 'workspace.zip');
      writeFileSync(path, archive);
      const answer = execFileSync(
        'python3',
        [
          '-c',
          'import json,sys,zipfile\n' +
            'z = zipfile.ZipFile(sys.argv[1])\n' +
            'assert z.testzip() is None\n' +
            'print(json.dumps({n: z.read(n).decode("utf-8") for n in z.namelist()}))',
          path,
        ],
        { encoding: 'utf8' },
      );
      expect(JSON.parse(answer)).toEqual(files);
      // And the bytes on disk are the bytes this module answered.
      expect(readFileSync(path).equals(Buffer.from(archive))).toBe(true);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it.runIf(!available)('is not asked on a machine with no interpreter', () => {
    // Recorded rather than silently skipped: the oracle needs `python3` and so does this.
    expect(available).toBe(false);
  });
});
