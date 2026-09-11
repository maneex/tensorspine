/**
 * `Download Workspace as Zip` (§4.4, §4.3) — the archive, written here because nothing else can.
 *
 * > Download Workspace as Zip (a snapshot edited in the browser, whole)
 *
 * A read-only snapshot is a copy the page was handed: every Save downloads one file, and a
 * session of twenty edits is twenty downloads the user has to put back one by one. The command
 * hands over the whole folder instead, as the user's own files with the editor's edits in them.
 *
 * **Stored, never deflated.** A ZIP entry may be stored (method 0) or deflated (method 8), and
 * nothing in a browser compresses a byte string synchronously without a stream API this package
 * may not name (it is compiled without the DOM). Stored is the format's own answer, every reader
 * takes it, and a workspace of JSON is a few megabytes at most — the corpus and the reference base
 * together are 3.5 MB. The alternative was a dependency, and a dependency that writes bytes we
 * cannot check is worse than a hundred lines that are checked.
 *
 * **What it does not do.** No Zip64, so an archive of more than 65 535 entries, or one where a
 * file or the archive itself passes 4 GiB, is refused by name rather than written wrong; no
 * directory entries (every reader creates the directories a path implies); no comment; no data
 * descriptor (the sizes are known before the entry is written).
 */

/** One file of the archive. */
export interface ZipEntry {
  /** The path inside the archive, `/`-separated, relative — a workspace path as it stands. */
  readonly path: string;
  /** The bytes, or the text they are the UTF-8 of. */
  readonly content: string | Uint8Array;
  /** When it was last written; the epoch's start where a workspace cannot say. */
  readonly modified?: Date;
}

/** Raised where an archive cannot be written the way this module writes one. */
export class ZipError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ZipError';
  }
}

/** What the format can hold without Zip64: an unsigned 32-bit size, and 16 bits of entry count. */
const LIMIT = { size: 0xffff_ffff, entries: 0xffff } as const;

/** The four signatures, as the format writes them. */
const SIGNATURE = { local: 0x0403_4b50, central: 0x0201_4b50, end: 0x0605_4b50 } as const;

/** Version 2.0: what "stored, with a UTF-8 name" needs, and what every reader since 1993 has. */
const VERSION = 20;

/** Bit 11 of the general-purpose flags: the name and comment are UTF-8. */
const UTF8_NAMES = 0x0800;

/** Method 0 — the bytes as they are. */
const STORED = 0;

/** The table of CRC-32/ISO-HDLC, built once from its reversed polynomial. */
const CRC_TABLE = ((): Uint32Array => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) value = value & 1 ? (value >>> 1) ^ 0xedb8_8320 : value >>> 1;
    table[index] = value >>> 0;
  }
  return table;
})();

/** CRC-32/ISO-HDLC of some bytes — the checksum every entry of a ZIP carries. */
export function crc32(bytes: Uint8Array): number {
  let value = 0xffff_ffff;
  for (const byte of bytes) value = (value >>> 8) ^ (CRC_TABLE[(value ^ byte) & 0xff] as number);
  return (value ^ 0xffff_ffff) >>> 0;
}

/** A date as MS-DOS wrote one: a 16-bit date and a 16-bit time, two-second resolution. */
export function dosStamp(when: Date): { date: number; time: number } {
  // 1980 is the epoch the format counts years from; anything before it is written as its start.
  const year = Math.max(1980, when.getFullYear());
  const date = ((year - 1980) << 9) | ((when.getMonth() + 1) << 5) | when.getDate();
  const time = (when.getHours() << 11) | (when.getMinutes() << 5) | (when.getSeconds() >> 1);
  return { date: date & 0xffff, time: time & 0xffff };
}

/** A growable little-endian byte writer — what the three record kinds are written through. */
class Bytes {
  private held = new Uint8Array(1024);
  private at = 0;

  private room(more: number): void {
    if (this.at + more <= this.held.length) return;
    let size = this.held.length * 2;
    while (size < this.at + more) size *= 2;
    const grown = new Uint8Array(size);
    grown.set(this.held.subarray(0, this.at));
    this.held = grown;
  }

  get length(): number {
    return this.at;
  }

  u16(value: number): void {
    this.room(2);
    this.held[this.at] = value & 0xff;
    this.held[this.at + 1] = (value >>> 8) & 0xff;
    this.at += 2;
  }

  u32(value: number): void {
    this.room(4);
    this.held[this.at] = value & 0xff;
    this.held[this.at + 1] = (value >>> 8) & 0xff;
    this.held[this.at + 2] = (value >>> 16) & 0xff;
    this.held[this.at + 3] = (value >>> 24) & 0xff;
    this.at += 4;
  }

  put(bytes: Uint8Array): void {
    this.room(bytes.length);
    this.held.set(bytes, this.at);
    this.at += bytes.length;
  }

  done(): Uint8Array {
    return this.held.slice(0, this.at);
  }
}

/** One entry, prepared: its name, its bytes, its checksum and where its local header went. */
interface Prepared {
  readonly name: Uint8Array;
  readonly bytes: Uint8Array;
  readonly crc: number;
  readonly date: number;
  readonly time: number;
  offset: number;
}

/**
 * The archive, as bytes.
 *
 * The entries are written in the order they are given — the caller sorts, because a listing's
 * order is the caller's business — and each appears once: a repeated path is a caller's mistake
 * and is refused rather than written twice.
 */
export function zipOf(entries: readonly ZipEntry[]): Uint8Array {
  if (entries.length > LIMIT.entries) {
    throw new ZipError(
      `${String(entries.length)} files is more than a ZIP without Zip64 can hold (${String(LIMIT.entries)})`,
    );
  }
  const encoder = new TextEncoder();
  const seen = new Set<string>();
  const prepared: Prepared[] = entries.map((entry) => {
    const path = entry.path.replace(/^\/+/, '');
    if (path === '') throw new ZipError('an archive entry has no path');
    if (seen.has(path)) throw new ZipError(`${path} appears twice in the archive`);
    seen.add(path);
    const bytes = typeof entry.content === 'string' ? encoder.encode(entry.content) : entry.content;
    if (bytes.length > LIMIT.size) {
      throw new ZipError(`${path} is larger than a ZIP without Zip64 can hold`);
    }
    const stamp = dosStamp(entry.modified ?? new Date(0));
    return { name: encoder.encode(path), bytes, crc: crc32(bytes), date: stamp.date, time: stamp.time, offset: 0 };
  });

  const out = new Bytes();
  for (const one of prepared) {
    one.offset = out.length;
    out.u32(SIGNATURE.local);
    out.u16(VERSION);
    out.u16(UTF8_NAMES);
    out.u16(STORED);
    out.u16(one.time);
    out.u16(one.date);
    out.u32(one.crc);
    out.u32(one.bytes.length);
    out.u32(one.bytes.length);
    out.u16(one.name.length);
    out.u16(0);
    out.put(one.name);
    out.put(one.bytes);
  }

  const directory = out.length;
  for (const one of prepared) {
    out.u32(SIGNATURE.central);
    out.u16(VERSION);
    out.u16(VERSION);
    out.u16(UTF8_NAMES);
    out.u16(STORED);
    out.u16(one.time);
    out.u16(one.date);
    out.u32(one.crc);
    out.u32(one.bytes.length);
    out.u32(one.bytes.length);
    out.u16(one.name.length);
    out.u16(0);
    out.u16(0);
    out.u16(0);
    out.u16(0);
    // The external attributes of a regular file, as a Unix writer records them: 0644 in the high
    // sixteen bits. A reader that ignores them creates the file with its own default.
    out.u32(0o100_644 << 16);
    out.u32(one.offset);
    out.put(one.name);
  }
  const directorySize = out.length - directory;
  if (out.length > LIMIT.size) throw new ZipError('the archive is larger than a ZIP without Zip64 can hold');

  out.u32(SIGNATURE.end);
  out.u16(0);
  out.u16(0);
  out.u16(prepared.length);
  out.u16(prepared.length);
  out.u32(directorySize);
  out.u32(directory);
  out.u16(0);
  return out.done();
}
