/**
 * Lexeme-preserving JSON: the tree the editor holds a document in, the parser that reads it as
 * the tools read it, and the serializer that writes the corpus's bytes back (plan D12, §5.5,
 * §7 F7).
 *
 * `parse` and `serialize` are the two calls the plan's API sketch names (Appendix C); everything
 * else here is what they are made of, and what the store and the sheets will edit a tree with.
 */
export {
  isJsonArray,
  isJsonNumber,
  isJsonObject,
  jsonInteger,
  jsonNumber,
  jsonObject,
  jsonReal,
  getMember,
  hasMember,
  indexOfMember,
  memberNames,
  put,
  toPlain,
  withMember,
  withoutMember,
  type JsonArray,
  type JsonMember,
  type JsonNumber,
  type JsonObject,
  type JsonValue,
} from './tree.js';
export { formatNumber, isNumberLexeme, lexemeDenotes, lexemeIsReal, NON_FINITE_LEXEMES } from './number.js';
export { JsonParseError, parse, spansOf, type JsonSpan, type ParseOptions } from './parse.js';
// `pointerSegment` is *not* re-exported here: `schema/index.ts` already exports it and the
// package root star-exports both, where two exports of one name are silently excluded (the
// lesson of feature 1.4). One name, one place.
export { jsonPointerOf } from './pointer.js';
export { serialize } from './serialize.js';
