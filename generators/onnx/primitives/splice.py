"""splice@2.0.0 — inserts an already-projected stream into the token sequence; emitted for a source
that delivers nothing (the text-only path of a multimodal document, §7): the output is the text.

| branch                     | status                                                        |
|----------------------------|---------------------------------------------------------------|
| empty `source`             | emitted: the output is `text`                                 |
| non-empty `source`         | refused: the language does not yet say where the elements go  |
"""
from primitives._common import supports_from

PRIMITIVE = ("splice", "2.0.0")
CAPABILITIES = {"arguments": {"width": "any"}, "states": [], "transforms": ["insert"],
                "notes": ["source must deliver nothing: the language does not yet say where inserted elements go"]}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def emit(ctx, arguments, inputs, params, states):
    if 'source' in inputs:
        raise ValueError("splice: the placement of the inserted elements in the token stream is not declared by the language")
    return {'output': ctx.b.node('Identity', [inputs['text']], hint=f"{ctx.node}.text")}
