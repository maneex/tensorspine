"""The sets `--lint` is read over, and the base one of them adds (feature 1.10).

`--lint`'s answer is a function of the **set** of documents it was given, not of one document:
"called by none of the N model(s) linted" names the set's size, and what the set calls is what
decides which primitives are reported. Feature 0.1 recorded that for the oracle — one file per
set, the manifest saying which file holds a document's findings — and this step is that reading
carried through: every case here is one whole `lint.run` invocation, recorded as the lines it
printed.

What the repository alone reaches, measured. The corpus of fourteen models lints **clean** —
every primitive of the reference base is called, every axis outside the `storage` space and every
precision role is cited, and no document of the repository produces an advisory at all (feature
1.6c's finding). The template alone reaches the other summary line and thirty-two uncalled
primitives. Nothing else of `lint.py` is exercised:

- an **advisory**, which needs a self-indexed state on a fragmented stream that is not carried —
  feature 1.6c's `advisory-self-indexed-not-carried`, reused here at the set level;
- the **deduplication** `run` does over `(scope, message)`, which needs one finding produced
  twice — the same document linted twice;
- an **axis or precision role no primitive cites**, which no edit of a *document* can reach: the
  reference base cites all of them. It needs a base of its own, which is what `LAB` below is;
- the **`storage` skip** the other way round, and the fact that the vocabulary is read against
  **one definition per name, the highest version** — both of which need two versions of one
  primitive, again a base of its own;
- the two places `--lint` **raises**: `uncalled_primitives` indexes `instances` and
  `compositions` on every document *before* the grammar is checked, and `_called_primitives`
  catches `OSError` alone, so a document missing either member or not JSON at all crashes the
  command instead of being reported as off the schema;
- the **template recursion**, which the corpus does not make observable at all: measured,
  `shieldstral-3b-composite` is the one document that instantiates a template and it calls every
  primitive the template calls itself, so a port that never followed a template would answer the
  same on every document of the repository.

What no edit of a repository file can reach at all, stated rather than left open:

- `_called_primitives`' `OSError` branch. The command line refuses a path that is not a file
  before it dispatches, and a template's file was checked to exist by the load that pinned it, so
  no `--lint` invocation can reach the branch that returns the set unchanged. The port has no such
  branch: the caller hands it texts (§5.3), and reading is the caller's.
- a primitive `--lint` reports whose message names a version other than `1.0.0` **from the
  reference base alone**: every one of its thirty-six primitives is at `1.0.0`. `LAB` carries a
  second version so that the version in the message is read from the definition rather than
  assumed.
"""

# --- the base one set adds, built out of repository units -------------------
#
# Each unit is a copy of a reference unit — a pointer into a repository file with a few values
# changed, the idiom features 1.3, 1.4, 1.5 and 1.6c all use — so that the base is on the unit
# schema and resolves against the reference base's own axes and roles without stating any of it
# twice.

AXIS = 'data/primitive-library/axes/model/width.json'
STORAGE_AXIS = 'data/primitive-library/axes/storage/multiplicity.json'
ROLE = 'data/primitive-library/precision/norm/scale.json'
PRIMITIVE = 'data/primitive-library/primitives/norm/rms/1.0.0.json'

#: The base's own manifest. `primitive_library` and `title` are the two members
#: `base_definition` requires; no `templates`, since the base carries no template primitive.
LAB_MANIFEST = {
    'schema': 'tensorspine-primitive-library-unit/2.0',
    'kind': 'base',
    'name': 'tensorspine.lint-lab',
    'definition': {
        'primitive_library': 'tensorspine/lint-lab',
        'title': 'A base for the lint suite',
        'summary': 'Vocabulary no primitive cites, and one primitive at two versions.',
    },
}

#: One unit of the base: where it goes, which repository unit it copies, and the edits.
#:
#: `zzz.demo` is `norm.rms` under another name, at two versions. Version 1.0.0's parameter is
#: shaped on `zzz.only`; version 1.1.0's is not — and since `unreferenced_vocabulary` walks
#: `cat['primitives']`, which carries the *highest* version alone, `zzz.only` reads as cited by no
#: primitive. That is the rule as written, and it is what this unit is here to state.
#:
#: A citation is a string *equal* to the name: `_strings` yields whole strings, so `zzz.only`
#: written inside a sentence would not count and the shape's own `axis` is what makes it count.
LAB = [
    ('axes/zzz/unused.json', AXIS, [('/name', 'zzz.unused')]),
    ('axes/zzz/stored.json', STORAGE_AXIS, [('/name', 'zzz.stored')]),
    ('axes/zzz/only.json', AXIS, [('/name', 'zzz.only')]),
    ('precision/zzz/unused.json', ROLE, [('/name', 'zzz.unused')]),
    ('primitives/zzz/demo/1.0.0.json', PRIMITIVE, [
        ('/name', 'zzz.demo'),
        ('/definition/parameters/weight/shape/axes/0/axis', 'zzz.only'),
    ]),
    ('primitives/zzz/demo/1.1.0.json', PRIMITIVE, [
        ('/name', 'zzz.demo'),
        ('/definition/version', '1.1.0'),
    ]),
]

# --- the edited documents some sets lint ------------------------------------

LLAMA = 'data/models/llama3-8b.json'
VOXTRAL = 'data/models/voxtral-realtime.json'

#: `voxtral-realtime` cut to two layers per composition, as feature 1.6c cuts it, so that the
#: advisory below is two lines instead of sixty-four.
VOX_SHORT = [
    {'pointer': '/compositions/encoder/indices/layer/stop/literal', 'op': 'set', 'value': 2},
    {'pointer': '/compositions/decoder/indices/layer/stop/literal', 'op': 'set', 'value': 2},
    {'pointer': '/quantities/enc_layers/source/value', 'op': 'set', 'value': 2},
    {'pointer': '/quantities/dec_layers/source/value', 'op': 'set', 'value': 2},
]

#: The advisory: a self-indexed state on a fragmented stream that is not carried, which
#: `attention.dense` produces once its `streaming` argument is off (feature 1.6c's case).
ADVISORY = VOX_SHORT + [
    {'pointer': '/compositions/encoder/instances/attn/arguments/streaming',
     'op': 'set', 'value': {'literal': False}},
]

#: A document the grammar refuses for a member `model_advisories` never reaches the analysis of.
OFF_SCHEMA = [{'pointer': '/quantities/d/type', 'op': 'delete'}]

#: A document without `compositions`: `uncalled_primitives` indexes it before anything checks
#: the grammar, so `--lint` raises instead of reporting the document as off the schema.
NO_COMPOSITIONS = [{'pointer': '/compositions', 'op': 'delete'}]

#: A document whose `primitive` names nothing the bases carry: `_called_primitives` adds the name
#: to the called set and walks no template for it, and the command goes on.
UNKNOWN_PRIMITIVE = [{'pointer': '/instances/embed/primitive/name', 'op': 'set', 'value': 'zzz.absent'}]

#: A document that instantiates the one template primitive of the repository and nothing else.
#:
#: Measured: `shieldstral-3b-composite` is the only document that instantiates a template, and it
#: calls every primitive the template calls itself (`attention.dense`, `ffn.gated`, `norm.rms`,
#: `residual.add`), so the recursion adds **nothing** to its called set — a port that did not
#: follow templates at all would answer the same on the whole corpus. This document is two
#: members long for that reason: the names the template contributes are exactly the primitives
#: the report does *not* list, and nothing else of the document decides anything. It is off the
#: grammar, and deliberately: `uncalled_primitives` walks `instances` and `compositions` before
#: anything checks it, which is the same fact the `no-compositions` set states from the other side.
TEMPLATE_INSTANCE_ONLY = (
    '{\n'
    '  "instances": {\n'
    '    "text": {"primitive": {"name": "decoder.causal_yarn", "version": "1.0.0"}}\n'
    '  },\n'
    '  "compositions": {}\n'
    '}\n'
)

#: A document with one member written twice.
#:
#: The two rules read it differently, and the difference is the tools' own: `_called_primitives`
#: uses a plain `json.load`, which resolves a duplicate to the **last** value, while
#: `model_advisories` crosses `validate.structural`, whose JSON layer **refuses** it as V12. So the
#: template is called — the report does not list the four primitives it calls — and the document is
#: reported as off the schema with the JSON layer's own line, the one place a `[V12]` reaches
#: `--lint` at all.
DUPLICATE_MEMBER = (
    '{\n'
    '  "instances": {},\n'
    '  "instances": {\n'
    '    "text": {"primitive": {"name": "decoder.causal_yarn", "version": "1.0.0"}}\n'
    '  },\n'
    '  "compositions": {}\n'
    '}\n'
)

#: A text that is not JSON at all. `_called_primitives` catches `OSError` and nothing else, so
#: `json.load`'s own refusal leaves `--lint` uncaught — the hole features 1.1 and 1.3 recorded on
#: the model and unit sides, met here a third time.
NOT_JSON = 'this is not a document\n'

# Each set: a name, the documents it lints, and the bases — `None` for "the first document's
# own", which is what the command line does without `--primitive-library`.
#
# A document is either a repository path, `('edit', <source>, <name>, <edits>)` — written where
# the tools can open it, and re-derived by the port from the same source and the same edits — or
# `('raw', <name>, <text>)`, whose bytes are the fixture's and are read identically by both.
CASES = [
    # --- the two sets feature 0.1's files already record, recomputed here so that the in-process
    # reading and the command line's are held to each other ----------------------
    ('corpus', 'corpus', None),
    ('template', [('repository', 'data/models/decoder-causal-yarn/1.0.0.json')], None),

    # --- the size of the set is in every uncalled line -------------------------
    ('llama-alone', [('repository', LLAMA)], None),
    # The one corpus document that instantiates a template. It calls every primitive the template
    # calls itself, so the recursion changes nothing here; `template-instance-only` below is what
    # makes the recursion visible.
    ('composite-alone', [('repository', 'data/models/shieldstral-3b-composite.json')], None),
    ('llama-and-composite', [('repository', LLAMA),
                             ('repository', 'data/models/shieldstral-3b-composite.json')], None),

    # --- the advisory, and the deduplication over (scope, message) -------------
    ('advisory', [('edit', VOXTRAL, 'voxtral-streaming-off', ADVISORY)], ['data/primitive-library']),
    ('advisory-twice', [('edit', VOXTRAL, 'voxtral-streaming-off', ADVISORY),
                        ('edit', VOXTRAL, 'voxtral-streaming-off', ADVISORY)],
     ['data/primitive-library']),

    # --- a document the grammar refuses, from the repository and edited --------
    ('rejections', [('repository', 'tests/rejections/models/structural-unknown-field.json'),
                    ('repository', 'tests/rejections/models/v1-unknown-primitive.json'),
                    ('repository', 'tests/rejections/models/v7-input-fed-twice.json')],
     ['data/primitive-library']),
    ('off-schema', [('edit', LLAMA, 'llama-without-a-quantity-type', OFF_SCHEMA)],
     ['data/primitive-library']),
    ('unknown-primitive', [('edit', LLAMA, 'llama-unknown-primitive', UNKNOWN_PRIMITIVE)],
     ['data/primitive-library']),

    # --- the vocabulary rule, which needs a base of its own --------------------
    ('lab', 'corpus', ['data/primitive-library', 'lab']),
    ('lab-alone', [('repository', LLAMA)], ['data/primitive-library', 'lab']),

    # --- the template recursion, in isolation ----------------------------------
    ('template-instance-only', [('raw', 'template-instance-only', TEMPLATE_INSTANCE_ONLY)],
     ['data/primitive-library']),
    ('duplicate-member', [('raw', 'duplicate-member', DUPLICATE_MEMBER)],
     ['data/primitive-library']),

    # --- the two places the command raises -------------------------------------
    ('no-compositions', [('edit', LLAMA, 'llama-without-compositions', NO_COMPOSITIONS)],
     ['data/primitive-library']),
    ('not-json', [('raw', 'not-a-document', NOT_JSON)], ['data/primitive-library']),
]
