# Reference fixtures

Every file conforms to `schemas/tensorspine-fixture.schema.json`; the
[fixture-format guide](../../../docs/TENSORSPINE-FIXTURE.md) defines its metadata and tensor-key
grammar. `tests/run_fixtures.py` validates both without loading tensor payloads.

## Layout

| Path | Kind | Source |
|---|---|---|
| `contracts/<contract>@<version>/<case>.safetensors` | Unit | The contract's reference implementation, executed by `ref.py witness` on generated parameters and inputs |
| `*.hf.safetensors` | Integration | The model's delivery implementation, dumped at the TensorSpine document's legal cuts and states |

The fixture metadata carries the contract or model identity, compute dtype, tolerances, artifact
provenance, truncation and delivery-library versions. Do not duplicate those values here; the
generated [status page](https://maneex.github.io/tensorspine/status/) reports recorded verification
state.

## Produce an integration fixture

```sh
python3 generators/reference/fixtures/dump_hf.py \
  --model "$TENSORSPINE_MODEL_ARTIFACTS/weights/<artifact>" \
  --document <model> \
  --artifact-id <published-id> \
  --composition <composition> \
  --layers <count> \
  --ids <comma-separated-token-ids> \
  --steps <count> \
  --dtype f32 \
  --out /tmp/<model>.hf.safetensors
```

For an encoder–decoder with an audio input, `--audio "$TENSORSPINE_MODEL_ARTIFACTS/audio/<sample>.wav"`
(mono 16-bit at the extractor's rate) replaces the causal-LM path: the checkpoint's own feature
extractor turns the sample into the frames the document's `audio` input takes, recorded as
`in/audio` with their provenance (`inputs`), the encoder runs whole and its output is recorded as
the cross source, and `--layers` truncates the decoder; `--attn-site` names the decoder's
self-attention site. The cross-attention cache is not recorded, and `hook_map` says so.

For a streaming model whose token stream joins the audio stream (Voxtral Realtime, told by the
config's `model_type`), the same `--audio` flag takes the artifact's processor (`mistral_common`
and `soundfile` beside `transformers`): the prompt and the delay are the processor's streaming
prefill, the prompt takes its tokens' frames, every step a token and eight frames, and the file
records the frames consumed as `in/audio`, the delay as `in/delay` (a setting: provenance
without a file), the decoder layer outputs, the encoder's final norm, the projector's output and
the time embedding at the prefill, the decoder caches, the two convolution histories and the
held conditions after it. The encoder's sliding-window caches are left out unless
`--encoder-rings` is given — 32 layers of keys and values per encoder position, about 82 MB for
the committed fixture's 156 positions — since every decode step's four encoder positions
exercise them; `hook_map` says so.

The dumper records cut values, post-prefill state, exposed outputs, generated tokens and the
non-token inputs the prefill delivered. Its `hook_map` is the only mapping between
delivery-implementation names and TensorSpine D1/D4 names. Captured tensors are cloned
immediately because delivery implementations may update state in place.

## Recorded samples

The audio a fixture records under `in/audio` comes from a sample in `$TENSORSPINE_MODEL_ARTIFACTS/audio/`
(a directory outside the repository, like the weights). Every sample is open content; the fixture's
`inputs` entry names the file, its hash and the extractor, and this table says where each file
came from and under which licence.

| Sample | Content | Origin | Licence | Preparation |
|---|---|---|---|---|
| `jfk.wav` | John F. Kennedy, inaugural address, 20 January 1961, "And so my fellow Americans…" (11.0 s) | the [whisper.cpp](https://github.com/ggml-org/whisper.cpp) repository's `samples/jfk.wav` | public domain: a work of the United States government | as distributed: 16 kHz, mono, 16-bit; sha256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` |
| `la-cigale-et-la-fourmi.wav` | Jean de La Fontaine, *Fables*, livre I, 1: "La Cigale et la Fourmi", the title and the first half of the fable, read in French by Jean Lambert (33.8 s) | LibriVox, *Fables de La Fontaine, livre 01*, archive.org item [`fables_de_la_fontaine_01_jl_0809_librivox`](https://archive.org/details/fables_de_la_fontaine_01_jl_0809_librivox), file `fables01_01_lafontaine_jl_64kb.mp3` (sha256 `b21986a6b13af473a5cfa8a2ed456fb035a15c4230d6d0f1bdf1d725d3afb020`) | public domain: LibriVox releases its recordings into the public domain ([creativecommons.org/licenses/publicdomain](http://creativecommons.org/licenses/publicdomain/)); the text is public domain (1668) | seconds 14.0 to 47.84 of the file, after the reader's announcement, converted by ffmpeg to 16 kHz, mono, 16-bit; sha256 `8bae051b7d4ba412b996e5289c552de860a63c3d4507f388dcad6a86c0fbf35a` |

A new sample goes in the same directory with a row here: what is spoken, where the file came from,
its licence, and how the WAV was made from it.

## Compare and validate

```sh
python3 generators/reference/ref.py run data/models/<model>.json \
  --checkpoint "$TENSORSPINE_MODEL_ARTIFACTS/weights/<artifact>" \
  --truncate <composition>.<layer-site>=<count> \
  --ids <comma-separated-token-ids> \
  --input <input>=/tmp/<model>.hf.safetensors      # a non-token input, from the fixture's own in/<input>
  --dump /tmp/reference.safetensors

python3 generators/reference/ref.py compare \
  /tmp/reference.safetensors /tmp/<model>.hf.safetensors

python3 tests/run_fixtures.py
python3 generators/reference/tests/run_reference.py
```

`compare` reads the tolerance for the implementation's compute dtype from the fixture. The
reference suite regenerates unit fixtures from their recorded seeds and compares integration
fixtures whenever their artifacts are available.
