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

The dumper records cut values, post-prefill state, exposed outputs and generated tokens. Its
`hook_map` is the only mapping between delivery-implementation names and TensorSpine D1/D4 names.
Captured tensors are cloned immediately because delivery implementations may update state in
place.

## Compare and validate

```sh
python3 generators/reference/ref.py run data/models/<model>.json \
  --checkpoint "$TENSORSPINE_MODEL_ARTIFACTS/weights/<artifact>" \
  --truncate <composition>.<layer-site>=<count> \
  --ids <comma-separated-token-ids> \
  --dump /tmp/reference.safetensors

python3 generators/reference/ref.py compare \
  /tmp/reference.safetensors /tmp/<model>.hf.safetensors

python3 tests/run_fixtures.py
python3 generators/reference/tests/run_reference.py
```

`compare` reads the tolerance for the implementation's compute dtype from the fixture. The
reference suite regenerates unit fixtures from their recorded seeds and compares integration
fixtures whenever their artifacts are available.
