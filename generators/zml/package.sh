#!/bin/sh
# Package `tspl`, one derived document and its weights into a single .tar.bz2 that runs
# on another machine.
#
#   generators/zml/package.sh MODEL [-o FILE] [--weights NAME] [--no-weights] [--jobs N]
#
# The binary is not standalone: it panics with "Unable to initialize runfiles" unless
# `tspl.runfiles/` sits beside it, and that tree carries the PJRT plugins — 3.7 GB of
# CUDA alone. So the package is the binary, its runfiles, the document and the weights,
# laid out exactly as `$TENSORSPINE_MODEL_ARTIFACTS` describes, plus a `run.sh` that
# needs no environment set on the far side.
#
# Nothing of this machine goes in: the runfiles are symlinks into a Bazel cache and the
# weights may be a symlink of their own, so everything is dereferenced on the way in.
set -eu

usage() {
    cat >&2 <<'USAGE'
usage: package.sh MODEL [-o FILE] [--weights NAME] [--no-weights] [--jobs N]

  MODEL           a corpus document, e.g. qwen3.8-27b-text
  -o FILE         output archive (default: tensorspine-MODEL-YYYYMMDD.tar.bz2)
  --weights NAME  the artifact directory under $TENSORSPINE_MODEL_ARTIFACTS/weights;
                  by default the one the test harness maps this document to
  --no-weights    package the binary and the document only — for a target that already
                  has the weights, and the difference between 4 GB and 56 GB
  --jobs N        compressor threads (default: all cores, when pbzip2 or lbzip2 is here)

  $ZML_HOME and $TENSORSPINE_MODEL_ARTIFACTS must be set; see generators/zml/README.md.
USAGE
    exit 2
}

model=''; out=''; artifact=''; with_weights=1; jobs=''
while [ $# -gt 0 ]; do
    case "$1" in
        -o) out="${2:?-o needs a path}"; shift 2 ;;
        --weights) artifact="${2:?--weights needs a name}"; shift 2 ;;
        --no-weights) with_weights=0; shift ;;
        --jobs) jobs="${2:?--jobs needs a number}"; shift 2 ;;
        -h|--help) usage ;;
        -*) echo "package.sh: unknown option $1" >&2; usage ;;
        *) [ -z "$model" ] || usage; model="$1"; shift ;;
    esac
done
[ -n "$model" ] || usage
[ -n "${ZML_HOME:-}" ] || { echo "package.sh: \$ZML_HOME is not set" >&2; exit 1; }
[ -n "${TENSORSPINE_MODEL_ARTIFACTS:-}" ] || { echo "package.sh: \$TENSORSPINE_MODEL_ARTIFACTS is not set" >&2; exit 1; }

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/../.." && pwd)

binary="$ZML_HOME/bazel-bin/external/+local_repository+tensorspine/tspl"
runfiles="$binary.runfiles"
derived="$TENSORSPINE_MODEL_ARTIFACTS/derived/$model.derived.json"

[ -x "$binary" ] || { echo "package.sh: no binary at \$ZML_HOME/bazel-bin/... — build it first (README §1)" >&2; exit 1; }
[ -d "$runfiles" ] || { echo "package.sh: no runfiles beside the binary; tspl cannot start without them" >&2; exit 1; }
[ -f "$derived" ] || { echo "package.sh: no $model.derived.json — derive it first (README §2)" >&2; exit 1; }

# The document-to-artifact mapping lives in the test harness; read it rather than keep a
# second copy that can drift.
if [ -z "$artifact" ] && [ "$with_weights" -eq 1 ]; then
    artifact=$(python3 - "$root" "$model" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + '/generators/zml/tests')
import run_zml
name = run_zml.WEIGHTS.get(sys.argv[2])
if not name:
    sys.exit(f"package.sh: no artifact known for '{sys.argv[2]}' — pass --weights NAME")
print(name)
PY
    )
fi

stamp=$(date -u +%Y%m%d)
[ -n "$out" ] || out="tensorspine-$model-$stamp.tar.bz2"
name="tensorspine-$model"

# A parallel bzip2 writes the same format and is the difference between minutes and
# hours on tens of gigabytes; plain bzip2 is the fallback and says so.
if command -v lbzip2 >/dev/null 2>&1; then   compressor=lbzip2
elif command -v pbzip2 >/dev/null 2>&1; then compressor=pbzip2
else                                         compressor=bzip2
     echo "package.sh: only plain bzip2 here — single-threaded, and slow on weights." >&2
fi
[ "$compressor" = bzip2 ] || compressor="$compressor -p${jobs:-$(nproc 2>/dev/null || echo 4)}"

# Bazel's runfiles are read-only, and a copy of them keeps those modes; without the
# chmod the cleanup cannot remove its own staging directory.
staging=$(mktemp -d "${TMPDIR:-/tmp}/tspl-package-XXXXXX")
trap 'chmod -R u+w "$staging" 2>/dev/null || :; rm -rf "$staging"' EXIT INT TERM
stage="$staging/$name"
mkdir -p "$stage/artifacts/derived"

echo "packaging $model"
cp "$binary" "$stage/tspl"
chmod +x "$stage/tspl"
cp "$derived" "$stage/artifacts/derived/"

# -h: the runfiles are symlinks into a Bazel cache, and the cache is not going with us.
echo "  runfiles ($(du -shL "$runfiles" | cut -f1))"
tar -C "$(dirname "$runfiles")" -chf - "$(basename "$runfiles")" | tar -C "$stage" -xf -

if [ "$with_weights" -eq 1 ]; then
    weights="$TENSORSPINE_MODEL_ARTIFACTS/weights/$artifact"
    [ -d "$weights" ] || { echo "package.sh: no weights at weights/$artifact" >&2; exit 1; }
    echo "  weights $artifact ($(du -shL "$weights" | cut -f1))"
    mkdir -p "$stage/artifacts/weights"
    tar -C "$(dirname "$weights")" -chf - "$artifact" | tar -C "$stage/artifacts/weights" -xf -
fi

cat > "$stage/run.sh" <<RUN
#!/bin/sh
# Launch the packaged model. Every path is relative to this script, so the package runs
# from wherever it was unpacked and needs nothing set in the environment.
set -eu
here=\$(CDPATH= cd -- "\$(dirname -- "\$0")" && pwd)
exec "\$here/tspl" \\
    --derived="\$here/artifacts/derived/$model.derived.json" \\
    ${with_weights:+--checkpoint="\$here/artifacts/weights/$artifact"} \\
    "\$@"
RUN
chmod +x "$stage/run.sh"

# Bazel's runfiles are read-only; carried into the archive they leave the recipient a
# tree they cannot delete without a chmod of their own.
chmod -R u+w "$stage"

cuda=no; [ -d "$stage/tspl.runfiles" ] && ls "$stage/tspl.runfiles" | grep -q cuda && cuda=yes

cat > "$stage/MANIFEST" <<MANIFEST
model            $model
artifact         ${artifact:-(not packaged)}
cuda plugin      $cuda
built from       $(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)
packaged         $(date -u +%Y-%m-%dT%H:%M:%SZ)
tspl sha256      $(sha256sum "$stage/tspl" | cut -d' ' -f1)

Unpack with a parallel bzip2 — plain \`tar -xf\` decompresses on one thread and takes
tens of minutes on an archive this size:

    tar --use-compress-program='pbzip2 -d' -xf ARCHIVE     # or lbzip2 -d

Then run ./run.sh from the unpacked directory; it takes tspl's own options:

    ./run.sh --max-tokens=8 --compute=bf16            # generate
    ./run.sh --chat --compute=bf16 --capacity=512
    ./run.sh --until='decoder/mlp_r[layer=0].output' --out=v.bin

The first line of output names the backend it chose. It reads 'platform: cuda' only on
a machine with a device this plugin supports; anything else falls back to the CPU and
says so.

Check the unpack before trusting what comes out of it. A transfer or an extraction that
dropped a file leaves a package that still starts, still loads and still answers — with
nonsense, and with nothing in the log to say why:

    sha256sum -c SHA256SUMS
MANIFEST

# The weights are the bulk of the archive and the part whose corruption is silent: a
# missing shard stops the loader, but a truncated one does not, and neither does a
# runfile the extraction never wrote. Hashing the tree costs a minute here and a minute
# on the far side; not being able to tell a bad unpack from a bad port costs a day.
echo "  hashing $(du -shL "$stage" | cut -f1)"
(cd "$stage" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

echo "  compressing with ${compressor%% *}"
tar -C "$staging" -cf - "$name" | $compressor > "$out"
echo "$out ($(du -sh "$out" | cut -f1))"
