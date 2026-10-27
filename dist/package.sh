#!/bin/sh
# Packages a build for release: bin/ (mako_serve, mako_calibrate; stripped and ad-hoc signed again), README.md,
# LICENSE, THIRD_PARTY_LICENSES.txt and examples/presets.json in <out dir>/<name>/, then <name>.tar.gz with its
# .sha256 and the programs' debug symbols in <name>-dSYM.tar.gz.
#   dist/package.sh <build dir> <out dir> <name>
set -eu

if [ $# -ne 3 ]; then
    echo "usage: dist/package.sh <build dir> <out dir> <name>" >&2
    exit 2
fi
build=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
name=$3
src=$(cd "$(dirname "$0")/.." && pwd)
commit=$(git -C "$src" rev-parse --short=12 HEAD)

pkg=$out/$name
rm -rf "$pkg" "$out/$name-dSYM"
mkdir -p "$pkg/bin" "$pkg/examples" "$out/$name-dSYM"

for tool in mako_serve mako_calibrate; do
    cp "$build/tools/$tool" "$pkg/bin/"
    dsymutil "$pkg/bin/$tool" -o "$out/$name-dSYM/$tool.dSYM"
    strip "$pkg/bin/$tool"
    codesign --force --sign - "$pkg/bin/$tool" # stripping invalidates the linker's ad-hoc signature
done

sed -e "s/@NAME@/$name/g" -e "s/@COMMIT@/$commit/g" "$src/dist/README.md" > "$pkg/README.md"
cp "$src/LICENSE" "$pkg/LICENSE"
cp "$src/docs/presets.example.json" "$pkg/examples/presets.json"

# Each license from the copy the build downloaded, under the component and the version the build pins.
version() { sed -n "$2" "$src/$1" | head -n 1; }
llama=$(version cmake/llama_cpp_sources.cmake 's/^set(LLAMA_CPP_RELEASE \(.*\))$/\1/p')
httplib=$(version cmake/server_deps.cmake 's/.*cpp-httplib\/\(v[0-9.]*\)\/.*/\1/p')
glaze=$(version cmake/server_deps.cmake 's/.*glaze\/archive\/refs\/tags\/\(v[0-9.]*\)\.tar\.gz.*/\1/p')
metal_cpp=$(version CMakeLists.txt 's/.*metal-cpp_\([0-9.]*\)\.zip.*/\1/p')
{
    echo "Third-party code built into mako, with its licenses."
    for entry in \
        "llama.cpp $llama (GEMV kernels, Jinja engine; MIT)|$build/third_party/gemv-kernels/LICENSE" \
        "cpp-httplib $httplib (MIT)|$build/third_party/httplib/LICENSE" \
        "glaze $glaze (MIT)|$build/_deps/glaze-src/LICENSE" \
        "metal-cpp $metal_cpp (Apache License 2.0)|$build/_deps/metal_cpp-src/LICENSE.txt"; do
        title=${entry%%|*}
        file=${entry#*|}
        [ -n "$title" ] && [ -f "$file" ] || { echo "missing license: $entry" >&2; exit 1; }
        printf '\n\n==== %s ====\n\n' "$title"
        cat "$file"
    done
} > "$pkg/THIRD_PARTY_LICENSES.txt"
grep -q '^==== llama.cpp b[0-9]' "$pkg/THIRD_PARTY_LICENSES.txt" || { echo "no llama.cpp version" >&2; exit 1; }

cd "$out"
tar czf "$name.tar.gz" "$name"
shasum -a 256 "$name.tar.gz" > "$name.tar.gz.sha256"
tar czf "$name-dSYM.tar.gz" "$name-dSYM"
echo "$out/$name.tar.gz"
