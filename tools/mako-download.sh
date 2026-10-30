#!/bin/sh
# Downloads a model mako runs from Hugging Face, at the repository revisions this version of mako was tested with,
# and checks each file's size and SHA-256. Interrupted downloads resume; files already complete are kept.
#   tools/mako-download.sh <model> [--vision] [--dspark] [--ablate] [--dir DIR] [--no-verify] [--recheck] [--dry-run]
#                          [--curl]
# With the hf CLI when it is installed (Xet storage, its saved login), else with curl. HF_TOKEN, when set, is sent with
# each request.
set -eu

usage() {
    cat >&2 <<'EOF'
usage: mako-download.sh <model> [--vision] [--dspark] [--ablate] [--dir DIR] [--no-verify] [--recheck] [--dry-run]
                        [--curl]

models:
  glm           GLM-5.3-Flash, Maya-S-v2 IQ2_XXS (96.5 GB)
  ds            DeepSeek-V4.1-Flash-1017, Maya-S IQ2_XXS with its Engram tables (339 GB)
  ds-ista-xs    DeepSeek-V4.1-Flash-1017, ISTA GSQ-RCO IQ2_XS and the IQ4_NL Engram file (266 GB)
  ds-ista-xxs   DeepSeek-V4.1-Flash-1017, ISTA GSQ-RCO IQ2_XXS and the IQ4_NL Engram file (249 GB)

  --vision      the vision projector (1.0-1.2 GB; mako_serve --mmproj)
  --dspark      DeepSeek's DSpark drafter (8 GB, in DIR/dspark; mako_serve --dspark)
  --ablate      the model's refusal ablation file (3-4 MB; mako_serve --ablate)
  --dir DIR     where to put the files (default ~/models/<model>)
  --no-verify   skip the SHA-256 check (sizes are still checked)
  --recheck     also check the SHA-256 of files already in DIR
  --dry-run     list the files and the space needed, download nothing
  --curl        download with curl even when the hf CLI is installed
EOF
    exit 2
}

# Repositories and the revisions the files below are from.
repo_glm=peasantsmith/GLM-5.3-Flash-Maya-GGUF@f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da
repo_ds=peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF@d1fc01009f21e26157acb6858c315d0365483b72
repo_unsloth=unsloth/GLM-5.3-Flash-GGUF@a38483c8cd5df544f53d70fb281afe97369d5ab6
repo_ista=ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF@a419d74cd28d5b9f8c5e2b9077d11a609c17bc32
repo_abl=skafte/mako-ablations@b5a4188af1eb61375ebe2caf38f14d453052419d

# group repo path size sha256
manifest() {
    cat <<'EOF'
glm glm Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf 47835993056 e56ef50efd71d81ddf6249e08bd2ef5af96ed348c2b7a004491d46168d0ecaa7
glm glm Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00002-of-00003.gguf 47797478080 2d65d88a69f8dc124c8d24bd33b33161ddab218918b4253ad4f500aeede84df5
glm glm Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00003-of-00003.gguf 835121120 a6a981c4fee7a53d78bdbd97d8f465d48ddf439cf938ff90617f395e0351b5a6
glm-vision unsloth mmproj-BF16.gguf 1164010176 82b7adb44411dbb298b1c689567804e5ed0790d80f308ed8b5b6b8ac0527d2ac
ds ds Maya-S/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00001-of-00006.gguf 47887697312 1f3437f57042e69f94c141ffbcba0d5610dab63feb07244fb468be4940d6763f
ds ds Maya-S/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00002-of-00006.gguf 47669578400 66501fee8702ffd54e6f065323ab54bb3150d20b10b01922c589e7462c7b5ba6
ds ds Maya-S/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00003-of-00006.gguf 47103347360 3195b23f33337c46a07c48583946d99147c3635412bc4adc6acdc4cf42bf30ad
ds ds Maya-S/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00004-of-00006.gguf 35532532032 d60dba35fac9601cd5c6026439dd980c457a5bb1c009192ea539b220108ea0db
ds ds Maya-S/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00005-of-00006.gguf 80641295472 c98d46f96a25d17c4c54da5a60782c130255c364a22e621ea28d5c98a692ceb2
ds ds Maya-S/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00006-of-00006.gguf 80643503412 98d65ef4f36e0b193935d44c3586241ca09b1b47e90f118e85332cfdb5860263
dspark ds dspark/dspark.bin 7967616768 42df65159f980a9189d80597e42b30012c7b10e0dd385a47051f3b2fd659bb7b
dspark ds dspark/dspark.json 96965 639c3144bca8cfe4db59057047dca40411f0cc732796e2a9321feb98fee2295f
glm-abl abl glm-5.3-flash/refusal-3dir.abl 4423736 d75f26455b9dd26a99eb604ad21b233485da6f181ae922c47f417208bbde33c5
ds-abl abl deepseek-v4.1-flash-1017/refusal.abl 3276856 b8c10979b684ea4c84c3146d69bef5313cd84a21e6391f2e1b97a9fabe094994
ds-vision ista mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf 972208096 0fdc4b7693f287e9665940cf42bd39c546ffe4f0def89c569d7d0e45d60d2df1
ista-xs ista DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf 5780704 0182a2a1f42d319a0dbc6e34aa82a063880c069fe3aafbbea86018f386e64e5f
ista-xs ista DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00002-of-00006.gguf 39581533600 5c88039fa640460a9fc3465a6dfcd162a4083dd738d55e5b52e025be14555a08
ista-xs ista DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00003-of-00006.gguf 39417972640 0fddebfd05f0a36cfd107ba5dfd2547e64df61adbab39560a5a5f5521ca099a5
ista-xs ista DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00004-of-00006.gguf 39079187648 c577a43c6e6e615838a1655a0b162541edeac6f36542f08c4a9593a70dcaf500
ista-xs ista DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00005-of-00006.gguf 37141423776 851d8605bf504962054d17ea8a0d9e2e0a228d085d396bd865a4de9f16d90fbd
ista-xxs ista DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf 5780704 f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9
ista-xxs ista DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf 39535781184 25af3785ccf925f54137e1ead4c481a8952315dc0b1af6519a99407f67bce8f3
ista-xxs ista DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf 39736712000 8168d0998fc494def2e1952d83c131425a108660aae9d29190f5a8670ab62b48
ista-xxs ista DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf 39036908384 9eb9b4de1968f150dfc207eb31c27be9cd9b5666a9a6246bde8a22b184b4bee0
ista-xxs ista DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf 19923784832 7e35f6b4e6fc46f6c4b152b593d84d683c5ece695ae6ad734f556198c041bec1
ista-engram ista DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf 110595290816 3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3
EOF
}

[ $# -ge 1 ] || usage
model=$1
shift
vision=0 dspark=0 ablate=0 verify=1 recheck=0 dry=0 dir= curl=0
while [ $# -gt 0 ]; do
    case $1 in
    --curl) curl=1 ;;
    --vision) vision=1 ;;
    --dspark) dspark=1 ;;
    --ablate) ablate=1 ;;
    --no-verify) verify=0 ;;
    --recheck) recheck=1 ;;
    --dry-run) dry=1 ;;
    --dir) [ $# -ge 2 ] || usage; dir=$2; shift ;;
    *) usage ;;
    esac
    shift
done

case $model in
glm) groups=glm vgroup=glm-vision agroup=glm-abl link= ;;
ds) groups=ds vgroup=ds-vision agroup=ds-abl link= ;;
ds-ista-xs) groups="ista-xs ista-engram" vgroup=ds-vision agroup=ds-abl link=DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS ;;
ds-ista-xxs) groups="ista-xxs ista-engram" vgroup=ds-vision agroup=ds-abl link=DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS ;;
*) usage ;;
esac
[ $vision = 1 ] && groups="$groups $vgroup"
[ $ablate = 1 ] && groups="$groups $agroup"
if [ $dspark = 1 ]; then
    case $model in
    glm) echo "--dspark: GLM-5.3 has no DSpark drafter (it drafts with its own MTP layer)" >&2; exit 2 ;;
    ds-ista-*) echo "note: the DSpark drafter has not been tried with the ISTA builds" >&2 ;;
    esac
    groups="$groups dspark"
fi
dir=${dir:-$HOME/models/$model}

files=$(for want in $groups; do
    manifest | while read -r g repo path size sha; do
        if [ "$g" = "$want" ]; then echo "$repo $path $size $sha"; fi
    done
done)

# A file's place in DIR: the drafter keeps its folder, everything else goes in DIR itself.
dest_of() {
    case $1 in
    dspark/*) echo "$dir/$1" ;;
    *) echo "$dir/${1##*/}" ;;
    esac
}

fsize() { stat -L -f %z "$1" 2>/dev/null || echo 0; }

need=0 total=0
while read -r repo path size sha; do
    dest=$(dest_of "$path")
    total=$((total + size))
    if [ "$(fsize "$dest")" != "$size" ]; then
        need=$((need + size - $(fsize "$dest.part")))
    fi
done <<EOF
$files
EOF

gb() { echo "$1" | awk '{ printf "%.1f GB", $1 / 1e9 }'; }
echo "$model: $(echo "$files" | wc -l | tr -d ' ') files, $(gb $total), into $dir; $(gb $need) left to download"
mkdir -p "$dir"
free=$(($(df -k "$dir" | awk 'NR == 2 { print $4 }') * 1024))
if [ $need -gt $free ]; then
    echo "not enough space: $(gb $need) needed, $(gb $free) free on $dir" >&2
    exit 1
fi

if [ $dry = 1 ]; then
    echo "$files" | while read -r repo path size sha; do
        printf '  %-70s %s\n' "$(dest_of "$path")" "$(gb $size)"
    done
    exit 0
fi

check() { # file sha; a mismatch renames it to <name>.bad
    echo "     checking SHA-256"
    got=$(openssl dgst -sha256 -r "$1" | cut -d ' ' -f 1)
    if [ "$got" != "$2" ]; then
        bad=$dest.bad
        mv "$1" "$bad"
        echo "${bad##*/}: SHA-256 $got, expected $2" >&2
        exit 1
    fi
}

download() { # repo path size sha
    eval "spec=\$repo_$1"
    url="https://huggingface.co/${spec%@*}/resolve/${spec#*@}/$2"
    dest=$(dest_of "$2")
    part=$dest.part
    if [ "$(fsize "$dest")" = "$3" ]; then
        echo "have ${dest##*/}"
        if [ $recheck = 1 ]; then check "$dest" "$4"; fi
        return
    fi
    mkdir -p "$(dirname "$dest")"
    if [ -n "$hf" ]; then
        # hf keeps the repository's folders under --local-dir (and its resume state in .cache there), so the file is
        # staged in DIR/.hf and moved into place once checked.
        part=$hf/$2
        echo "get  ${dest##*/} ($(gb $3))"
        hf download "${spec%@*}" "$2" --revision "${spec#*@}" --local-dir "$hf" >/dev/null
    else
        [ "$(fsize "$part")" -gt "$3" ] && rm -f "$part"
        if [ "$(fsize "$part")" != "$3" ]; then
            echo "get  ${dest##*/} ($(gb $3))"
            set -- "$@" "$url" "$part"
            if [ -n "${HF_TOKEN:-}" ]; then
                curl -fL --retry 10 --retry-delay 5 --retry-all-errors -C - -H "Authorization: Bearer $HF_TOKEN" \
                    -o "$6" "$5"
            else
                curl -fL --retry 10 --retry-delay 5 --retry-all-errors -C - -o "$6" "$5"
            fi
        fi
    fi
    if [ "$(fsize "$part")" != "$3" ]; then
        echo "${dest##*/}: $(fsize "$part") bytes, expected $3" >&2
        exit 1
    fi
    if [ $verify = 1 ]; then check "$part" "$4"; fi
    mv "$part" "$dest"
}

hf=
if [ $curl = 0 ] && command -v hf >/dev/null 2>&1; then
    hf=$dir/.hf
    echo "downloading with $(command -v hf)"
fi
while read -r repo path size sha; do
    download "$repo" "$path" "$size" "$sha" </dev/null
done <<EOF
$files
EOF
if [ -n "$hf" ]; then rm -rf "$hf"; fi

# The ISTA builds open the shared Engram file as their last shard.
if [ -n "$link" ]; then
    ln -sf DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf "$dir/$link-00006-of-00006.gguf"
fi

first=$(echo "$files" | awk 'NR == 1 { print $2 }')
first=$(dest_of "$first")
here=$(cd "$(dirname "$0")" && pwd)
bin=
for b in "$here" "$here/../build/tools"; do
    [ -x "$b/mako_serve" ] && bin=$(cd "$b" && pwd)/ && break
done
extra=
[ $vision = 1 ] && extra="$extra --mmproj $(dest_of "$(echo "$files" | awk '/mmproj/ { print $2 }')")"
[ $dspark = 1 ] && extra="$extra --dspark $dir/dspark"
[ $ablate = 1 ] && extra="$extra --ablate $(dest_of "$(echo "$files" | awk '/\.abl / { print $2 }')")"
case $model in ds-ista-*) extra="$extra --miss-penalty 0.1" ;; esac

echo
echo "done. Once per model (which experts start in memory):"
echo "  ${bin}mako_calibrate $first"
echo "then:"
echo "  ${bin}mako_serve $first$extra --port 8080 --ctx 65536"
echo "or as a preset (~/Library/Application Support/mako/presets.json, --preset $model):"
printf '  "%s": {\n    "model": "%s"' "$model" "$first"
set -- $extra
while [ $# -ge 2 ]; do
    case $2 in
    [0-9]*) printf ',\n    "%s": %s' "$1" "$2" ;;
    *) printf ',\n    "%s": "%s"' "$1" "$2" ;;
    esac
    shift 2
done
printf '\n  }\n'
