#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 2 ]] || { echo "usage: $0 <category> <video> [offset] [frames] [width]" >&2; exit 2; }

category=$1 video=$2 offset=${3:-0} frames=${4:-0} width=${5:-160}
root=$(cd "$(dirname "$0")/.." && pwd)
dir=$root/data/$video
url=https://huggingface.co/datasets/tauseefakhund/D2-CDNet/resolve/main/raw_tar/$category.tar

if [[ ! -f $dir/.complete ]]; then
    mkdir -p "$dir"
    fifo=$dir/.stream
    rm -f "$fifo" && mkfifo "$fifo"
    tar -x -C "$dir" --strip-components=3 --wildcards \
        "*/$video/input/*" "*/$video/groundtruth/*" "*/$video/temporalROI.txt" < "$fifo" 2>/dev/null &
    extractor=$!
    curl -sfL "$url" | tee "$fifo" | tar -t 2>/dev/null \
        | { seen=0; while read -r name; do
                if [[ $name == */$video/* ]]; then seen=1; elif (( seen )); then break; fi
            done; } || true
    wait "$extractor" || true
    rm -f "$fifo"
    [[ -f $dir/temporalROI.txt ]] || { echo "$video not found in $category" >&2; exit 1; }
    touch "$dir/.complete"
fi

read -r first last < "$dir/temporalROI.txt" || [[ -n $last ]]
first=$(( first + offset ))
(( first <= last )) || { echo "offset is past the ROI end" >&2; exit 1; }
(( frames > 0 && frames < last - first + 1 )) || frames=$(( last - first + 1 ))
IFS=, read -r W H < <(ffprobe -v error -select_streams v -show_entries stream=width,height \
    -of csv=p=0 "$(printf '%s/input/in%06d.jpg' "$dir" "$first")")
height=$(( (width * H / W + 1) / 2 * 2 ))

ffmpeg -v error -y -start_number "$first" -i "$dir/input/in%06d.jpg" -frames:v "$frames" \
    -vf "scale=$width:$height:flags=area,format=gray" -f rawvideo "$dir/frames.u8"
ffmpeg -v error -y -start_number "$first" -i "$dir/groundtruth/gt%06d.png" -frames:v "$frames" \
    -vf "scale=$width:$height:flags=neighbor,format=gray" -f rawvideo "$dir/gt.u8"

echo "$width $height $frames" > "$dir/size.txt"
echo "$video: $frames frames of ${width}x$height from frame $first (ROI ends $last)"
