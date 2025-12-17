#!/bin/bash
# collect.sh - wrapper to collect samples

label=""
count=100
out="data.npz"
cam=0
mediapipe=false
pi_camera=false
swap_rb=false
script="sign_trans.py"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --label) label="$2"; shift ;;
    --count) count="$2"; shift ;;
    --out) out="$2"; shift ;;
    --cam) cam="$2"; shift ;;
    --mediapipe) mediapipe=true ;;
    --pi-camera) pi_camera=true ;;
    --swap-rb) swap_rb=true ;;
    --script) script="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

if [[ -z "$label" ]]; then
  echo "Error: --label is required."
  exit 1
fi

args=( "python" "./$script" "collect" "--label" "$label" "--count" "$count" "--out" "$out" )

$mediapipe && args+=( "--mediapipe" )
$pi_camera && args+=( "--pi-camera" )
$swap_rb && args+=( "--swap-rb" )

# Only pass --cam for USB webcams
if ! $pi_camera; then
  args+=( "--cam" "$cam" )
fi

echo "Running: ${args[*]}"
"${args[@]}"
