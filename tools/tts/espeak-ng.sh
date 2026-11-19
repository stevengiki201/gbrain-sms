#!/usr/bin/env bash
# Runs the project-local (rootless) espeak-ng install.
# Usage: espeak-ng.sh <out.wav> <text>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$HERE/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ESPEAK_DATA_PATH="$HERE/root/usr/lib/x86_64-linux-gnu/espeak-ng-data"
exec "$HERE/root/usr/bin/espeak-ng" -w "$1" "$2"
