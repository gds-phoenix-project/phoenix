#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "Error: exactly two arguments are required."
  usage
fi

type="$1"
file_path="$2"

if [[ "$type" != "gds" && "$type" != "phxfs" ]]; then
  echo "Error: <type> must be 'gds' or 'phxfs'." >&2
  usage
fi

if [[ "$type" == "gds" ]]; then
  type_code=1
else                  # phxfs
  type_code=0
fi

echo "exec_path: $exec_path"

if [[ ! -f "$file_path" ]]; then
  echo "Error: data file not found - $file_path" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec_path="${SCRIPT_DIR}/../build/bin/breakdown"

if [[ ! -x "$exec_path" ]]; then
  echo "Error: executable not found - $exec_path" >&2
  exit 1
fi

io_sizes=(4 8 16 32 64 128 256 512 1024 2048 4096)

echo "Executable : $exec_path"
echo "Type       : $type"
echo "Data file  : $file_path"
echo "-----------------------------------------"

for size in "${io_sizes[@]}"; do
  echo "Running buffer size = ${size} KiB ..."
  "$exec_path" "$file_path" "$type_code" "$size"  10
done
