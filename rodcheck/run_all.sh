#!/bin/sh
# Usage: sh run_all.sh [path to the serial crystal executable]
CRYSTAL="${1:-crystal}"
cd "$(dirname "$0")"
for deck in rod_*.d12; do
  "$CRYSTAL" < "$deck" > "${deck%.d12}.out" 2>&1
done
