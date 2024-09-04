#!/bin/sh

set -e
shopt -s extglob

profile_and_save_res()
{
    if [ ${filename: -6} == ".synap" ]
        then
            output=$(synap_cli -m "$filename" -r 10 random)
            result=$(echo "$output" | tail -n 1)
            base_name=$(basename ${filename})
            echo "$base_name: $result" >> "$results_f"
    fi
}

results_f="results.txt"
if [ -e "$results_f" ]; then
    rm "$results_f"
fi
touch "$results_f"

if [ $# -eq 1 ] && [ -f $1 ]; then
    filename=$1
    profile_and_save_res
elif [ $# -eq 1 ] && [ -d $1 ]; then
    for filename in $1/*; do
        profile_and_save_res
    done
elif [ $# -gt 1 ]; then
    for filename in "$@"; do
        profile_and_save_res
    done
else
    echo "No models found for benchmarking"
fi
