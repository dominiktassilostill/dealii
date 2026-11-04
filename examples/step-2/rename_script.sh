#!/usr/bin/env bash

for file in error_CG_SpMV_Mixed_p_*.tex; do
    [[ -e "$file" ]] || continue

    newfile="${file/error_CG_SpMV_Mixed_p_/error_CG_SpMV_MixedPyramid_p_}"

    mv -- "$file" "$newfile"
    echo "Renamed: $file -> $newfile"
done

