#!/bin/bash
# usage: count.sh keywords.tsv clean_dir  -> matrix.tsv (cat, count, keyword, hit files)
N=$(ls "$2"/*.md | wc -l)
while IFS=$'\t' read -r cat name re; do c=0; hits=""; for f in "$2"/*.md; do if /usr/bin/grep -qiE "$re" "$f"; then c=$((c+1)); hits="$hits $(basename "$f" .md | cut -c1-18)"; fi; done; printf "%s\t%d\t%s\t%s\n" "$cat" "$c" "$name" "$hits"; done < "$1" > matrix.tsv
echo "JDs=$N"; echo "== HARD"; awk -F'\t' -v n=$N '$1=="H"{printf "%2d  %3.0f%%  %s\n",$2,100*$2/n,$3}' matrix.tsv | sort -rn
echo "== GENERIC/SOFT"; awk -F'\t' -v n=$N '$1=="G"{printf "%2d  %3.0f%%  %s\n",$2,100*$2/n,$3}' matrix.tsv | sort -rn
