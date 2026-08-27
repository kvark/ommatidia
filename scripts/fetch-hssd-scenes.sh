#!/usr/bin/env bash
# Fetch a bounded, renderer-compatible furnished HSSD subset.
set -euo pipefail

root="${1:-data/catalog}"
shift || true
specs=("$@")
if (( ${#specs[@]} == 0 )); then
  specs=(train:102343992 holdout:102344022)
fi

scenes=()
splits=()
for index in "${!specs[@]}"; do
  spec="${specs[$index]}"
  if [[ "$spec" == *:* ]]; then
    split="${spec%%:*}"
    scene="${spec#*:}"
  else
    # Preserve the original positional shorthand: the first scene trains and
    # every later scene is held out.
    (( index == 0 )) && split=train || split=holdout
    scene="$spec"
  fi
  if [[ "$split" != train && "$split" != holdout ]]; then
    echo "invalid HSSD split in $spec (use train:ID or holdout:ID)" >&2
    exit 2
  fi
  scenes+=("$scene")
  splits+=("$split")
done

hab_files=(README.md)
for scene in "${scenes[@]}"; do
  hab_files+=("stages/$scene.glb" "scenes/$scene.scene_instance.json")
done
hf download hssd/hssd-hab --repo-type dataset "${hab_files[@]}" \
  --local-dir "$root/hssd"

templates=()
declare -A template_splits=()
for index in "${!scenes[@]}"; do
  scene="${scenes[$index]}"
  split="${splits[$index]}"
  mapfile -t current < <(
    jq -r '.object_instances[].template_name' \
      "$root/hssd/scenes/$scene.scene_instance.json"
  )
  for template in "${current[@]}"; do
    previous="${template_splits[$template]:-}"
    if [[ -n "$previous" && "$previous" != "$split" ]]; then
      echo "HSSD object $template occurs in both $previous and $split scenes" >&2
      exit 1
    fi
    template_splits[$template]="$split"
  done
  templates+=("${current[@]}")
done

mapfile -t unique < <(printf '%s\n' "${templates[@]}" | sort -u)
model_files=()
for template in "${unique[@]}"; do
  # Decomposed `_part_` instances are absent from the public uncompressed
  # model repository. The loader reports and omits those instances.
  [[ "$template" == *_part_* ]] && continue
  model_files+=("objects/${template:0:1}/$template.glb")
done
hf download hssd/hssd-models --repo-type dataset "${model_files[@]}" \
  --local-dir "$root/hssd-uncompressed"

catalog="$root/catalog.json"
[[ -f "$catalog" ]] || printf '{"entries":[]}\n' > "$catalog"
for index in "${!scenes[@]}"; do
  scene="${scenes[$index]}"
  split="${splits[$index]}"
  tmp="$(mktemp "$root/catalog.XXXXXX")"
  jq --arg id "hssd/$scene" --arg scene_id "$scene" --arg split "$split" '
    .entries |= (map(select(.id != $id)) + [{
      id: $id,
      source: "hssd",
      license: "CC-BY-NC-4.0",
      kind: "interior",
      family: $id,
      split: $split,
      path: ("hssd/stages/" + $scene_id + ".glb"),
      scene: ("hssd/scenes/" + $scene_id + ".scene_instance.json"),
      object_root: "hssd-uncompressed/objects"
    }])
  ' "$catalog" > "$tmp"
  mv "$tmp" "$catalog"
done

printf 'fetched %d scene(s) and %d requested object models under %s; updated %s\n' \
  "${#scenes[@]}" "${#model_files[@]}" "$root" "$catalog"
