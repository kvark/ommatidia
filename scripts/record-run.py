#!/usr/bin/env python3
"""Run a command once, retaining its log, resolved source revisions and input hashes.

Example:
  python3 scripts/record-run.py runs/baseline --input data/train.omd -- \
    cargo run --release -p ommatidia-train --locked -- --data data/train.omd ...

The output directory must not exist. Checkpoints still need an explicit --out.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--input", type=Path, action="append", default=[])
    if "--" not in sys.argv:
        parser.error("separate the command with --")
    boundary = sys.argv.index("--")
    args = parser.parse_args(sys.argv[1:boundary])
    command = sys.argv[boundary + 1:]
    if not command:
        parser.error("missing command")
    for path in args.input:
        if not path.is_file():
            parser.error(f"missing input: {path}")
    args.directory.mkdir(parents=True, exist_ok=False)
    manifest_path = args.directory / "manifest.json"
    manifest = {
        "started": now(), "cwd": str(Path.cwd()), "command": command,
        "status": "preparing", "recorder_sha256": sha256(Path(__file__)),
    }

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        shutil.copyfile(__file__, args.directory / "recorder.py")
        manifest["inputs"] = []
        for index, path in enumerate(args.input):
            item = {"path": str(path.resolve()), "bytes": path.stat().st_size,
                    "sha256": sha256(path)}
            if item["bytes"] <= 1024 * 1024:
                item["snapshot"] = f"input-{index}-{path.name}"
                shutil.copyfile(path, args.directory / item["snapshot"])
            manifest["inputs"].append(item)
        toolchain = []
        if command[0] == "cargo" and len(command) > 1 and command[1].startswith("+"):
            toolchain = command[1:2]
        manifest["rustc"] = capture(["rustc", *toolchain, "-vV"])
        manifest["cargo_lock_sha256"] = sha256(Path("Cargo.lock"))
        # Cargo.toml git pins can be overridden by sibling path patches.
        metadata = json.loads(capture([
            "cargo", *toolchain, "metadata", "--locked", "--format-version", "1",
        ]))
        manifest["packages"] = []
        manifest["repositories"] = {}
        selected = {"ommatidia", "blade-graphics", "blade-render", "meganeura", "naga"}
        for package in metadata["packages"]:
            if package["name"] not in selected:
                continue
            source_dir = Path(package["manifest_path"]).parent
            root = capture(["git", "rev-parse", "--show-toplevel"], source_dir)
            manifest["packages"].append({
                key: package[key] for key in ("name", "version", "source", "manifest_path")
            })
            if root in manifest["repositories"]:
                continue
            patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
            patch_name = f"source-{len(manifest['repositories'])}.patch"
            (args.directory / patch_name).write_bytes(patch)
            manifest["repositories"][root] = {
                "head": capture(["git", "rev-parse", "HEAD"], root),
                "status": capture(["git", "status", "--porcelain"], root),
                "tracked_diff": patch_name,
                "tracked_diff_sha256": hashlib.sha256(patch).hexdigest(),
            }
        manifest["environment"] = {
            key: value for key, value in os.environ.items()
            if key.startswith(("MEGANEURA_", "OMMATIDIA_", "VK_", "CARGO_PROFILE_"))
            or key in ("RUSTFLAGS", "RUST_LOG", "RUSTC", "RUSTC_WRAPPER",
                       "RUSTC_WORKSPACE_WRAPPER", "RUSTUP_TOOLCHAIN", "CARGO_TARGET_DIR",
                       "CARGO_BUILD_TARGET", "CARGO_ENCODED_RUSTFLAGS")
        }
        manifest["status"] = "running"
        save()
        validation_errors = 0
        with (args.directory / "output.log").open("w") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                try:
                    for line in process.stdout:
                        validation_errors += "Validation Error:" in line
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
        manifest["exit_code"] = code
        manifest["validation_errors"] = validation_errors
        manifest["status"] = "complete" if code == 0 and not validation_errors else "failed"
        return code if code != 0 else int(validation_errors != 0)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished"] = now()
        save()


if __name__ == "__main__":
    sys.exit(main())
