#!/usr/bin/env python3
"""Prepare the pinned compiler with the tracked Vulkan workgroup-layout fix.

Idempotent for the exact patched tree; refuses to overwrite a different revision
or local changes. Does not modify Cargo's cache or an existing sibling checkout.
"""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "target/patched-naga"
REVISION = "323acfb729a00c3030e362faa27252afe6368792"
PATCH = ROOT / "patches/naga-workgroup-layout.patch"


def git(*args, check=True, input_bytes=None):
    return subprocess.run(["git", "-C", str(DESTINATION), *args], check=check,
                          input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def prepare():
    if not DESTINATION.exists():
        DESTINATION.mkdir(parents=True)
        git("init", "--quiet")
        git("config", "core.autocrlf", "false")
        git("remote", "add", "origin", "https://github.com/gfx-rs/wgpu")
        git("fetch", "--depth=1", "origin", REVISION)
        git("checkout", "--detach", REVISION)
    actual = git("rev-parse", "HEAD").stdout.decode().strip()
    if actual != REVISION:
        raise RuntimeError(f"refusing to modify {DESTINATION}: unexpected revision {actual}")
    # Intent-to-add lets the complete patch, including its new regression test,
    # be compared without staging file contents or committing in the dependency.
    test = "naga/tests/naga/spirv_workgroup_layout.rs"
    if (DESTINATION / test).is_file():
        git("add", "-N", "--", test)
    expected = PATCH.read_text().encode()
    diff = git("diff", "HEAD", "--binary", "--full-index").stdout
    status = git("status", "--porcelain").stdout
    if any(line.startswith(b"??") for line in status.splitlines()):
        raise RuntimeError(f"unexpected untracked files in {DESTINATION}; left untouched")
    if diff == expected:
        print(f"Compiler patch already prepared at {DESTINATION}")
        return
    if diff or status:
        raise RuntimeError(f"refusing to overwrite local changes in {DESTINATION}")
    git("apply", "--check", "-", input_bytes=expected)
    git("apply", "-", input_bytes=expected)
    git("add", "-N", "--", test)
    if git("diff", "HEAD", "--binary", "--full-index").stdout != expected:
        raise RuntimeError("applied compiler patch does not match the tracked patch")
    print(f"Prepared compiler patch at {DESTINATION}")


if __name__ == "__main__":
    try:
        prepare()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stderr.decode()) from error
