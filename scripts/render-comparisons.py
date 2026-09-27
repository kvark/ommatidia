#!/usr/bin/env python3
"""Render matched, unscaled evaluation PNGs as labeled videos and review sheets.

Requires ffmpeg with libx264/drawtext and ffprobe. Outputs are presentation assets, never
metric inputs. Reference identity is checked for every frame before rendering.
The output directory must be new; no existing results are overwritten.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_pages(frames):
    """Four consecutive native-resolution comparisons per page, without gaps."""
    return [list(range(start, min(start + 4, frames))) for start in range(0, frames, 4)]


def verify_video(path, frames, fps):
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of", "json", str(path)], text=True))
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one comparison video stream: {path}")
    stream = streams[0]
    if ([stream.get("width"), stream.get("height")] != [768, 288]
            or int(stream.get("nb_read_frames", 0)) != frames
            or Fraction(stream.get("avg_frame_rate", "0")) != fps):
        raise ValueError(f"encoded dimensions, frame count or playback rate differ: {stream}")
    return stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequences", type=int, nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--review-sheets", action="store_true",
                        help="Include every frame in consecutive four-frame review pages")
    args = parser.parse_args()
    if args.frames != 64 or args.fps != 24:
        parser.error("the current publication protocol requires 64 frames at 24 fps")
    if len(set(args.sequences)) != len(args.sequences) or min(args.sequences) < 0:
        parser.error("sequence ids must be distinct and nonnegative")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
              "frames": args.frames, "fps": args.fps, "size": [768, 288],
              "metric_source": False, "sequences": []}
    for sequence in args.sequences:
        sources = [(args.before, "learned"), (args.after, "learned"), (args.after, "reference")]
        streams = [hashlib.sha256() for _ in sources]
        for frame in range(args.frames):
            prefix = f"{sequence:03}-{frame:03}"
            if digest(args.before / f"{prefix}-reference.png") != digest(args.after / f"{prefix}-reference.png"):
                raise ValueError(f"reference differs at {prefix}")
            for (directory, role), stream in zip(sources, streams):
                data = (directory / f"{prefix}-{role}.png").read_bytes()
                if data[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", data[16:24]) != (256, 256):
                    raise ValueError("expected native 256x256 PNGs")
                stream.update(data)
        inputs = [value for directory, role in sources for value in [
            "-threads", "1", "-framerate", str(args.fps), "-i",
            str(directory / f"{sequence:03}-%03d-{role}.png")]]
        layout = ("[0:v][1:v][2:v]hstack=inputs=3,pad=768:288:0:32:color=0x202020,"
                  "drawtext=text='Previous':x=8:y=7:fontsize=18:fontcolor=white,"
                  "drawtext=text='Candidate':x=264:y=7:fontsize=18:fontcolor=white,"
                  "drawtext=text='Reference':x=520:y=7:fontsize=18:fontcolor=white,"
                  "drawtext=text='%{n}':x=728:y=7:fontsize=18:fontcolor=white")
        common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", *inputs,
                  "-filter_complex_threads", "1", "-filter_complex"]
        video = args.out / f"sequence-{sequence:03}.mp4"
        command = [*common, layout, "-frames:v", str(args.frames), "-an", "-c:v", "libx264",
                   "-crf", "12", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(video)]
        subprocess.run(command, check=True)
        entry = {"sequence": sequence, "path": str(video), "sha256": digest(video),
                 "decoded_stream": verify_video(video, args.frames, args.fps),
                 "source_png_stream_sha256": dict(zip(["before", "after", "reference"],
                                                       [s.hexdigest() for s in streams])),
                 "command": command, "review_sheets": []}
        if args.review_sheets:
            for page, frames in enumerate(review_pages(args.frames)):
                sheet = args.out / f"sequence-{sequence:03}-review-{page}.png"
                selected = "+".join(f"eq(n,{frame})" for frame in frames)
                subprocess.run([*common, f"{layout},select='{selected}',tile=1x4",
                                "-frames:v", "1", "-threads", "1", str(sheet)], check=True)
                entry["review_sheets"].append({"path": str(sheet), "frames": frames, "sha256": digest(sheet)})
        report["sequences"].append(entry)
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
