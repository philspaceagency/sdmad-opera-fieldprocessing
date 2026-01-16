import argparse
import os
import subprocess
import glob

def concatenate_ffmpeg_folder(input_folder, output_path):
    """
    Concatenates all .mp4 files in a folder using ffmpeg (no re-encoding).
    Assumes all input files are compatible (same codec, resolution, framerate).
    """
    # Get sorted list of all .mp4 files in the folder
    clips = sorted(glob.glob(os.path.join(input_folder, "*.MP4")))
    if len(clips) < 2:
        raise ValueError("Need at least two .mp4 files to concatenate.")

    # Create temporary file list for ffmpeg
    file_list_path = os.path.join(input_folder, "concat_list.txt")
    with open(file_list_path, "w") as f:
        for clip in clips:
            f.write(f"file '{os.path.abspath(clip)}'\n")

    # Build ffmpeg command
    command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", file_list_path,
        "-c", "copy",
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Successfully concatenated into: {output_path}")
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg failed:", e)
    finally:
        if os.path.exists(file_list_path):
            os.remove(file_list_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate all .mp4 files in a folder using ffmpeg.")
    parser.add_argument("-f", "--folder", required=True, help="Folder containing .mp4 files")
    parser.add_argument("-o", "--output", required=True, help="Output video file path (e.g., output.mp4)")
    args = parser.parse_args()

    concatenate_ffmpeg_folder(args.folder, args.output)
