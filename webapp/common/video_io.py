"""Small video helpers: re-encode to H.264 so the browser can play the result."""
import shutil
import subprocess


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def to_h264(src, dst):
    """OpenCV writes mp4v, which most browsers cannot play. Re-encode to H.264.

    Returns True if dst is a browser-playable H.264 file. If ffmpeg is missing,
    the original file is copied and False is returned.
    """
    exe = ffmpeg_exe()
    if exe is None:
        shutil.copyfile(src, dst)
        return False
    cmd = [
        exe, "-y", "-loglevel", "error", "-i", str(src),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # H.264 needs even sizes
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart", str(dst),
    ]
    subprocess.run(cmd, check=True)
    return True
