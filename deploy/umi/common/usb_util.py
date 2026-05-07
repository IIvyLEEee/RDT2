import glob
import os


def get_sorted_v4l_paths():
    """Return stable V4L2 camera device paths for UVC/GoPro capture cards."""
    by_id_paths = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    if by_id_paths:
        return by_id_paths

    video_paths = []
    for path in sorted(glob.glob("/dev/video*")):
        if os.path.exists(path):
            video_paths.append(path)
    return video_paths


def reset_all_elgato_devices():
    """Compatibility no-op for UMI code paths that reset capture cards."""
    return None
