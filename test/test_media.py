import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wave
import numpy as np
import tempfile

from pathlib import Path
from moviepy.video.VideoClip import ImageClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from PIL import Image as PILImage
from io import BytesIO
from pydantic import BaseModel
from thinkthinksyn.common_utils.data_structs import Image, Audio, Video

def _make_audio_wav_bytes(duration_sec: float = 0.5, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    wave_data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    pcm16 = np.clip(wave_data * 32767, -32768, 32767).astype(np.int16)

    buf = BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()

def _make_image_png_bytes(width: int = 64, height: int = 64) -> bytes:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 1] = 180
    img = PILImage.fromarray(arr, mode='RGB')
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def _make_video_mp4_bytes(image_png_bytes: bytes, audio_wav_bytes: bytes, duration_sec: float = 0.5) -> bytes:
    with tempfile.TemporaryDirectory(prefix='kp_file_test_') as td:
        tmp_dir = Path(td)
        image_path = tmp_dir / 'frame.png'
        audio_path = tmp_dir / 'audio.wav'
        video_path = tmp_dir / 'video.mp4'

        image_path.write_bytes(image_png_bytes)
        audio_path.write_bytes(audio_wav_bytes)

        arr = np.array(PILImage.open(BytesIO(image_png_bytes)).convert('RGB'))
        clip = ImageClip(arr)
        if hasattr(clip, 'with_duration'):
            clip = clip.with_duration(duration_sec)
        else:
            clip = clip.set_duration(duration_sec)  # type: ignore[attr-defined]

        audio_clip = AudioFileClip(str(audio_path))
        if hasattr(audio_clip, 'subclipped'):
            audio_clip = audio_clip.subclipped(0, duration_sec)
        else:
            audio_clip = audio_clip.subclip(0, duration_sec)  # type: ignore[attr-defined]

        if hasattr(clip, 'with_audio'):
            clip = clip.with_audio(audio_clip)
        else:
            clip = clip.set_audio(audio_clip)  # type: ignore[attr-defined]

        clip.write_videofile(
            str(video_path),
            codec='libx264',
            audio_codec='aac',
            fps=24,
            logger=None,
        )

        try:
            clip.close()
        except Exception:
            pass
        try:
            audio_clip.close()
        except Exception:
            pass

        return video_path.read_bytes()
    
class TestMedia(BaseModel):
    audio: Audio
    image: Image
    video: Video
    
if __name__ == "__main__":
    audio_bytes = _make_audio_wav_bytes()
    audio = Audio.Load(audio_bytes)
    image_bytes = _make_image_png_bytes()
    image = Image.Load(image_bytes)
    video_bytes = _make_video_mp4_bytes(image_bytes, audio_bytes)
    video = Video.Load(video_bytes)
    test_media = TestMedia(audio=audio, image=image, video=video)
    data = test_media.model_dump_json()
    test_media = TestMedia.model_validate_json(data)
    data = test_media.model_dump()
    for key in ['audio', 'image', 'video']:
        print(f"{key} length: {len(data[key]['data'])}")