import os
from pydub import AudioSegment


def convert_to_wav(
    input_path: str, output_path: str, sample_rate: int = 16000
):
    """將任何音訊檔轉為 16kHz 單聲道 WAV 檔"""
    ext = os.path.splitext(input_path)[1].replace('.', '').lower()
    try:
        audio = AudioSegment.from_file(input_path, format=ext)
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"[轉檔失敗] {input_path}: {e}")
        return None


# 測試
if __name__ == "__main__":
    src = "data/wukong.mp3"
    dst = "data/wukong.wav"
    path = convert_to_wav(src, dst)
    print(f"[完成轉檔] ➜ {path}")
