import librosa
import numpy as np
import tensorflow_hub as hub
from utils.audio_converter import convert_to_wav
from utils.yamnet_labels import load_yamnet_labels
from utils.yamnet_emotion_map import map_yamnet_labels_to_emotions


class AudioEmotionAnalyzer:
    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.labels = load_yamnet_labels()

    def analyze_emotion(self, audio_path: str, top_n: int = 10):
        processed_path = "data/new_person.wav"
        convert_to_wav(audio_path, processed_path)
        waveform, sr = librosa.load(processed_path, sr=16000)
        waveform = np.array(waveform, dtype=np.float32)
        scores, _, _ = self.model(waveform)
        mean_scores = np.mean(scores.numpy(), axis=0)

        # 取得前 top_n 的音訊標籤與分數
        top_indices = mean_scores.argsort()[-top_n:][::-1]
        top_labels = {
            self.labels[i]: float(mean_scores[i])
            for i in top_indices
        }
        # 映射到情緒分類
        emotion_results = map_yamnet_labels_to_emotions(top_labels)
        return emotion_results
