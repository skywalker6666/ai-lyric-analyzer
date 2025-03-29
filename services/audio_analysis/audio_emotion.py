import csv
import os
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
        processed_path = audio_path.replace(".mp3", ".wav")
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
        print("[Top Raw YAMNet Predictions]:")
        for label, score in top_labels.items():
            print(f"  - {label}: {score:.4f}")
        # 映射到情緒分類
        emotion_results = map_yamnet_labels_to_emotions(top_labels)
        if isinstance(emotion_results, list):
            emotion_results = {label: "unknown" for label in top_labels}  # 預設值
        return top_labels, emotion_results

    def process_directory(
        self,
        directory: str,
        top_n: int = 10,
        output_csv: str = "emotion_results.csv"
    ):
        results = []
        for file in os.listdir(directory):
            if file.endswith(".mp3"):
                audio_path = os.path.join(directory, file)
                print(f"Processing file: {audio_path}")
                top_labels, emotion_results = self.analyze_emotion(
                    audio_path, top_n
                )

                # 對應與不對應的標籤
                matching_labels = [
                    label for label, emotion in emotion_results.items()
                    if emotion == "match"
                ]
                non_matching_labels = [
                    label for label, emotion in emotion_results.items()
                    if emotion == "non-match"
                ]

                print(f"Matching Labels: {matching_labels}")
                print(f"Non-Matching Labels: {non_matching_labels}")

                results.append({
                    "file": file,
                    "top_labels": top_labels,
                    "matching_labels": matching_labels,
                    "non_matching_labels": non_matching_labels
                })

        # 將結果輸出到 CSV
        with open(
            output_csv, mode="w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "file",
                    "top_labels",
                    "matching_labels",
                    "non_matching_labels"
                ]
            )
            writer.writeheader()
            for result in results:
                writer.writerow({
                    "file": result["file"],
                    "top_labels": result["top_labels"],
                    "matching_labels": ", ".join(result["matching_labels"]),
                    "non_matching_labels": ", ".join(
                        result["non_matching_labels"]
                    )
                })
        print(f"Results saved to {output_csv}")
