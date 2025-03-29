from services.nlp_analysis.lyrics_classification import LyricsClassifier
from services.audio_analysis.audio_emotion import AudioEmotionAnalyzer

if __name__ == "__main__":
    lyrics = "I'm walking alone under the moonlight, feeling empty."
    classifier = LyricsClassifier()
    result = classifier.classify_lyrics(lyrics)
    print("[Lyrics Emotion Classification]:", result)

    analyzer = AudioEmotionAnalyzer()
    analyzer.process_directory(
        directory="data", top_n=10, output_csv="emotion_results.csv"
    )
