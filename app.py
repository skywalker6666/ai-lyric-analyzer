from services.nlp_analysis.lyrics_classification import LyricsClassifier
from services.audio_analysis.audio_emotion import AudioEmotionAnalyzer

if __name__ == "__main__":
    lyrics = "I'm walking alone under the moonlight, feeling empty."
    classifier = LyricsClassifier()
    result = classifier.classify_lyrics(lyrics)
    print("[Lyrics Emotion Classification]:", result)

    audio_path = "data/new_person.wav"
    analyzer = AudioEmotionAnalyzer()
    emotion = analyzer.analyze_emotion(audio_path)
    print("[Audio Emotion Scores]:", emotion)
