from transformers import pipeline


class LyricsClassifier:
    def __init__(self):
        self.model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

    def classify_lyrics(self, lyrics: str):
        return self.model(lyrics)
