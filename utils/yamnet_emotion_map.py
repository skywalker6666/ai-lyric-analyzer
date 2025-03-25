# utils/yamnet_emotion_map.py

EMOTION_KEYWORDS = {
    "happy": [
        "Laughter",
        "Child speech",
        "Cheering",
        "Applause",
        "Hubbub, speech noise"
    ],
    "sad": ["Crying, sobbing", "Sad music", "Wail, moan", "Sigh"],
    "angry": ["Shout", "Screaming", "Argument", "Yell"],
    "fear": ["Screaming", "Shout", "Groan"],
    "surprise": ["Cheering", "Gasp", "Whoop"],
    "calm": ["Chant", "Silence", "Ambient music", "Whispering"]
}


def map_yamnet_labels_to_emotions(scores: dict):
    emotion_scores = {k: 0.0 for k in EMOTION_KEYWORDS}

    for label, score in scores.items():
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if label in keywords:
                emotion_scores[emotion] += score

    # 過濾 0 值並排序
    sorted_emotions = sorted(
        [
            {"emotion": k, "score": v}
            for k, v in emotion_scores.items()
            if v > 0
        ],
        key=lambda x: x["score"],
        reverse=True
    )
    return sorted_emotions
