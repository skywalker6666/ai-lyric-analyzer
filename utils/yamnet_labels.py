import os

LABELS_URL = (
    'https://raw.githubusercontent.com/tensorflow/models/master/research/'
    'audioset/yamnet/yamnet_class_map.csv'
)


def load_yamnet_labels(local_path='yamnet_class_map.csv'):
    if not os.path.exists(local_path):
        import urllib.request
        urllib.request.urlretrieve(LABELS_URL, local_path)
    with open(local_path, 'r', encoding='utf-8') as f:
        return [line.strip().split(',')[2] for line in f.readlines()[1:]]
