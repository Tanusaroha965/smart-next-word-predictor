# import pickle
# from collections import defaultdict, Counter
# from nltk.util import ngrams
# from nltk.tokenize import word_tokenize
# import nltk

# print("Starting model training...")
# nltk.download('punkt')
# nltk.download('punkt_tab')

# files = [
#     "fi_FI.twitter.txt",
#     "ru_RU.twitter.txt",
#     "ru_RU.news.txt"
# ]

# all_text = ""

# for file_name in files:
#     print(f"Reading {file_name}...")
#     with open(file_name, "r", encoding="utf-8") as file:
#         all_text += file.read().lower() + " "

# print("Tokenizing data...")

# tokens = word_tokenize(all_text)

# bigram_model = defaultdict(Counter)

# print("Creating bigram model...")

# for w1, w2 in ngrams(tokens, 2):
#     bigram_model[w1][w2] += 1

# with open("model.pkl", "wb") as file:
#     pickle.dump(dict(bigram_model), file)

# print("Model trained successfully")

import os
import re
import pickle
import nltk
from collections import defaultdict, Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

nltk.download('punkt')

FILES = [
   "training-dataset.txt"
]

MODEL_FILE = "bigram_model.pkl"


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_combine_text(files):
    texts = []

    for file_name in files:
        if os.path.exists(file_name):
            print(f"Reading {file_name}...")
            with open(file_name, "r", encoding="utf-8") as file:
                texts.append(file.read())
        else:
            print(f"Warning: {file_name} not found")

    return " ".join(texts)


def train_bigram_model(text):
    print("Preprocessing text...")
    clean_text = preprocess_text(text)

    print("Tokenizing...")
    tokens = word_tokenize(clean_text)

    bigram_counts = defaultdict(Counter)
    unigram_counts = Counter(tokens)

    print("Creating bigram counts...")
    for w1, w2 in ngrams(tokens, 2):
        bigram_counts[w1][w2] += 1

    print("Calculating probabilities...")
    bigram_probabilities = {}

    for w1, next_words in bigram_counts.items():
        total_count = unigram_counts[w1]
        bigram_probabilities[w1] = {
            w2: count / total_count
            for w2, count in next_words.items()
        }

    return bigram_probabilities


def save_model(model, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved as {file_name}")


def load_model(file_name):
    with open(file_name, "rb") as file:
        return pickle.load(file)


def predict_next_word(model, word, top_k=3):
    word = word.lower()

    if word not in model:
        return ["No prediction available"]

    predictions = sorted(
        model[word].items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [word for word, prob in predictions[:top_k]]


def main():
    print("Starting model training...")

    all_text = load_and_combine_text(FILES)

    if not all_text:
        print("No valid text files found")
        return

    model = train_bigram_model(all_text)

    save_model(model, MODEL_FILE)

    print("\nModel trained successfully")

    while True:
        user_input = input("\nEnter a word (or type 'exit'): ")

        if user_input.lower() == "exit":
            break

        predictions = predict_next_word(model, user_input)

        print("Predictions:", predictions)


if __name__ == "__main__":
    main()