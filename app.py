from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

with open("bigram_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"].lower().strip()

    if not text:
        return jsonify({"prediction": []})

    last_word = text.split()[-1]

    if last_word in model:
        suggestions = sorted(
            model[last_word].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        predicted_words = [word for word, prob in suggestions]

        return jsonify({"prediction": predicted_words})

    return jsonify({"prediction": ["No suggestion"]})

if __name__ == "__main__":
    app.run(debug=True)