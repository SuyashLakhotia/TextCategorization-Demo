from flask import Flask, make_response, jsonify, request, render_template

import preprocess
import models
import rcv1_constants as dataset

app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Resource not found!"}), 404)


@app.route('/')
def root():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify_text():
    req = request.get_json()
    text = req.get("text")
    model_id = req.get("model_id")

    data_tfidf, data_word2ind = preprocess.preprocess(text, dataset.VOCABULARY)
    results = models.run_models(model_id, data_tfidf, data_word2ind)

    _res = []
    for model_id, result in results.items():
        _res.append(_json_record(model_id, dataset.CLASS_NAMES[dataset.CLASSES[result[0]]], result[1]))

    return jsonify(_res)


def _json_record(model_id, class_name, confidence):
    if confidence != "N/A":
        record = {"model_id": model_id, "class_name": class_name, "confidence": "{:.2f}".format(confidence)}
    else:
        record = {"model_id": model_id, "class_name": class_name}

    return record


if __name__ == "__main__":
    app.run(debug=True)
