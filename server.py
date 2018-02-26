from flask import Flask, make_response, jsonify, request, render_template


MODEL_IDS = ["all", "linear_svc", "multinomial_nb", "mlp", "cnn_fchollet", "cnn_ykim", "gcnn_chebyshev",
             "gcnn_spline", "gcnn_fourier"]
CLASS_NAMES = ["POSITIVE", "NEGATIVE"]


app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Resource not found!"}), 404)


@app.route('/')
def root():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def categorize_text():
    req = request.get_json()
    text = req.get("text")
    model_id = req.get("model_id")

    res = []
    if model_id == "all":
        for i in range(1, len(MODEL_IDS)):
            res.append(_json_record(MODEL_IDS[i], "POSITIVE", 90.23))
    elif model_id == "linear_svc":
        res.append(_json_record("linear_svc", "POSITIVE", 90.23))
    elif model_id == "multinomial_nb":
        res.append(_json_record("multinomial_nb", "POSITIVE", 90.23))
    elif model_id == "mlp":
        res.append(_json_record("mlp", "POSITIVE", 90.23))
    elif model_id == "cnn_fchollet":
        res.append(_json_record("cnn_fchollet", "POSITIVE", 90.23))
    elif model_id == "cnn_ykim":
        res.append(_json_record("cnn_ykim", "POSITIVE", 90.23))
    elif model_id == "gcnn_chebyshev":
        res.append(_json_record("gcnn_chebyshev", "POSITIVE", 90.23))
    elif model_id == "gcnn_spline":
        res.append(_json_record("gcnn_spline", "POSITIVE", 90.23))
    elif model_id == "gcnn_fourier":
        res.append(_json_record("gcnn_fourier", "POSITIVE", 90.23))

    return jsonify(res)


def _json_record(model_id, class_name, confidence):
    return {"model_id": model_id, "class_name": class_name, "confidence": "{:g}".format(confidence)}


if __name__ == "__main__":
    app.run(debug=True)
