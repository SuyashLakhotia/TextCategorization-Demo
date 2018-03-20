import pickle

import sklearn


MODEL_IDS = ["all", "linear_svc", "multinomial_nb", "mlp", "cnn_fchollet", "cnn_ykim", "gcnn_chebyshev",
             "gcnn_spline", "gcnn_fourier"]


def run_models(model_id, data_tfidf, data_word2ind):
    results = {}

    if model_id == "all" or model_id == "linear_svc":
        results["linear_svc"] = run_linear_svc(data_tfidf)
    if model_id == "all" or model_id == "multinomial_nb":
        results["multinomial_nb"] = run_multinomial_nb(data_tfidf)
    if model_id == "all" or model_id == "mlp":
        results["mlp"] = (0, 00.00)
    if model_id == "all" or model_id == "cnn_fchollet":
        results["cnn_fchollet"] = (0, 00.00)
    if model_id == "all" or model_id == "cnn_ykim":
        results["cnn_ykim"] = (0, 00.00)
    if model_id == "all" or model_id == "gcnn_chebyshev":
        results["gcnn_chebyshev"] = (0, 00.00)
    if model_id == "all" or model_id == "gcnn_spline":
        results["gcnn_spline"] = (0, 00.00)
    if model_id == "all" or model_id == "gcnn_fourier":
        results["gcnn_fourier"] = (0, 00.00)

    return results


def run_linear_svc(data_tfidf):
    clf = pickle.load(open("models/linear_svc/1521482564/pickle.pkl", "rb"))
    predicted = clf.predict(data_tfidf)[0]
    confidence = "N/A"
    return (predicted, "N/A")


def run_multinomial_nb(data_tfidf):
    clf = pickle.load(open("models/multinomial_nb/1521482564/pickle.pkl", "rb"))
    predicted = clf.predict(data_tfidf)[0]
    confidence = clf.predict_proba(data_tfidf)
    if predicted >= 45:  # issue with 'GVOTE' class
        _predicted = predicted - 1
    else:
        _predicted = predicted
    return (predicted, confidence[0][_predicted] * 100)
