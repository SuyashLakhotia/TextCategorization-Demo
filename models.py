import pickle

import sklearn
import tensorflow as tf
import numpy as np

import rcv1_constants as dataset


MODEL_IDS = ["all", "linear_svc", "multinomial_nb", "softmax", "mlp", "cnn_fchollet", "cnn_ykim",
             "gcnn_chebyshev", "gcnn_spline", "gcnn_fourier"]


def run_models(model_id, data_tfidf, data_word2ind):
    results = {}
    data_tfidf = data_tfidf.toarray()  # convert sparse matrix to array

    if model_id == "all" or model_id == "linear_svc":
        results["linear_svc"] = run_linear_svc(data_tfidf)
    if model_id == "all" or model_id == "multinomial_nb":
        results["multinomial_nb"] = run_multinomial_nb(data_tfidf)
    if model_id == "all" or model_id == "softmax":
        results["softmax"] = run_softmax(data_tfidf)
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


def run_softmax(data_tfidf):
    predicted, probability = _run_tf_model(data_tfidf,
                                           "models/mlp/1521146411/checkpoints/model-164100.meta",
                                           "models/mlp/1521146411/checkpoints/.")
    return (predicted, probability * 100)


def _run_tf_model(data, graph_filepath, checkpoints_filepath):
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with graph.as_default():
        saver = tf.train.import_meta_graph(graph_filepath)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoints_filepath))

        scores_op = tf.get_default_graph().get_tensor_by_name("output/scores:0")
        predictions_op = tf.get_default_graph().get_tensor_by_name("output/predictions:0")

        x_batch = np.zeros((dataset.BATCH_SIZE, data.shape[1]))
        x_batch[0] = data[0]
        feed_dict = {
            "input_x:0": x_batch,
            "train_flag:0": False
        }
        scores, predictions = sess.run([scores_op, predictions_op], feed_dict)

    predicted = predictions[0]

    e_x = np.exp(scores[0] - np.max(scores[0]))
    probability = e_x[predicted] / np.sum(e_x)

    return predicted, probability
