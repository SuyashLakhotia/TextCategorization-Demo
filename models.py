import pickle

import sklearn
import tensorflow as tf
import numpy as np

import rcv1_constants as dataset


MODEL_IDS = ["linear_svc", "multinomial_nb", "softmax", "mlp", "cnn_fchollet", "cnn_ykim", "gcnn_chebyshev",
             "gcnn_spline", "gcnn_fourier"]


def run_models(model_id_arr, data_tfidf, data_word2ind):
    results = {}

    if "linear_svc" in model_id_arr:
        results["linear_svc"] = run_linear_svc(data_tfidf)
    if "multinomial_nb" in model_id_arr:
        results["multinomial_nb"] = run_multinomial_nb(data_tfidf)
    if "softmax" in model_id_arr:
        results["softmax"] = run_softmax(data_tfidf)
    if "mlp" in model_id_arr:
        results["mlp"] = run_mlp(data_tfidf)
    if "cnn_fchollet" in model_id_arr:
        results["cnn_fchollet"] = run_cnn_fchollet(data_word2ind)
    if "cnn_ykim" in model_id_arr:
        results["cnn_ykim"] = run_cnn_ykim(data_word2ind)
    if "gcnn_chebyshev" in model_id_arr:
        results["gcnn_chebyshev"] = run_gcnn_chebyshev(data_tfidf)
    if "gcnn_spline" in model_id_arr:
        results["gcnn_spline"] = run_gcnn_spline(data_tfidf)
    if "gcnn_fourier" in model_id_arr:
        results["gcnn_fourier"] = run_gcnn_fourier(data_tfidf)

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


def run_mlp(data_tfidf):
    predicted, probability = _run_tf_model(data_tfidf,
                                           "models/mlp/1521613235/checkpoints/model-164100.meta",
                                           "models/mlp/1521613235/checkpoints/.")
    return (predicted, probability * 100)


def run_cnn_fchollet(data_word2ind):
    predicted, probability = _run_tf_model(data_word2ind,
                                           "models/cnn_fchollet/1520951089/checkpoints/model-164100.meta",
                                           "models/cnn_fchollet/1520951089/checkpoints/.")
    return (predicted, probability * 100)


def run_cnn_ykim(data_word2ind):
    predicted, probability = _run_tf_model(data_word2ind,
                                           "models/cnn_ykim/1520942963/checkpoints/model-164100.meta",
                                           "models/cnn_ykim/1520942963/checkpoints/.")
    return (predicted, probability * 100)


def run_gcnn_chebyshev(data_tfidf):
    predicted, probability = _run_tf_model(data_tfidf,
                                           "models/gcnn_chebyshev/1521146641/checkpoints/model-164100.meta",
                                           "models/gcnn_chebyshev/1521146641/checkpoints/.")
    return (predicted, probability * 100)


def run_gcnn_spline(data_tfidf):
    predicted, probability = _run_tf_model(data_tfidf,
                                           "models/gcnn_spline/1521456262/checkpoints/model-164100.meta",
                                           "models/gcnn_spline/1521456262/checkpoints/.")
    return (predicted, probability * 100)


def run_gcnn_fourier(data_tfidf):
    predicted, probability = _run_tf_model(data_tfidf,
                                           "models/gcnn_fourier/1521585284/checkpoints/model-164100.meta",
                                           "models/gcnn_fourier/1521585284/checkpoints/.")
    return (predicted, probability * 100)


def _run_tf_model(data, graph_filepath, checkpoints_filepath):
    with tf.Session(graph=tf.Graph()) as sess:
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
