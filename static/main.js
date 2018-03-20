const API_BASE_URL = 'http://127.0.0.1:5000';

const MODEL_IDS = ["all", "linear_svc", "multinomial_nb", "softmax", "mlp", "cnn_fchollet", "cnn_ykim",
                   "gcnn_chebyshev", "gcnn_spline", "gcnn_fourier"]
const MODEL_NAMES = {"linear_svc": "Linear SVC", "multinomial_nb": "Multinomial Naive Bayes", "softmax": "Softmax", "mlp": "Multilayer Perceptron", "cnn_fchollet": "F. Chollet CNN", "cnn_ykim": "Y. Kim CNN", "gcnn_chebyshev": "Graph CNN (Chebyshev)", "gcnn_spline": "Graph CNN (Spline)", "gcnn_fourier": "Graph CNN (Fourier)"}

function submit() {
    clearResults();

    let text = $('#text').val();
    let modelID = MODEL_IDS[$('#model-select').prop('selectedIndex')];
    let request = {
        model_id: modelID,
        text: text
    };

    $.ajax({
        async: true,
        crossDomain: true,
        url: API_BASE_URL + "/classify",
        method: "POST",
        headers: {
            "content-type": "application/json",
            "cache-control": "no-cache",
        },
        processData: false,
        data: JSON.stringify(request)
    }).then((data) => {
        showResults(data);
    });
}

function showResults(data) {
    let $results = $('#results');
    for (const datum of data) {
        let $result = $('#result-template').clone().removeAttr('id');
        
        let html = '<strong>' + MODEL_NAMES[datum['model_id']] + ':</strong> ' + datum['class_name'];
        if (datum['confidence'] != undefined)
            html += ' (' + datum['confidence'] + '%)';

        $result.html(html);
        $results.append($result);
    }
    $('#results-section').show();
}

function clearResults() {
    $('#results .result').remove();
    $('#results-section').hide();
}
