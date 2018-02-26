const API_BASE_URL = 'http://127.0.0.1:5000';

const MODEL_IDS = ["all", "linear_svc", "multinomial_nb", "mlp", "cnn_fchollet", "cnn_ykim",
                   "gcnn_chebyshev", "gcnn_spline", "gcnn_fourier"]

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
        $result.html('<strong>' + datum['model_id'] + ':</strong> ' + datum['class_name'] + ' (' +
                     datum['confidence'] + '%)');
        $results.append($result);
    }
    $('#results-section').show();
}

function clearResults() {
    $('#results .result').remove();
    $('#results-section').hide();
}
