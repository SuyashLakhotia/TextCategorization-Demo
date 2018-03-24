const API_BASE_URL = 'http://127.0.0.1:5000';

const MODEL_IDS = ["linear_svc", "multinomial_nb", "softmax", "mlp", "cnn_fchollet", "cnn_ykim",
                   "gcnn_chebyshev", "gcnn_spline", "gcnn_fourier"]
const MODEL_NAMES = {"linear_svc": "Linear SVC", "multinomial_nb": "Multinomial Naive Bayes", "softmax": "Softmax", "mlp": "Multilayer Perceptron", "cnn_fchollet": "F. Chollet CNN", "cnn_ykim": "Y. Kim CNN", "gcnn_chebyshev": "Graph CNN (Chebyshev)", "gcnn_spline": "Graph CNN (Spline)", "gcnn_fourier": "Graph CNN (Fourier)"}

function submit() {
    clearResults();

    let text = $('#text').val();

    let $selectedModels = $('input[name="model-checkbox"]:checked');
    let selectedModels = [];
    for (const $modelCheckbox of $selectedModels) {
        selectedModels.push($modelCheckbox.value);
    }
    
    let request = {
        modelIDs: selectedModels,
        text: text
    };

    $('#submit-btn').html('<i class="fa fa-circle-o-notch fa-spin"></i>&nbsp;&nbsp;Loading');
    $('#submit-btn').prop('disabled', true);

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
        
        let html = '<strong>' + MODEL_NAMES[datum['modelID']] + ':</strong> ' + datum['className'];
        if (datum['confidence'] != undefined)
            html += ' (' + datum['confidence'] + '%)';

        $result.html(html);
        $results.append($result);
    }
    $('#results-section').show();
    $('#submit-btn').html('Submit');
    $('#submit-btn').prop('disabled', false);
}

function clearResults() {
    $('#results .result').remove();
    $('#results-section').hide();
}
