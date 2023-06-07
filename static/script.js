const form = document.querySelector('#upload-form');
const inputImage = document.querySelector('#input-image');
const outputImage = document.querySelector('#output-image');
const predicted_class = document.querySelector('#predicted-class');
const saffronProbability = document.querySelector('#saffron-probability');
const nonSaffronProbability = document.querySelector('#non-saffron-probability');

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const formData = new FormData(form);
  fetch(form.action, {
    method: form.method,
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    inputImage.src = 'data:image/png;base64,' + data.input_image;
    outputImage.src = 'data:image/png;base64,' + data.gradcam_image;
    saffronProbability.textContent = 'Saffron Probability: ' + data.saffron_probability + '%';
    nonSaffronProbability.textContent = 'Non-Saffron Probability: ' + data.non_saffron_probability + '%';
    predicted_class.textContent = 'Predicted Class: ' + data.predicted_class;
    // Display the result container
    const resultContainer = document.querySelector('#result');
    resultContainer.style.display = 'block';
  })
  .catch(error => console.error(error));
});