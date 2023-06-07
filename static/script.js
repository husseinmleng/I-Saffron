document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  const form = event.target;
  const formData = new FormData(form);

  try {
    const response = await fetch(form.action, {
      method: form.method,
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();

      document.getElementById("input-image").src = `data:image/png;base64,${data.input_image}`;
      document.getElementById("output-image").src = `data:image/png;base64,${data.gradcam_image}`;
      document.getElementById("predicted-class").textContent = `Predicted Class: ${data.predicted_class}`;
      document.getElementById("saffron-probability").textContent = `Saffron Probability: ${data.saffron_probability}%`;
      document.getElementById("non-saffron-probability").textContent = `Non-Saffron Probability: ${data.non_saffron_probability}%`;
    } else {
      const error = await response.json();
      alert(`Error: ${error.error}`);
    }
  } catch (error) {
    alert(`Error: ${error.message}`);
  }
});
