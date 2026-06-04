const form = document.querySelector("#prediction-form");
const priceDisplay = document.querySelector("#price-display");
const resultCopy = document.querySelector("#result-copy");
const impactList = document.querySelector("#impact-list");

// Format numbers as US dollars for the UI.
const dollars = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function formPayload() {
  // Collect all form fields and turn them into a simple object.
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function readableFeatureName(feature) {
  // Convert names like accident_history into Accident History.
  return feature
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function renderImpactList(items) {
  // Show the top important features under the prediction.
  impactList.innerHTML = items
    .map(
      (item) => `
        <div class="impact-item">
          <span>${readableFeatureName(item.feature)}</span>
          <strong>${dollars.format(item.impact)}</strong>
        </div>
      `
    )
    .join("");
}

async function predict(event) {
  // Stop the browser from refreshing the page when the form is submitted.
  event.preventDefault();
  resultCopy.textContent = "Running prediction...";

  // Send the form data to the Flask prediction API.
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(formPayload()),
  });

  const data = await response.json();

  if (!response.ok) {
    // Show an error if the API rejects the input.
    resultCopy.textContent = data.error || "Something went wrong.";
    return;
  }

  // Update the page with the predicted car price.
  priceDisplay.textContent = dollars.format(data.predicted_price);
  resultCopy.textContent = `${data.best_model} estimates this vehicle price based on the trained regression pipeline.`;
  renderImpactList(data.top_impact);
}

async function loadDashboard() {
  // Load model metrics and feature impact from Flask.
  const response = await fetch("/api/metrics");
  const data = await response.json();

  // Prepare data for the model comparison chart.
  const names = data.metrics.map((item) => item.name);
  const maeValues = data.metrics.map((item) => item.mae);

  // Prepare data for the feature impact chart.
  const impactLabels = data.feature_impact.slice(0, 6).map((item) => readableFeatureName(item.feature));
  const impactValues = data.feature_impact.slice(0, 6).map((item) => item.impact);

  // Bar chart that compares model error.
  new Chart(document.querySelector("#metrics-chart"), {
    type: "bar",
    data: {
      labels: names,
      datasets: [
        {
          label: "MAE",
          data: maeValues,
          backgroundColor: "#0f766e",
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          ticks: {
            callback: (value) => dollars.format(value),
          },
        },
      },
    },
  });

  // Horizontal bar chart that shows the most important features.
  new Chart(document.querySelector("#impact-chart"), {
    type: "bar",
    data: {
      labels: impactLabels,
      datasets: [
        {
          label: "Impact",
          data: impactValues,
          backgroundColor: ["#0f766e", "#d97706", "#2563eb", "#7c3aed", "#dc2626", "#475569"],
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          ticks: {
            callback: (value) => dollars.format(value),
          },
        },
      },
    },
  });

  renderImpactList(data.feature_impact.slice(0, 3));
}

// Listen for form submissions.
form.addEventListener("submit", predict);

// Load charts when the page opens.
loadDashboard();
