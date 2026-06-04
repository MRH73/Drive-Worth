const form = document.querySelector("#prediction-form");
const makeSelect = document.querySelector('select[name="make"]');
const modelSelect = document.querySelector('select[name="model"]');
const priceDisplay = document.querySelector("#price-display");
const resultCopy = document.querySelector("#result-copy");
let predictionChart;
let impactChart;

const dollars = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function formPayload() {
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function updateModelDropdown() {
  const selectedMake = makeSelect.value;
  const models = window.makeModelMap[selectedMake] || [];

  modelSelect.innerHTML = '<option value="">Select model</option>';
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    modelSelect.appendChild(option);
  });
}

function createPredictionChart() {
  predictionChart = new Chart(document.querySelector("#prediction-chart"), {
    type: "scatter",
    data: {
      datasets: [
        {
          type: "line",
          label: "Prediction curve",
          data: [],
          borderColor: "#0f766e",
          backgroundColor: "rgba(15, 118, 110, 0.12)",
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: "Your car",
          data: [],
          backgroundColor: "#d97706",
          pointRadius: 7,
          pointHoverRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      parsing: false,
      plugins: {
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${dollars.format(context.parsed.y)} at ${Math.round(context.parsed.x).toLocaleString()} miles`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Mileage" },
          ticks: {
            callback: (value) => Number(value).toLocaleString(),
          },
        },
        y: {
          title: { display: true, text: "Predicted Price" },
          ticks: {
            callback: (value) => dollars.format(value),
          },
        },
      },
    },
  });
}

function readableFeatureName(feature) {
  return feature
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function createImpactChart() {
  const impacts = window.featureImpacts || [];

  impactChart = new Chart(document.querySelector("#impact-chart"), {
    type: "bar",
    data: {
      labels: impacts.map((item) => readableFeatureName(item.feature)),
      datasets: [
        {
          label: "Impact",
          data: impacts.map((item) => item.impact),
          backgroundColor: ["#0f766e", "#d97706", "#2563eb", "#7c3aed", "#dc2626"],
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
          beginAtZero: true,
          ticks: {
            callback: (value) => dollars.format(value),
          },
        },
      },
    },
  });
}

function updatePredictionChart(linePoints, predictionPoint) {
  predictionChart.data.datasets[0].data = linePoints.map((point) => ({
    x: point.mileage,
    y: point.predicted_price,
  }));
  predictionChart.data.datasets[1].data = [
    {
      x: predictionPoint.mileage,
      y: predictionPoint.predicted_price,
    },
  ];
  predictionChart.update();
}

async function predict(event) {
  event.preventDefault();
  resultCopy.textContent = "Running prediction...";

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(formPayload()),
  });

  const data = await response.json();

  if (!response.ok) {
    resultCopy.textContent = data.error || "Something went wrong.";
    return;
  }

  priceDisplay.textContent = dollars.format(data.predicted_price);
  resultCopy.textContent = `${data.model_name} predicted the price and plotted your car on the curve.`;
  updatePredictionChart(data.line_points, data.prediction_point);
}

function resetInitialState() {
  form.reset();
  priceDisplay.textContent = "$0";
  resultCopy.textContent = "Complete the form to generate a prediction.";
  updateModelDropdown();
  createPredictionChart();
  createImpactChart();
}

form.addEventListener("submit", predict);
makeSelect.addEventListener("change", updateModelDropdown);
resetInitialState();
