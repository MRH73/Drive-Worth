const form = document.querySelector("#prediction-form");
const makeSelect = document.querySelector('select[name="make"]');
const modelSelect = document.querySelector('select[name="model"]');
const priceDisplay = document.querySelector("#price-display");
const resultCopy = document.querySelector("#result-copy");
const trainingSearch = document.querySelector("#training-search");
const trainingSearchButton = document.querySelector("#training-search-button");
const trainingTableBody = document.querySelector("#training-table-body");
const trainingSummary = document.querySelector("#training-summary");
const trainingPageLabel = document.querySelector("#training-page-label");
const trainingPrev = document.querySelector("#training-prev");
const trainingNext = document.querySelector("#training-next");
let predictionChart;
let impactChart;
let trainingPage = 1;
let trainingTotalPages = 1;

const dollars = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const wholeNumber = new Intl.NumberFormat("en-US");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

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
          borderColor: "#4f46e5",
          backgroundColor: "rgba(79, 70, 229, 0.12)",
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: "Your car",
          data: [],
          backgroundColor: "#f97316",
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
          backgroundColor: ["#4f46e5", "#0891b2", "#f97316", "#7c3aed", "#16a34a"],
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

function renderTrainingRows(data) {
  trainingTotalPages = data.total_pages;
  trainingPage = data.page;

  trainingSummary.textContent = `Showing ${data.rows.length} of ${wholeNumber.format(data.total_rows)} matching rows.`;
  trainingPageLabel.textContent = `Page ${data.page} of ${data.total_pages}`;
  trainingPrev.disabled = data.page <= 1;
  trainingNext.disabled = data.page >= data.total_pages;

  if (!data.rows.length) {
    trainingTableBody.innerHTML = '<tr><td colspan="6">No rows match that search.</td></tr>';
    return;
  }

  trainingTableBody.innerHTML = data.rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.make)}</td>
          <td>${escapeHtml(row.model)}</td>
          <td>${row.year}</td>
          <td>${wholeNumber.format(row.mileage)}</td>
          <td>${escapeHtml(row.condition)}</td>
          <td>${dollars.format(row.price)}</td>
        </tr>
      `,
    )
    .join("");
}

async function loadTrainingData(page = 1) {
  trainingSummary.textContent = "Loading training rows...";

  const params = new URLSearchParams({
    page,
    per_page: 10,
    search: trainingSearch.value.trim(),
  });
  const response = await fetch(`/api/training-data?${params.toString()}`);
  const data = await response.json();

  if (!response.ok) {
    trainingSummary.textContent = data.error || "Could not load training rows.";
    return;
  }

  renderTrainingRows(data);
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
  loadTrainingData();
}

form.addEventListener("submit", predict);
makeSelect.addEventListener("change", updateModelDropdown);
trainingSearchButton.addEventListener("click", () => loadTrainingData(1));
trainingSearch.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    loadTrainingData(1);
  }
});
trainingPrev.addEventListener("click", () => loadTrainingData(Math.max(trainingPage - 1, 1)));
trainingNext.addEventListener("click", () => loadTrainingData(Math.min(trainingPage + 1, trainingTotalPages)));
resetInitialState();
