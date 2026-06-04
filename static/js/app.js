const form = document.querySelector("#prediction-form");
const priceDisplay = document.querySelector("#price-display");
const resultCopy = document.querySelector("#result-copy");
let costChart;

const dollars = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function formPayload() {
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function createCostChart(values = []) {
  costChart = new Chart(document.querySelector("#cost-chart"), {
    type: "line",
    data: {
      labels: values.map((_value, index) => index + 1),
      datasets: [
        {
          label: "Cost",
          data: values,
          borderColor: "#0f766e",
          backgroundColor: "rgba(15, 118, 110, 0.12)",
          fill: true,
          tension: 0.25,
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
}

function updateCostChart(values) {
  costChart.data.labels = values.map((_value, index) => index + 1);
  costChart.data.datasets[0].data = values;
  costChart.update();
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
  resultCopy.textContent = `${data.model_name} predicted the price from year and mileage.`;
  updateCostChart(data.cost_history);
}

function resetInitialState() {
  form.reset();
  priceDisplay.textContent = "$0";
  resultCopy.textContent = "Complete the form to generate a prediction.";
  createCostChart([]);
}

form.addEventListener("submit", predict);
resetInitialState();
