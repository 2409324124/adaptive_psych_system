const startBtn = document.querySelector("#startBtn");
const modelInput = document.querySelector("#model");
const maxItemsInput = document.querySelector("#maxItems");
const questionArea = document.querySelector("#questionArea");
const progressEl = document.querySelector("#progress");
const questionText = document.querySelector("#questionText");
const responsesEl = document.querySelector("#responses");
const resultsEl = document.querySelector("#results");
const irtScores = document.querySelector("#irtScores");
const classicalScores = document.querySelector("#classicalScores");

let sessionId = null;
let currentQuestion = null;

const responseLabels = {
  1: "1 Very inaccurate",
  2: "2 Moderately inaccurate",
  3: "3 Neutral",
  4: "4 Moderately accurate",
  5: "5 Very accurate",
};

startBtn.addEventListener("click", async () => {
  resultsEl.hidden = true;
  const response = await fetch("/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scoring_model: modelInput.value,
      max_items: Number(maxItemsInput.value),
      coverage_min_per_dimension: 2,
    }),
  });
  const payload = await response.json();
  sessionId = payload.session_id;
  renderQuestion(payload.next_question);
});

function renderQuestion(question) {
  currentQuestion = question;
  if (!question) {
    loadResults();
    return;
  }
  questionArea.hidden = false;
  const progress = question.progress;
  progressEl.textContent = `${progress.answered + 1} / ${progress.max_items}`;
  questionText.textContent = question.text;
  responsesEl.innerHTML = "";
  for (const value of [1, 2, 3, 4, 5]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = responseLabels[value];
    button.addEventListener("click", () => submitResponse(value));
    responsesEl.appendChild(button);
  }
}

async function submitResponse(value) {
  const response = await fetch(`/sessions/${sessionId}/responses`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      item_id: currentQuestion.item_id,
      response: value,
    }),
  });
  const payload = await response.json();
  if (payload.complete) {
    questionArea.hidden = true;
    await loadResults();
    return;
  }
  renderQuestion(payload.next_question);
}

async function loadResults() {
  const response = await fetch(`/sessions/${sessionId}/result`);
  const result = await response.json();
  resultsEl.hidden = false;
  renderScores(irtScores, result.irt_t_scores);
  const classical = {};
  for (const [trait, score] of Object.entries(result.classical_big5)) {
    classical[trait] =
      score.tendency_t_score === null
        ? `NA (${score.answered_count})`
        : `${score.tendency_t_score.toFixed(1)} (${score.answered_count})${score.answered_count < 2 ? " low evidence" : ""}`;
  }
  renderScores(classicalScores, classical);
}

function renderScores(target, scores) {
  target.innerHTML = "";
  for (const [trait, score] of Object.entries(scores)) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = trait;
    dd.textContent = typeof score === "number" ? score.toFixed(1) : score;
    target.append(dt, dd);
  }
}
