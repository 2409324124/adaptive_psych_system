const startBtn = document.querySelector("#startBtn");
const restartBtn = document.querySelector("#restartBtn");
const newSessionBtn = document.querySelector("#newSessionBtn");
const exportBtn = document.querySelector("#exportBtn");
const modelInput = document.querySelector("#model");
const maxItemsInput = document.querySelector("#maxItems");
const setupStage = document.querySelector("#setupStage");
const setupPanel = document.querySelector("#setupPanel");
const sessionStage = document.querySelector("#sessionStage");
const questionArea = document.querySelector("#questionArea");
const sessionTitle = document.querySelector("#sessionTitle");
const progressLabel = document.querySelector("#progressLabel");
const progressHint = document.querySelector("#progressHint");
const progressBar = document.querySelector("#progressBar");
const questionText = document.querySelector("#questionText");
const responsesEl = document.querySelector("#responses");
const chatLog = document.querySelector("#chatLog");
const resultsEl = document.querySelector("#results");
const resultProgress = document.querySelector("#resultProgress");
const resultNotice = document.querySelector("#resultNotice");
const paramModeLabel = document.querySelector("#paramModeLabel");
const paramPathLabel = document.querySelector("#paramPathLabel");
const stopReasonLabel = document.querySelector("#stopReasonLabel");
const stopReasonList = document.querySelector("#stopReasonList");
const paramMetaLabel = document.querySelector("#paramMetaLabel");
const confidenceTitle = document.querySelector("#confidenceTitle");
const confidenceCopy = document.querySelector("#confidenceCopy");
const meanStandardError = document.querySelector("#meanStandardError");
const confidenceReady = document.querySelector("#confidenceReady");
const confidenceGrid = document.querySelector("#confidenceGrid");
const interpretationOverview = document.querySelector("#interpretationOverview");
const interpretationRange = document.querySelector("#interpretationRange");
const highlightList = document.querySelector("#highlightList");
const lowlightList = document.querySelector("#lowlightList");
const cautionList = document.querySelector("#cautionList");
const coverageGrid = document.querySelector("#coverageGrid");
const irtScores = document.querySelector("#irtScores");
const classicalScores = document.querySelector("#classicalScores");

let sessionId = null;
let currentQuestion = null;
let submitting = false;

const responseLabels = {
  1: "1 Very inaccurate",
  2: "2 Moderately inaccurate",
  3: "3 Neutral",
  4: "4 Moderately accurate",
  5: "5 Very accurate",
};

startBtn.addEventListener("click", startSession);
restartBtn.addEventListener("click", restartSession);
newSessionBtn.addEventListener("click", resetApp);
exportBtn.addEventListener("click", exportSession);

function resetApp() {
  sessionId = null;
  currentQuestion = null;
  submitting = false;
  setupStage.hidden = false;
  setupPanel.hidden = false;
  sessionStage.hidden = true;
  resultsEl.hidden = true;
  questionArea.hidden = false;
  questionText.textContent = "";
  responsesEl.innerHTML = "";
  chatLog.innerHTML = "";
  coverageGrid.innerHTML = "";
  confidenceTitle.textContent = "";
  confidenceCopy.textContent = "";
  meanStandardError.textContent = "";
  confidenceReady.textContent = "";
  confidenceGrid.innerHTML = "";
  irtScores.innerHTML = "";
  classicalScores.innerHTML = "";
  resultProgress.textContent = "";
  resultNotice.textContent = "";
  paramModeLabel.textContent = "";
  paramPathLabel.textContent = "";
  stopReasonLabel.textContent = "";
  stopReasonList.innerHTML = "";
  paramMetaLabel.textContent = "";
  interpretationOverview.textContent = "";
  interpretationRange.textContent = "";
  highlightList.innerHTML = "";
  lowlightList.innerHTML = "";
  cautionList.innerHTML = "";
  updateProgress({ answered: 0, max_items: Number(maxItemsInput.value), remaining: Number(maxItemsInput.value), complete: false });
}

async function startSession() {
  startBtn.disabled = true;
  try {
    const maxItems = Number(maxItemsInput.value);
    const response = await fetch("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scoring_model: modelInput.value,
        max_items: maxItems,
        min_items: Math.min(8, maxItems),
        param_mode: "legacy",
        coverage_min_per_dimension: 2,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to create session.");
    }
    sessionId = payload.session_id;
    chatLog.innerHTML = "";
    appendBubble("system", "Session started. Answer naturally. The next item will adapt to your last response.");
    setupStage.hidden = true;
    setupPanel.hidden = true;
    resultsEl.hidden = true;
    sessionStage.hidden = false;
    renderQuestion(payload.next_question);
  } catch (error) {
    window.alert(error.message);
  } finally {
    startBtn.disabled = false;
  }
}

async function restartSession() {
  if (!sessionId) {
    resetApp();
    return;
  }

  restartBtn.disabled = true;
  try {
    const response = await fetch(`/sessions/${sessionId}/restart`, {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to restart session.");
    }
    chatLog.innerHTML = "";
    appendBubble("system", "Session restarted. You are back at the first routed item.");
    resultsEl.hidden = true;
    sessionStage.hidden = false;
    renderQuestion(payload.next_question);
  } catch (error) {
    window.alert(error.message);
  } finally {
    restartBtn.disabled = false;
  }
}

function renderQuestion(question) {
  currentQuestion = question;
  if (!question) {
    questionArea.hidden = true;
    loadResults();
    return;
  }

  questionArea.hidden = false;
  updateProgress(question.progress);
  sessionTitle.textContent = `Question ${question.progress.answered + 1}`;
  questionText.textContent = question.text;
  appendBubble("system", question.text);

  responsesEl.innerHTML = "";
  for (const value of [1, 2, 3, 4, 5]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = responseLabels[value];
    button.disabled = submitting;
    button.addEventListener("click", () => submitResponse(value));
    responsesEl.appendChild(button);
  }
}

async function submitResponse(value) {
  if (submitting || !currentQuestion) {
    return;
  }

  submitting = true;
  setResponseButtonsDisabled(true);

  try {
    appendBubble("answer", responseLabels[value]);
    const response = await fetch(`/sessions/${sessionId}/responses`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_id: currentQuestion.item_id,
        response: value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to submit response.");
    }
    if (payload.complete) {
      questionArea.hidden = true;
      await loadResults();
      return;
    }
    renderQuestion(payload.next_question);
  } catch (error) {
    appendBubble("system", `Submission failed: ${error.message}`);
    window.alert(error.message);
  } finally {
    submitting = false;
    setResponseButtonsDisabled(false);
  }
}

async function loadResults() {
  const response = await fetch(`/sessions/${sessionId}/result`);
  const result = await response.json();
  if (!response.ok) {
    window.alert(result.detail || "Failed to load results.");
    return;
  }

  sessionStage.hidden = true;
  resultsEl.hidden = false;
  renderResultProgress(result.progress);
  resultNotice.textContent = result.disclaimer;
  renderExperimentContext(result);
  renderConfidence(result.progress, result.standard_errors, result.uncertainty);
  renderInterpretation(result.interpretation);
  renderCoverage(result.dimension_answer_counts, result.progress.answered, result.progress.max_items);
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

function renderInterpretation(interpretation) {
  interpretationOverview.textContent = interpretation.overview;
  interpretationRange.textContent = interpretation.range_summary;
  renderList(highlightList, interpretation.highlights);
  renderList(lowlightList, interpretation.lowlights);
  renderList(cautionList, interpretation.cautions);
}

function renderExperimentContext(result) {
  const mode = result.param_mode || "custom";
  paramModeLabel.textContent = `${mode}${result.key_aligned ? " | key-aligned" : " | response-flip mode"}`;
  paramPathLabel.textContent = result.param_path || "No parameter file recorded";
  paramMetaLabel.textContent = `generator ${result.param_metadata?.generator || "unknown"} | seed ${result.param_metadata?.seed ?? "n/a"}`;

  const stopReasons = [];
  if (result.progress.min_items_met) {
    stopReasons.push(`Minimum items met (${result.progress.answered}/${result.progress.min_items}).`);
  } else {
    stopReasons.push(`Minimum items gate still active (${result.progress.answered}/${result.progress.min_items}).`);
  }
  if (result.progress.coverage_ready) {
    stopReasons.push(`Coverage floor met at ${result.progress.coverage_min_per_dimension} item(s) per trait.`);
  } else {
    stopReasons.push(`Coverage floor not met yet: target ${result.progress.coverage_min_per_dimension} item(s) per trait.`);
  }
  if (result.progress.standard_error_ready) {
    stopReasons.push(`Mean standard error reached the ${Number(result.progress.stop_mean_standard_error).toFixed(2)} threshold.`);
  } else {
    stopReasons.push(`Mean standard error is still above the ${Number(result.progress.stop_mean_standard_error).toFixed(2)} threshold.`);
  }
  stopReasonLabel.textContent = `Current stop state: ${String(result.progress.stopped_by).replaceAll("_", " ")}.`;
  renderList(stopReasonList, stopReasons);
}

function exportSession() {
  if (!sessionId) {
    return;
  }
  window.open(`/sessions/${sessionId}/export`, "_blank", "noopener");
}

function updateProgress(progress) {
  const answered = progress.answered;
  const maxItems = progress.max_items;
  const percent = maxItems > 0 ? (answered / maxItems) * 100 : 0;
  progressLabel.textContent = `${answered} / ${maxItems} answered`;
  if (progress.complete) {
    progressHint.textContent = `Assessment complete | mean SE ${Number(progress.mean_standard_error ?? 0).toFixed(2)}`;
  } else {
    progressHint.textContent =
      `${progress.remaining} item${progress.remaining === 1 ? "" : "s"} remaining | mean SE ${Number(progress.mean_standard_error ?? 0).toFixed(2)}`;
  }
  progressBar.style.width = `${percent}%`;
}

function renderConfidence(progress, standardErrors, uncertainty) {
  meanStandardError.textContent = Number(uncertainty.mean_standard_error).toFixed(2);
  confidenceReady.textContent = uncertainty.confidence_ready ? "Yes" : "No";

  if (progress.stopped_by === "max_items_cap") {
    confidenceTitle.textContent = "Stopped at the item cap";
    confidenceCopy.textContent =
      `The session reached the ${progress.max_items}-item cap before the current stop rule was satisfied. Mean standard error finished at ${Number(uncertainty.mean_standard_error).toFixed(2)}.`;
  } else if (progress.stopped_by === "item_bank_exhausted") {
    confidenceTitle.textContent = "Item bank exhausted";
    confidenceCopy.textContent =
      `The routed item pool ran out before another prompt could improve certainty. Mean standard error finished at ${Number(uncertainty.mean_standard_error).toFixed(2)}.`;
  } else if (progress.stopped_by === "standard_error_threshold") {
    confidenceTitle.textContent = "Confidence threshold reached";
    confidenceCopy.textContent =
      `The session stopped before the ${progress.max_items}-item cap because the current confidence rule was satisfied. Mean standard error is ${Number(uncertainty.mean_standard_error).toFixed(2)} across tracked traits.`;
  } else if (progress.answered < progress.min_items) {
    confidenceTitle.textContent = "Still collecting minimum evidence";
    confidenceCopy.textContent =
      `The engine is still inside the minimum item window (${progress.answered}/${progress.min_items}). Early stopping is disabled until that floor is met.`;
  } else {
    confidenceTitle.textContent = "More evidence still needed";
    confidenceCopy.textContent =
      `The session has not yet met the stopping rule. Mean standard error is ${Number(uncertainty.mean_standard_error).toFixed(2)}, so the engine keeps routing more items while evidence improves.`;
  }

  confidenceGrid.innerHTML = "";
  for (const [trait, value] of Object.entries(standardErrors)) {
    const article = document.createElement("article");
    const name = document.createElement("p");
    const score = document.createElement("p");
    const caption = document.createElement("p");
    name.className = "metric-name";
    score.className = "metric-value metric-value--small";
    caption.className = "metric-caption";
    name.textContent = trait;
    score.textContent = Number(value).toFixed(2);
    caption.textContent = "Standard error";
    article.append(name, score, caption);
    confidenceGrid.appendChild(article);
  }
}

function renderResultProgress(progress) {
  if (progress.answered < progress.max_items) {
    resultProgress.textContent = `Stopped after ${progress.answered} answered items before the ${progress.max_items}-item cap.`;
    return;
  }
  resultProgress.textContent = `Reached the ${progress.max_items}-item cap.`;
}

function renderCoverage(counts, answeredItems, maxItems) {
  coverageGrid.innerHTML = "";
  for (const [trait, count] of Object.entries(counts)) {
    const article = document.createElement("article");
    const name = document.createElement("p");
    const value = document.createElement("p");
    const caption = document.createElement("p");
    name.className = "metric-name";
    value.className = "metric-value";
    caption.className = "metric-caption";
    name.textContent = trait;
    value.textContent = String(count);
    caption.textContent = `${count} of ${answeredItems} answered items | cap ${maxItems}`;
    article.append(name, value, caption);
    coverageGrid.appendChild(article);
  }
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

function renderList(target, items) {
  target.innerHTML = "";
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  }
}

function appendBubble(role, text) {
  const item = document.createElement("article");
  const label = document.createElement("span");
  const body = document.createElement("p");
  item.className = `bubble bubble--${role}`;
  label.className = "bubble-label";
  body.textContent = text;
  label.textContent = role === "system" ? "Prompt" : "Your answer";
  item.append(label, body);
  chatLog.appendChild(item);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setResponseButtonsDisabled(disabled) {
  for (const button of responsesEl.querySelectorAll("button")) {
    button.disabled = disabled;
  }
}

resetApp();
