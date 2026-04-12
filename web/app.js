const startBtn = document.querySelector("#startBtn");
const restartBtn = document.querySelector("#restartBtn");
const newSessionBtn = document.querySelector("#newSessionBtn");
const exportBtn = document.querySelector("#exportBtn");
const paramModeInput = document.querySelector("#paramMode");
const modelInput = document.querySelector("#model");
const maxItemsInput = document.querySelector("#maxItems");
const minItemsInput = document.querySelector("#minItems");
const coverageMinInput = document.querySelector("#coverageMin");
const stopStabilityScoreInput = document.querySelector("#stopStabilityScore");
const setupStage = document.querySelector("#setupStage");
const setupPanel = document.querySelector("#setupPanel");
const setupSummary = document.querySelector("#setupSummary");
const advancedPanel = document.querySelector("#advancedPanel");
const sessionStage = document.querySelector("#sessionStage");
const questionArea = document.querySelector("#questionArea");
const sessionTitle = document.querySelector("#sessionTitle");
const sessionDetails = document.querySelector("#sessionDetails");
const sessionConfigLabel = document.querySelector("#sessionConfigLabel");
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
const stabilityScore = document.querySelector("#stabilityScore");
const stabilityStage = document.querySelector("#stabilityStage");
const confidenceGrid = document.querySelector("#confidenceGrid");
const interpretationOverview = document.querySelector("#interpretationOverview");
const interpretationRange = document.querySelector("#interpretationRange");
const highlightList = document.querySelector("#highlightList");
const lowlightList = document.querySelector("#lowlightList");
const cautionList = document.querySelector("#cautionList");
const coverageGrid = document.querySelector("#coverageGrid");
const irtScores = document.querySelector("#irtScores");
const classicalScores = document.querySelector("#classicalScores");
const experimentDetails = document.querySelector("#experimentDetails");

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

const evidenceStageCopy = {
  "building minimum evidence": "Building minimum evidence",
  "coverage in progress": "Coverage in progress",
  "early confidence screening": "Early confidence screening",
  "confidence refining": "Confidence refining",
  "stability refining": "Stability refining",
  "confidence target reached": "Confidence target reached",
  "screening plateau reached": "Screening plateau reached",
  "item cap reached": "Item cap reached",
  "item bank exhausted": "Item bank exhausted",
};

startBtn.addEventListener("click", startSession);
restartBtn.addEventListener("click", restartSession);
newSessionBtn.addEventListener("click", resetApp);
exportBtn.addEventListener("click", exportSession);
paramModeInput.addEventListener("change", renderSetupSummary);
modelInput.addEventListener("change", renderSetupSummary);
maxItemsInput.addEventListener("input", renderSetupSummary);
minItemsInput.addEventListener("input", renderSetupSummary);
coverageMinInput.addEventListener("input", renderSetupSummary);
  stopStabilityScoreInput.addEventListener("input", renderSetupSummary);

function resetApp() {
  sessionId = null;
  currentQuestion = null;
  submitting = false;
  setupStage.hidden = false;
  setupPanel.hidden = false;
  sessionStage.hidden = true;
  resultsEl.hidden = true;
  questionArea.hidden = false;
  advancedPanel.open = false;
  sessionDetails.open = false;
  experimentDetails.open = false;
  sessionConfigLabel.textContent = "";
  setupSummary.textContent = "";
  questionText.textContent = "";
  responsesEl.innerHTML = "";
  chatLog.innerHTML = "";
  coverageGrid.innerHTML = "";
  confidenceTitle.textContent = "";
  confidenceCopy.textContent = "";
  meanStandardError.textContent = "";
  confidenceReady.textContent = "";
  stabilityScore.textContent = "";
  stabilityStage.textContent = "";
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
  updateProgress(
    { answered: 0, max_items: Number(maxItemsInput.value), remaining: Number(maxItemsInput.value), complete: false },
    null
  );
  renderSetupSummary();
}

async function startSession() {
  startBtn.disabled = true;
  try {
    const maxItems = Number(maxItemsInput.value);
    const minItems = Number(minItemsInput.value);
    const coverageMin = Number(coverageMinInput.value);
    const stopStabilityScore = Number(stopStabilityScoreInput.value);
    const response = await fetch("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scoring_model: modelInput.value,
        max_items: maxItems,
        min_items: Math.min(minItems, maxItems),
        param_mode: paramModeInput.value,
        coverage_min_per_dimension: coverageMin,
        stop_stability_score: stopStabilityScore,
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
    renderSessionConfig(payload);
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
    renderSessionConfig(payload);
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
  updateProgress(question.progress, question.progress_estimate);
  sessionTitle.textContent = `Adaptive check-in | prompt ${question.progress.answered + 1}`;
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
  renderResultProgress(result.progress, result.progress_estimate);
  resultNotice.textContent = result.disclaimer;
  renderSessionConfig(result);
  renderExperimentContext(result);
  renderConfidence(result.progress, result.progress_estimate, result.standard_errors, result.uncertainty);
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

function renderSetupSummary() {
  const minItems = Math.min(Number(minItemsInput.value || 0), Number(maxItemsInput.value || 0));
  const modeLabel = paramModeInput.value === "keyed" ? "adaptive 30-item target" : "fast estimate";
  setupSummary.textContent =
    `Recommended ${modeLabel} | evidence floor ${minItems} | coverage ${coverageMinInput.value} | smart precision 0.85 → 0.65 | stability ${Number(stopStabilityScoreInput.value).toFixed(2)}`;
}

function renderSessionConfig(payload) {
  sessionConfigLabel.textContent =
    `Mode ${payload.param_mode} | model ${payload.scoring_model} | evidence floor ${payload.min_items} | coverage ${payload.coverage_min_per_dimension} | smart precision ${Number(payload.progress?.screening_stop_mean_standard_error ?? 0.85).toFixed(2)} → ${Number(payload.stop_mean_standard_error).toFixed(2)} after item ${payload.progress?.refinement_item_trigger ?? 15} | stop stability ${Number(payload.stop_stability_score).toFixed(2)} | hard cap ${payload.max_items}`;
}

function renderExperimentContext(result) {
  const mode = result.param_mode || "custom";
  paramModeLabel.textContent = `${mode}${result.key_aligned ? " | parameter-aligned" : " | runtime reverse-key handling"}`;
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
  if (result.progress.screening_threshold_ready) {
    stopReasons.push(`Early screening passed the ${Number(result.progress.screening_stop_mean_standard_error).toFixed(2)} threshold.`);
  } else {
    stopReasons.push(`Early screening is still above ${Number(result.progress.screening_stop_mean_standard_error).toFixed(2)}.`);
  }
  if (result.progress.precision_mode === "refining") {
    if (result.progress.standard_error_ready) {
      stopReasons.push(`Refinement reached the active ${Number(result.progress.effective_stop_mean_standard_error).toFixed(2)} threshold.`);
    } else {
      stopReasons.push(`Refinement is still above the active ${Number(result.progress.effective_stop_mean_standard_error).toFixed(2)} threshold.`);
    }
  } else if (result.progress.stopped_by === "screening_plateau") {
    stopReasons.push(
      `The session passed item ${result.progress.refinement_item_trigger} without clearing the early ${Number(result.progress.screening_stop_mean_standard_error).toFixed(2)} screen, so it wrapped at the screening stage.`
    );
  }
  if (result.progress.stability_ready) {
    stopReasons.push(`Response stability reached the ${Number(result.progress.stop_stability_score).toFixed(2)} threshold.`);
  } else {
    stopReasons.push(`Response stability is still below the ${Number(result.progress.stop_stability_score).toFixed(2)} threshold.`);
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

function updateProgress(progress, progressEstimate) {
  const answered = progress.answered;
  const maxItems = progressEstimate?.estimated_total_items ?? progress.max_items;
  const displayAnswered = progressEstimate?.display_answered ?? (progress.complete ? answered : answered + 1);
  const percent = progressEstimate?.estimated_completion_percent ?? (maxItems > 0 ? (displayAnswered / maxItems) * 100 : 0);
  progressLabel.textContent = `${displayAnswered} / ${maxItems} estimated | ${percent}%`;
  if (progress.complete) {
    progressHint.textContent =
      `${progressEstimate?.confidence_profile || "session complete"} | ${evidenceStageCopy[progressEstimate?.evidence_stage] || "Assessment complete"} | stability ${progress.stability_stage || "mixed"} | mean SE ${Number(progress.mean_standard_error ?? 0).toFixed(2)}`;
  } else {
    const estimateLabel = progressEstimate?.estimate_source === "lookup_table" ? "Estimated path" : "Custom path";
    progressHint.textContent =
      `${estimateLabel} | ${evidenceStageCopy[progressEstimate?.evidence_stage] || "Adaptive routing is active"} | stability ${progress.stability_stage || "mixed"} | mean SE ${Number(progress.mean_standard_error ?? 0).toFixed(2)}`;
  }
  progressBar.style.width = `${percent}%`;
}

function renderConfidence(progress, progressEstimate, standardErrors, uncertainty) {
  meanStandardError.textContent = Number(uncertainty.mean_standard_error).toFixed(2);
  confidenceReady.textContent = progress.standard_error_ready ? "Yes" : "No";
  stabilityScore.textContent = Number(progress.stability_score ?? 0).toFixed(2);
  stabilityStage.textContent = String(progress.stability_stage || "mixed");

  if (progress.stopped_by === "max_items_cap") {
    confidenceTitle.textContent = "Stopped at the item cap";
    confidenceCopy.textContent =
      `The session reached the ${progress.max_items}-item cap before the current stop rule was satisfied. Mean standard error finished at ${Number(uncertainty.mean_standard_error).toFixed(2)}.`;
  } else if (progress.stopped_by === "screening_plateau") {
    confidenceTitle.textContent = "Wrapped at the screening stage";
    confidenceCopy.textContent =
      `The session moved past item ${progress.refinement_item_trigger} without clearing the early ${Number(progress.screening_stop_mean_standard_error).toFixed(2)} screen, so it stopped instead of chasing the stricter refinement target. Mean standard error finished at ${Number(uncertainty.mean_standard_error).toFixed(2)}.`;
  } else if (progress.stopped_by === "item_bank_exhausted") {
    confidenceTitle.textContent = "Item bank exhausted";
    confidenceCopy.textContent =
      `The routed item pool ran out before another prompt could improve certainty. Mean standard error finished at ${Number(uncertainty.mean_standard_error).toFixed(2)}.`;
  } else if (progress.stopped_by === "screening_threshold") {
    confidenceTitle.textContent = "Early confidence screen passed";
    confidenceCopy.textContent =
      `The session stopped before item ${progress.refinement_item_trigger} because the early ${Number(progress.screening_stop_mean_standard_error).toFixed(2)} screen and the current stability rule were both satisfied. Mean standard error is ${Number(uncertainty.mean_standard_error).toFixed(2)}.`;
  } else if (progress.stopped_by === "stability_threshold") {
    confidenceTitle.textContent = "Confidence threshold reached";
    confidenceCopy.textContent =
      `The session moved past the early screen and stopped near the ${progressEstimate?.estimated_total_items ?? progress.max_items}-item estimate because the tighter ${Number(progress.effective_stop_mean_standard_error).toFixed(2)} refinement target and the stability rule were both satisfied. Mean standard error is ${Number(uncertainty.mean_standard_error).toFixed(2)} and the response pattern is ${String(progress.stability_stage || "stable")}.`;
  } else if (progress.answered < progress.min_items) {
    confidenceTitle.textContent = "Still collecting minimum evidence";
    confidenceCopy.textContent =
      `The engine is still inside the minimum item window (${progress.answered}/${progress.min_items}). Early stopping is disabled until that floor is met.`;
  } else if (progress.stopped_by === "screening_gate") {
    confidenceTitle.textContent = "Still clearing the early screen";
    confidenceCopy.textContent =
      `The engine is still trying to clear the early ${Number(progress.screening_stop_mean_standard_error).toFixed(2)} confidence screen before it decides whether a shorter session is enough.`;
  } else if (progress.stopped_by === "stability_gate") {
    confidenceTitle.textContent = "Checking for a stable pattern";
    confidenceCopy.textContent =
      `Coverage and precision are close, but the response pattern is still ${String(progress.stability_stage || "mixed")}. The engine keeps routing more items to confirm whether the signal settles down.`;
  } else {
    confidenceTitle.textContent = "More evidence still needed";
    confidenceCopy.textContent =
      `The session has not yet met the stopping rule. Mean standard error is ${Number(uncertainty.mean_standard_error).toFixed(2)} and response stability is ${Number(progress.stability_score ?? 0).toFixed(2)}, so the engine keeps routing more items while evidence improves.`;
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

function renderResultProgress(progress, progressEstimate) {
  const estimated = progressEstimate?.estimated_total_items ?? progress.max_items;
  const source = progressEstimate?.estimate_source === "lookup_table" ? "estimated path" : "custom path";
  resultProgress.textContent =
    `Answered ${progress.answered} items | estimated total ${estimated} (${source}) | stability ${progress.stability_stage || "mixed"} | stop state ${String(progress.stopped_by).replaceAll("_", " ")}.`;
}

function renderCoverage(counts, answeredItems) {
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
    caption.textContent = `${count} of ${answeredItems} answered items`;
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
