const startBtn = document.querySelector("#startBtn");
const restartBtn = document.querySelector("#restartBtn");
const newSessionBtn = document.querySelector("#newSessionBtn");
const exportBtn = document.querySelector("#exportBtn");
const copyLinkBtn = document.querySelector("#copyLinkBtn");
const commentSubmitBtn = document.querySelector("#commentSubmitBtn");
const commentSkipBtn = document.querySelector("#commentSkipBtn");
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
const commentPanel = document.querySelector("#commentPanel");
const commentInput = document.querySelector("#commentInput");
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
const catImage = document.querySelector("#catImage");
const catName = document.querySelector("#catName");
const catAnalysis = document.querySelector("#catAnalysis");
const shareHash = document.querySelector("#shareHash");
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
  "early stop candidate": "Early-stop candidate",
  "confirmation window": "Confirmation check in progress",
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
copyLinkBtn.addEventListener("click", copyShareLink);
commentSubmitBtn.addEventListener("click", submitCommentAndLoadResult);
commentSkipBtn.addEventListener("click", loadResults);
paramModeInput.addEventListener("change", renderSetupSummary);
modelInput.addEventListener("change", renderSetupSummary);
maxItemsInput.addEventListener("input", renderSetupSummary);
minItemsInput.addEventListener("input", renderSetupSummary);
coverageMinInput.addEventListener("input", renderSetupSummary);
stopStabilityScoreInput.addEventListener("input", renderSetupSummary);

function resetApp(clearResultParam = true) {
  if (clearResultParam) {
    const url = new URL(window.location.href);
    url.searchParams.delete("result");
    window.history.replaceState({}, "", url);
  }
  sessionId = null;
  currentQuestion = null;
  submitting = false;
  setupStage.hidden = false;
  setupPanel.hidden = false;
  sessionStage.hidden = true;
  questionArea.hidden = false;
  commentPanel.hidden = true;
  resultsEl.hidden = true;
  advancedPanel.open = false;
  sessionDetails.open = false;
  experimentDetails.open = false;
  commentInput.value = "";
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
  catImage.removeAttribute("src");
  catImage.style.objectPosition = "";
  catName.textContent = "";
  catAnalysis.textContent = "";
  shareHash.textContent = "";
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
    commentInput.value = "";
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
    const response = await fetch(`/sessions/${sessionId}/restart`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to restart session.");
    }
    commentInput.value = "";
    chatLog.innerHTML = "";
    appendBubble("system", "Session restarted. You are back at the first routed item.");
    resultsEl.hidden = true;
    sessionStage.hidden = false;
    questionArea.hidden = false;
    commentPanel.hidden = true;
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
    commentPanel.hidden = false;
    updateProgress(
      { answered: Number(progressLabel.dataset.answered || 0), max_items: Number(maxItemsInput.value), complete: true },
      null
    );
    return;
  }
  commentPanel.hidden = true;
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
      body: JSON.stringify({ item_id: currentQuestion.item_id, response: value }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to submit response.");
    }
    if (payload.complete) {
      currentQuestion = null;
      questionArea.hidden = true;
      commentPanel.hidden = false;
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

async function submitCommentAndLoadResult() {
  const comment = commentInput.value.trim();
  if (!sessionId) {
    return;
  }
  if (comment) {
    commentSubmitBtn.disabled = true;
    try {
      const response = await fetch(`/sessions/${sessionId}/comments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comment }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to save comment.");
      }
      appendBubble("answer", `补充说明：${comment}`);
    } catch (error) {
      window.alert(error.message);
      commentSubmitBtn.disabled = false;
      return;
    } finally {
      commentSubmitBtn.disabled = false;
    }
  }
  await loadResults();
}

async function loadResults() {
  if (!sessionId) {
    return;
  }
  const response = await fetch(`/sessions/${sessionId}/result`);
  const result = await response.json();
  if (!response.ok) {
    window.alert(result.detail || "Failed to load results.");
    return;
  }
  renderResultPayload(result);
}

async function loadSharedResult(sharedSessionId) {
  const response = await fetch(`/results/${sharedSessionId}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Failed to load shared result.");
  }
  renderResultPayload(payload);
}

function renderResultPayload(result) {
  sessionId = result.session_id;
  currentQuestion = null;
  sessionStage.hidden = true;
  resultsEl.hidden = false;
  questionArea.hidden = true;
  commentPanel.hidden = true;
  setupStage.hidden = true;
  setupPanel.hidden = true;
  renderResultProgress(result.progress || {}, result.progress_estimate || {});
  resultNotice.textContent = result.disclaimer || "本系统仅作为心理特质筛查与辅助参考工具，绝对不可替代专业精神科临床诊断。";
  renderSessionConfig(result);
  renderExperimentContext(result);
  renderConfidence(result.progress || {}, result.progress_estimate || {}, result.standard_errors || {}, result.uncertainty || {});
  renderInterpretation(result.interpretation || { overview: "", range_summary: "", highlights: [], lowlights: [], cautions: [] });
  renderCoverage(result.dimension_answer_counts || {}, result.progress?.answered || 0);
  renderScores(irtScores, result.irt_t_scores || {});

  const classical = {};
  for (const [trait, score] of Object.entries(result.classical_big5 || {})) {
    classical[trait] =
      score.tendency_t_score === null
        ? `NA (${score.answered_count})`
        : `${Number(score.tendency_t_score).toFixed(1)} (${score.answered_count})${score.answered_count < 2 ? " low evidence" : ""}`;
  }
  renderScores(classicalScores, classical);
  renderCatResult(result);
}

function renderCatResult(result) {
  catImage.src = result.cat_image || "";
  catImage.hidden = !result.cat_image;
  catImage.style.objectPosition = result.cat_image_position || "50% 50%";
  catName.textContent = result.cat_name || "猫娘结果准备中";
  catAnalysis.textContent = result.cat_analysis || "这份结果还没有生成角色化文案。";
  shareHash.textContent = result.session_id || "";
}

function renderInterpretation(interpretation) {
  interpretationOverview.textContent = interpretation.overview || "";
  interpretationRange.textContent = interpretation.range_summary || "";
  renderList(highlightList, interpretation.highlights || []);
  renderList(lowlightList, interpretation.lowlights || []);
  renderList(cautionList, interpretation.cautions || []);
}

function renderSetupSummary() {
  const minItems = Math.min(Number(minItemsInput.value || 0), Number(maxItemsInput.value || 0));
  const modeLabel = paramModeInput.value === "keyed" ? "adaptive 30-item target" : "fast estimate";
  setupSummary.textContent =
    `Recommended ${modeLabel} | evidence floor ${minItems} | coverage ${coverageMinInput.value} | smart precision 0.85 -> 0.65 | stability ${Number(stopStabilityScoreInput.value).toFixed(2)}`;
}

function renderSessionConfig(payload) {
  const screeningSe = payload.progress?.screening_stop_mean_standard_error ?? 0.85;
  const refinementSe = payload.stop_mean_standard_error ?? 0.65;
  const trigger = payload.progress?.refinement_item_trigger ?? 15;
  sessionConfigLabel.textContent =
    `Mode ${payload.param_mode || "keyed"} | model ${payload.scoring_model || "binary_2pl"} | evidence floor ${payload.min_items ?? 5} | coverage ${payload.coverage_min_per_dimension ?? 2} | smart precision ${Number(screeningSe).toFixed(2)} -> ${Number(refinementSe).toFixed(2)} after item ${trigger} | stop stability ${Number(payload.stop_stability_score ?? 0.7).toFixed(2)} | hard cap ${payload.max_items ?? 30}`;
}

function renderExperimentContext(result) {
  const mode = result.param_mode || "custom";
  paramModeLabel.textContent = `${mode}${result.key_aligned ? " | parameter-aligned" : " | runtime reverse-key handling"}`;
  paramPathLabel.textContent = result.param_path || "No parameter file recorded";
  paramMetaLabel.textContent = `generator ${result.param_metadata?.generator || "unknown"} | seed ${result.param_metadata?.seed ?? "n/a"}`;
  const progress = result.progress || {};
  const stopReasons = [];
  if (progress.min_items_met) {
    stopReasons.push(`Minimum items met (${progress.answered}/${progress.min_items}).`);
  } else {
    stopReasons.push(`Minimum items gate still active (${progress.answered}/${progress.min_items}).`);
  }
  if (progress.coverage_ready) {
    stopReasons.push(`Coverage floor met at ${progress.coverage_min_per_dimension} item(s) per trait.`);
  } else {
    stopReasons.push(`Coverage floor not met yet: target ${progress.coverage_min_per_dimension} item(s) per trait.`);
  }
  if (progress.screening_threshold_ready) {
    stopReasons.push(`Early screening passed the ${Number(progress.screening_stop_mean_standard_error).toFixed(2)} threshold.`);
  } else {
    stopReasons.push(`Early screening is still above ${Number(progress.screening_stop_mean_standard_error).toFixed(2)}.`);
  }
  if (progress.early_stop_candidate) {
    stopReasons.push(`Checkpoint ${progress.candidate_checkpoint} triggered an early-stop candidate; ${progress.confirmation_items_remaining} confirmation item(s) remain.`);
  } else if (progress.stopped_by === "screening_confirmed") {
    stopReasons.push("The early-stop candidate held through the confirmation window and was promoted to a confirmed stop.");
  }
  if (progress.precision_mode === "refining") {
    if (progress.standard_error_ready) {
      stopReasons.push(`Refinement reached the active ${Number(progress.effective_stop_mean_standard_error).toFixed(2)} threshold.`);
    } else {
      stopReasons.push(`Refinement is still above the active ${Number(progress.effective_stop_mean_standard_error).toFixed(2)} threshold.`);
    }
  } else if (progress.stopped_by === "screening_plateau") {
    stopReasons.push(
      `The session passed item ${progress.refinement_item_trigger} without clearing the early ${Number(progress.screening_stop_mean_standard_error).toFixed(2)} screen, so it wrapped at the screening stage.`
    );
  }
  if (progress.stability_ready) {
    stopReasons.push(`Response stability reached the ${Number(progress.stop_stability_score).toFixed(2)} threshold.`);
  } else {
    stopReasons.push(`Response stability is still below the ${Number(progress.stop_stability_score).toFixed(2)} threshold.`);
  }
  stopReasonLabel.textContent = `Current stop state: ${String(progress.stopped_by || "pending").replaceAll("_", " ")}.`;
  renderList(stopReasonList, stopReasons);
}

function exportSession() {
  if (!sessionId) {
    return;
  }
  window.open(`/sessions/${sessionId}/export`, "_blank", "noopener");
}

async function copyShareLink() {
  if (!sessionId) {
    return;
  }
  const link = new URL(window.location.origin);
  link.searchParams.set("result", sessionId);
  await navigator.clipboard.writeText(link.toString());
  copyLinkBtn.textContent = "链接已复制";
  setTimeout(() => {
    copyLinkBtn.textContent = "复制专属结果链接";
  }, 1600);
}

function updateProgress(progress, progressEstimate) {
  const answered = Number(progress.answered || 0);
  progressLabel.dataset.answered = String(answered);
  const maxItems = progressEstimate?.estimated_total_items ?? progress.max_items ?? Number(maxItemsInput.value);
  const displayAnswered = progressEstimate?.display_answered ?? (progress.complete ? answered : answered + 1);
  const percent = progressEstimate?.estimated_completion_percent ?? (maxItems > 0 ? Math.round((displayAnswered / maxItems) * 100) : 0);
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
  meanStandardError.textContent = Number(uncertainty.mean_standard_error ?? 0).toFixed(2);
  confidenceReady.textContent = progress.standard_error_ready ? "Yes" : "No";
  stabilityScore.textContent = Number(progress.stability_score ?? 0).toFixed(2);
  stabilityStage.textContent = String(progress.stability_stage || "mixed");

  if (progress.stopped_by === "max_items_cap") {
    confidenceTitle.textContent = "Stopped at the item cap";
    confidenceCopy.textContent =
      `The session reached the ${progress.max_items}-item cap before the current stop rule was satisfied. Mean standard error finished at ${Number(uncertainty.mean_standard_error ?? 0).toFixed(2)}.`;
  } else if (progress.stopped_by === "screening_plateau") {
    confidenceTitle.textContent = "Wrapped at the screening stage";
    confidenceCopy.textContent =
      `The session moved past item ${progress.refinement_item_trigger} without clearing the early ${Number(progress.screening_stop_mean_standard_error).toFixed(2)} screen, so it stopped instead of chasing the stricter refinement target. Mean standard error finished at ${Number(uncertainty.mean_standard_error ?? 0).toFixed(2)}.`;
  } else if (progress.stopped_by === "item_bank_exhausted") {
    confidenceTitle.textContent = "Item bank exhausted";
    confidenceCopy.textContent =
      `The routed item pool ran out before another prompt could improve certainty. Mean standard error finished at ${Number(uncertainty.mean_standard_error ?? 0).toFixed(2)}.`;
  } else if (progress.stopped_by === "screening_candidate") {
    confidenceTitle.textContent = "已触发早停候选";
    confidenceCopy.textContent =
      `Checkpoint ${progress.candidate_checkpoint || "?"} already looks strong enough for an early stop, but the engine is waiting to start the final confirmation pass before ending the session.`;
  } else if (progress.stopped_by === "confirmation_window") {
    confidenceTitle.textContent = "补 2 题确认中";
    confidenceCopy.textContent =
      `The engine is using the last ${progress.confirmation_items_remaining} confirmation item(s) to verify that precision and response stability still hold after the early-stop candidate fired.`;
  } else if (progress.stopped_by === "screening_confirmed") {
    confidenceTitle.textContent = "智能早停已确认";
    confidenceCopy.textContent =
      `The session cleared checkpoint ${progress.candidate_checkpoint || progress.refinement_item_trigger}, completed the confirmation window, and still satisfied the early-stop confidence rule. Mean standard error is ${Number(uncertainty.mean_standard_error ?? 0).toFixed(2)}.`;
  } else if (progress.stopped_by === "screening_threshold") {
    confidenceTitle.textContent = "智能早停已触发";
    confidenceCopy.textContent =
      `The session stopped before item ${progress.refinement_item_trigger} because the early-stop ${Number(progress.screening_stop_mean_standard_error).toFixed(2)} screen and the current stability rule were both satisfied. Mean standard error is ${Number(uncertainty.mean_standard_error ?? 0).toFixed(2)}.`;
  } else if (progress.stopped_by === "stability_threshold") {
    confidenceTitle.textContent = "Confidence threshold reached";
    confidenceCopy.textContent =
      `The session moved past the early screen and stopped near the ${progressEstimate?.estimated_total_items ?? progress.max_items}-item estimate because the tighter ${Number(progress.effective_stop_mean_standard_error).toFixed(2)} refinement target and the stability rule were both satisfied. Mean standard error is ${Number(uncertainty.mean_standard_error ?? 0).toFixed(2)} and the response pattern is ${String(progress.stability_stage || "stable")}.`;
  } else if ((progress.answered || 0) < (progress.min_items || 0)) {
    confidenceTitle.textContent = "Still collecting minimum evidence";
    confidenceCopy.textContent =
      `The engine is still inside the minimum item window (${progress.answered}/${progress.min_items}). Early stopping is disabled until that floor is met.`;
  } else if (progress.stopped_by === "screening_gate") {
    confidenceTitle.textContent = "仍在判断是否可以早停";
    confidenceCopy.textContent =
      `The engine is still trying to clear the early-stop ${Number(progress.screening_stop_mean_standard_error).toFixed(2)} confidence screen before it decides whether a shorter session is enough.`;
  } else if (progress.stopped_by === "stability_gate") {
    confidenceTitle.textContent = "Checking for a stable pattern";
    confidenceCopy.textContent =
      `Coverage and precision are close, but the response pattern is still ${String(progress.stability_stage || "mixed")}. The engine keeps routing more items to confirm whether the signal settles down.`;
  } else {
    confidenceTitle.textContent = "More evidence still needed";
    confidenceCopy.textContent =
      `The session has not yet met the stopping rule. Mean standard error is ${Number(uncertainty.mean_standard_error ?? 0).toFixed(2)} and response stability is ${Number(progress.stability_score ?? 0).toFixed(2)}, so the engine keeps routing more items while evidence improves.`;
  }

  confidenceGrid.innerHTML = "";
  for (const [trait, value] of Object.entries(standardErrors || {})) {
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
  const estimated = progressEstimate?.estimated_total_items ?? progress.max_items ?? 0;
  const source = progressEstimate?.estimate_source === "lookup_table" ? "estimated path" : "custom path";
  resultProgress.textContent =
    `Answered ${progress.answered ?? 0} items | estimated total ${estimated} (${source}) | stability ${progress.stability_stage || "mixed"} | stop state ${String(progress.stop_state || progress.stopped_by || "pending").replaceAll("_", " ")}.`;
}

function renderCoverage(counts, answeredItems) {
  coverageGrid.innerHTML = "";
  for (const [trait, count] of Object.entries(counts || {})) {
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
  for (const [trait, score] of Object.entries(scores || {})) {
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

async function boot() {
  const params = new URLSearchParams(window.location.search);
  resetApp(false);
  const resultId = params.get("result");
  if (!resultId) {
    return;
  }
  try {
    await loadSharedResult(resultId);
  } catch (error) {
    window.alert(error.message);
    resetApp();
  }
}

boot();
