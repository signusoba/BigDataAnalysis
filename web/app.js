"use strict";

// All predictions and evaluation numbers come from the local API.
const $ = (id) => document.getElementById(id);
const state = { letters: [], overview: null, file: null, page: null, pageMode: "words", pagesLoaded: false, evaluationLoaded: false, activeTab: "recognize" };
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const percent = (value) => finite(value) ? `${(value * 100).toFixed(1)}%` : "Not measured";
const numeric = (value) => finite(value) ? value.toLocaleString() : "Not reported";

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options = {}) {
  let response;
  try { response = await fetch(path, options); }
  catch { throw new Error("The local lab server could not be reached. Check that it is running, then try again."); }
  const contentType = response.headers.get("content-type") || "";
  let data;
  if (contentType.includes("application/json")) data = await response.json();
  else if (!response.ok) throw new Error(`The server returned an error (${response.status}). Try again or check its logs.`);
  else throw new Error("The server returned an unexpected response. Open the app through its local lab server.");
  if (!response.ok) {
    const message = typeof data.error === "string" ? data.error : typeof data.detail === "string" ? data.detail : typeof data.message === "string" ? data.message : `The request failed (${response.status}).`;
    throw new Error(message);
  }
  return data;
}

function message(id, text, isError = false) {
  const node = $(id);
  node.textContent = text || "";
  node.hidden = !text;
  node.classList.toggle("error", isError);
}

let toastTimer;
function toast(text) {
  $("toast").textContent = text;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4500);
}

function busy(button, active, label, loadingLabel) {
  button.disabled = active;
  button.textContent = active ? loadingLabel : label;
  button.setAttribute("aria-busy", String(active));
}

function switchTab(name, focus = false) {
  const current = document.querySelector(`[data-tab="${name}"]`);
  if (!current) return;
  state.activeTab = name;
  for (const tab of document.querySelectorAll("[data-tab]")) {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    $(`panel-${tab.dataset.tab}`).hidden = !active;
  }
  $("breadcrumb-current").textContent = { recognize: "Recognize", dataset: "Dataset studio", evaluation: "Evaluation", alphabet: "Alphabet" }[name];
  if (focus) current.focus();
  if (name === "dataset" && !state.pagesLoaded) loadPages();
  if (name === "evaluation" && !state.evaluationLoaded) loadEvaluation();
  if (name === "dataset") requestAnimationFrame(() => pageCrop.resize());
  if (name === "recognize") requestAnimationFrame(() => letterCrop.resize());
}

const tabs = [...document.querySelectorAll("[data-tab]")];
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  tab.addEventListener("keydown", (event) => {
    let next;
    if (["ArrowDown", "ArrowRight"].includes(event.key)) next = (index + 1) % tabs.length;
    if (["ArrowUp", "ArrowLeft"].includes(event.key)) next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); switchTab(tabs[next].dataset.tab, true); }
  });
});

class ImageCrop {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.image = null;
    this.crop = null;
    this.options = options;
    this.drag = null;
    this.words = [];
    this.selectedWord = null;
    this.loadToken = 0;
    canvas.tabIndex = 0;
    canvas.setAttribute("aria-description", "Crop mode: Enter creates a centered selection; arrow keys move it; Shift plus arrows resizes it; Escape clears it. Word mode: arrow keys move through the provided annotations.");
    canvas.addEventListener("pointerdown", (event) => this.start(event));
    canvas.addEventListener("pointermove", (event) => this.move(event));
    canvas.addEventListener("pointerup", (event) => this.end(event));
    canvas.addEventListener("pointercancel", (event) => this.end(event));
    canvas.addEventListener("keydown", (event) => this.key(event));
    window.addEventListener("resize", () => this.resize());
  }

  async load(src) {
    const token = ++this.loadToken;
    const image = new Image();
    image.src = src;
    await image.decode();
    if (token !== this.loadToken) return;
    if (image.naturalWidth * image.naturalHeight > 16000000) throw new Error("This image is larger than 16 megapixels. Resize it first and keep the letter clear.");
    this.image = image;
    this.crop = null;
    this.selectedWord = null;
    this.resize();
    this.changed();
  }

  resize() {
    if (!this.image || this.canvas.closest("[hidden]")) return;
    const parentWidth = this.canvas.parentElement.clientWidth || 400;
    const ratio = Math.min(parentWidth / this.image.naturalWidth, (this.options.maxHeight || 330) / this.image.naturalHeight, 1);
    const width = Math.max(1, Math.round(this.image.naturalWidth * ratio));
    const height = Math.max(1, Math.round(this.image.naturalHeight * ratio));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.displayRatio = ratio;
    this.draw();
  }

  point(event) {
    const rect = this.canvas.getBoundingClientRect();
    return [
      Math.round(Math.max(0, Math.min(this.image.naturalWidth, (event.clientX - rect.left) / rect.width * this.image.naturalWidth))),
      Math.round(Math.max(0, Math.min(this.image.naturalHeight, (event.clientY - rect.top) / rect.height * this.image.naturalHeight)))
    ];
  }

  start(event) {
    if (!this.image || (event.button !== undefined && event.button !== 0)) return;
    const point = this.point(event);
    if (this.options.inspect && this.options.inspect()) {
      const word = this.words.find((item) => {
        const b = item.bbox;
        return Array.isArray(b) && point[0] >= b[0] && point[0] <= b[2] && point[1] >= b[1] && point[1] <= b[3];
      });
      this.selectedWord = word || null;
      this.draw();
      this.options.onWord?.(word);
      return;
    }
    event.preventDefault();
    this.canvas.setPointerCapture(event.pointerId);
    let anchor = point;
    if (this.crop) {
      const [l, t, r, b] = this.crop;
      const corners = [[l, t], [r, t], [r, b], [l, b]];
      const found = corners.findIndex((corner) => Math.hypot(point[0] - corner[0], point[1] - corner[1]) * this.displayRatio < 18);
      if (found !== -1) anchor = corners[(found + 2) % 4];
    }
    this.drag = { anchor, pointerId: event.pointerId };
    this.crop = [Math.min(anchor[0], point[0]), Math.min(anchor[1], point[1]), Math.max(anchor[0], point[0]), Math.max(anchor[1], point[1])];
    this.draw();
  }

  move(event) {
    if (!this.drag || event.pointerId !== this.drag.pointerId) return;
    event.preventDefault();
    const point = this.point(event);
    const anchor = this.drag.anchor;
    this.crop = [Math.min(anchor[0], point[0]), Math.min(anchor[1], point[1]), Math.max(anchor[0], point[0]), Math.max(anchor[1], point[1])];
    this.draw();
  }

  end(event) {
    if (!this.drag || event.pointerId !== this.drag.pointerId) return;
    this.move(event);
    this.drag = null;
    if (this.crop && ((this.crop[2] - this.crop[0]) * this.displayRatio < 3 || (this.crop[3] - this.crop[1]) * this.displayRatio < 3)) this.crop = null;
    if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
    this.draw();
    this.changed();
  }

  reset() { this.crop = null; this.draw(); this.changed(); }
  changed() { this.options.onChange?.(this.crop); }

  key(event) {
    if (!this.image) return;
    if (this.options.inspect?.()) {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key) || !this.words.length) return;
      event.preventDefault();
      let index = this.words.indexOf(this.selectedWord);
      if (event.key === "Home") index = 0;
      else if (event.key === "End") index = this.words.length - 1;
      else index = (index + (["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 1) + this.words.length) % this.words.length;
      this.selectedWord = this.words[index];
      this.draw();
      this.options.onWord?.(this.selectedWord);
      return;
    }
    const width = this.image.naturalWidth;
    const height = this.image.naturalHeight;
    if (event.key === "Escape") { event.preventDefault(); this.reset(); return; }
    if (event.key === "Enter" && !this.crop) {
      event.preventDefault();
      this.crop = [Math.floor(width * .25), Math.floor(height * .25), Math.ceil(width * .75), Math.ceil(height * .75)];
      this.draw(); this.changed(); return;
    }
    if (!this.crop || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const step = Math.max(1, Math.round(2 / this.displayRatio));
    const dx = event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0;
    const dy = event.key === "ArrowUp" ? -step : event.key === "ArrowDown" ? step : 0;
    let [l, t, r, b] = this.crop;
    if (event.shiftKey) { r = Math.max(l + 1, Math.min(width, r + dx)); b = Math.max(t + 1, Math.min(height, b + dy)); }
    else { const moveX = Math.max(-l, Math.min(width - r, dx)); const moveY = Math.max(-t, Math.min(height - b, dy)); l += moveX; r += moveX; t += moveY; b += moveY; }
    this.crop = [l, t, r, b];
    this.draw();
    this.changed();
  }

  draw() {
    if (!this.image) return;
    const ctx = this.context;
    const width = this.image.naturalWidth;
    const height = this.image.naturalHeight;
    const scale = this.canvas.width / width;
    const line = 1 / this.displayRatio;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(this.image, 0, 0);
    if (this.options.inspect?.()) {
      this.words.forEach((word) => {
        if (!Array.isArray(word.bbox)) return;
        const [l, t, r, b] = word.bbox;
        const selected = this.selectedWord === word;
        ctx.fillStyle = selected ? "rgba(19,121,110,.25)" : "rgba(19,121,110,.055)";
        ctx.strokeStyle = selected ? "#13796e" : "rgba(19,121,110,.4)";
        ctx.lineWidth = (selected ? 2 : .7) * line;
        ctx.fillRect(l, t, r - l, b - t);
        ctx.strokeRect(l, t, r - l, b - t);
      });
    }
    if (!this.crop) return;
    const [l, t, r, b] = this.crop;
    ctx.fillStyle = "rgba(18,43,38,.37)";
    ctx.beginPath();
    ctx.rect(0, 0, width, height);
    ctx.rect(l, t, r - l, b - t);
    ctx.fill("evenodd");
    ctx.strokeStyle = "#13796e";
    ctx.lineWidth = 2 * line;
    ctx.strokeRect(l, t, r - l, b - t);
    for (const [x, y] of [[l, t], [r, t], [r, b], [l, b]]) {
      ctx.fillStyle = "#fff";
      ctx.fillRect(x - 4 * line, y - 4 * line, 8 * line, 8 * line);
      ctx.strokeRect(x - 4 * line, y - 4 * line, 8 * line, 8 * line);
    }
  }

  outputCanvas() {
    if (!this.image) throw new Error("Choose an image first.");
    const [l, t, r, b] = this.crop || [0, 0, this.image.naturalWidth, this.image.naturalHeight];
    if (r <= l || b <= t) throw new Error("Select a larger crop.");
    const out = document.createElement("canvas");
    out.width = r - l;
    out.height = b - t;
    out.getContext("2d").drawImage(this.image, l, t, r - l, b - t, 0, 0, r - l, b - t);
    return out;
  }

  async blob() {
    return new Promise((resolve, reject) => this.outputCanvas().toBlob((blob) => blob ? resolve(blob) : reject(new Error("The selected crop could not be read.")), "image/png"));
  }
}

const letterCrop = new ImageCrop($("letter-canvas"), {
  maxHeight: 310,
  onChange(crop) {
    $("crop-description").textContent = crop ? `${crop[2] - crop[0]} × ${crop[3] - crop[1]} px · selected crop` : "Using the full image";
    clearPrediction();
    message("sample-feedback", "");
  }
});

const pageCrop = new ImageCrop($("page-canvas"), {
  maxHeight: 950,
  inspect: () => state.pageMode === "words",
  onWord: showWord,
  onChange(crop) {
    $("save-annotation").disabled = !crop;
    $("page-crop-preview").hidden = !crop;
    $("reset-page-crop").hidden = !crop;
    if (crop) {
      $("page-crop-image").src = pageCrop.outputCanvas().toDataURL("image/png");
      $("page-instruction").textContent = `${crop[2] - crop[0]} × ${crop[3] - crop[1]} px · include the entire character and every dot.`;
    } else updatePageInstruction();
    message("annotation-feedback", "");
  }
});

function clearPrediction() {
  $("prediction-empty").hidden = false;
  $("prediction-result").hidden = true;
  $("result-state").textContent = letterCrop.image ? "READY TO READ" : "AWAITING IMAGE";
  message("recognize-error", "");
}

async function loadLetterFile(file) {
  if (!file) return;
  message("recognize-error", "");
  if (!file.type.startsWith("image/")) { message("recognize-error", "Choose an image such as PNG, JPEG, or WEBP.", true); return; }
  if (file.size > 10 * 1024 * 1024) { message("recognize-error", "Please choose an image smaller than 10 MB.", true); return; }
  const url = URL.createObjectURL(file);
  try {
    $("letter-editor").hidden = false;
    await letterCrop.load(url);
    state.file = file;
    $("file-name").textContent = file.name;
    $("upload-zone").hidden = true;
    $("predict-button").disabled = !state.overview?.model?.trained;
    $("save-sample").disabled = state.letters.length === 0;
    if (!state.overview?.model?.trained) message("recognize-error", "The model is not trained yet. You can still review and save character crops.", true);
  } catch (error) {
    if (!state.file) $("letter-editor").hidden = true;
    message("recognize-error", error.message || "This image could not be decoded. Try PNG or JPEG.", true);
  } finally { URL.revokeObjectURL(url); }
}

$("letter-file").addEventListener("change", (event) => loadLetterFile(event.target.files[0]));
$("upload-zone").addEventListener("dragover", (event) => { event.preventDefault(); $("upload-zone").classList.add("drag-over"); });
$("upload-zone").addEventListener("dragleave", () => $("upload-zone").classList.remove("drag-over"));
$("upload-zone").addEventListener("drop", (event) => { event.preventDefault(); $("upload-zone").classList.remove("drag-over"); loadLetterFile(event.dataTransfer.files[0]); });
$("reset-crop").addEventListener("click", () => letterCrop.reset());

$("predict-button").addEventListener("click", async () => {
  if (!letterCrop.image) return;
  const button = $("predict-button");
  message("recognize-error", "");
  busy(button, true, "Recognize character ↗", "Reading your letter…");
  $("result-state").textContent = "READING…";
  try {
    const form = new FormData();
    form.append("file", await letterCrop.blob(), "letter-crop.png");
    const data = await api("/api/predict", { method: "POST", body: form });
    showPrediction(data);
  } catch (error) {
    clearPrediction();
    message("recognize-error", error.message, true);
  } finally { busy(button, false, "Recognize character ↗", ""); }
});

function showPrediction(data) {
  if (!data.prediction || typeof data.prediction.character !== "string") throw new Error("The model did not return a character prediction.");
  const prediction = data.prediction;
  $("prediction-empty").hidden = true;
  $("prediction-result").hidden = false;
  $("result-state").textContent = "MODEL PREDICTION";
  $("predicted-character").textContent = prediction.character;
  $("predicted-name").textContent = prediction.name;
  $("prediction-score").textContent = `${percent(prediction.score)} model score (uncalibrated)`;
  $("review-indicator").classList.toggle("needs-review", Boolean(data.needs_review));
  $("review-indicator").textContent = data.needs_review ? "Review needed · the model is uncertain about this letter." : "Review the character and its dots before accepting this label.";
  $("prediction-latency").textContent = finite(data.latency_ms) ? `${data.latency_ms.toFixed(1)} ms processing` : "";
  $("prediction-warnings").replaceChildren();
  for (const warning of data.warnings || []) $("prediction-warnings").append(element("p", "prediction-warning", warning));
  $("prediction-alternatives").replaceChildren();
  for (const alternative of (data.top_predictions || [prediction]).slice(0, 3)) {
    const li = element("li", "alternative");
    const glyph = element("span", "glyph", alternative.character);
    glyph.lang = "fa";
    glyph.dir = "rtl";
    const track = element("div", "score-track");
    const fill = element("div", "score-fill");
    fill.style.width = `${finite(alternative.score) ? Math.max(0, Math.min(100, alternative.score * 100)) : 0}%`;
    track.setAttribute("aria-hidden", "true");
    track.append(fill);
    li.append(glyph, element("span", "alt-name", alternative.name), track, element("span", "alt-score", percent(alternative.score)));
    $("prediction-alternatives").append(li);
  }
  const image = $("preprocessed-image");
  image.closest("details").hidden = !(typeof data.preprocessed_image === "string" && /^data:image\/(png|jpeg|webp);base64,/.test(data.preprocessed_image));
  if (!image.closest("details").hidden) image.src = data.preprocessed_image;
}

$("sample-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!letterCrop.image) { message("sample-feedback", "Choose an image and review the crop first.", true); return; }
  const button = $("save-sample");
  busy(button, true, "Save reviewed crop ＋", "Saving…");
  message("sample-feedback", "");
  try {
    const form = new FormData();
    form.append("file", await letterCrop.blob(), "reviewed-letter.png");
    form.append("character", $("sample-character").value);
    form.append("reviewer", $("sample-reviewer").value.trim());
    form.append("group_id", $("sample-group").value.trim());
    form.append("split", $("sample-split").value);
    form.append("source", $("capture-source").value);
    await api("/api/samples", { method: "POST", body: form });
    message("sample-feedback", "Reviewed crop saved. It becomes a training or evaluation example only after the dataset is rebuilt and the model is retrained.");
    toast("Reviewed character saved.");
  } catch (error) { message("sample-feedback", error.message, true); }
  finally { busy(button, false, "Save reviewed crop ＋", ""); }
});

async function loadOverview() {
  try {
    const overview = await api("/api/overview");
    state.overview = overview;
    state.letters = Array.isArray(overview.letters) ? overview.letters : [];
    const trained = Boolean(overview.model?.trained);
    $("model-status").classList.toggle("unavailable", !trained);
    $("model-status").replaceChildren(element("span", "tiny-dot"), document.createTextNode(trained ? `${modelName(overview.model.selected_model || overview.model.summary?.selected_model)} ready` : "Model awaiting training"));
    $("predict-button").disabled = !(letterCrop.image && trained);
    $("save-sample").disabled = !(letterCrop.image && state.letters.length);
    fillLetterSelects();
    renderAlphabet();
    renderDatasetStats(overview.dataset || {});
    message("global-error", "");
  } catch (error) {
    $("model-status").textContent = "Server unavailable";
    $("model-status").classList.add("unavailable");
    message("global-error", error.message, true);
    $("alphabet-grid").replaceChildren(element("p", "field-hint", "The alphabet could not be loaded. Reconnect to the local lab server and reload."));
  }
}

function fillLetterSelects() {
  for (const id of ["sample-character", "annotation-character"]) {
    const select = $(id);
    select.replaceChildren();
    const placeholder = element("option", "", "Choose a letter");
    placeholder.value = "";
    select.append(placeholder);
    for (const letter of state.letters) {
      const option = element("option", "", `${letter.character} · ${letter.name}`);
      option.value = letter.character;
      select.append(option);
    }
  }
}

function renderAlphabet() {
  $("alphabet-grid").replaceChildren();
  for (const letter of state.letters) {
    const card = element("div", "alphabet-letter");
    const glyph = element("span", "glyph", letter.character);
    glyph.lang = "fa";
    glyph.dir = "rtl";
    card.append(element("span", "letter-index", String(letter.index + 1).padStart(2, "0")), glyph, element("span", "letter-name", letter.name));
    $("alphabet-grid").append(card);
  }
}

function stat(value, label) {
  const node = element("div", "stat-tile");
  node.append(element("strong", "", value), element("span", "", label));
  return node;
}

function renderDatasetStats(dataset) {
  $("dataset-stats").replaceChildren(
    stat(numeric(dataset.pages), "Source pages"),
    stat(numeric(dataset.words), "Provided word labels"),
    stat(numeric(dataset.unique_words), "Unique word transcriptions"),
    stat(String(state.letters.length), "Target letter classes")
  );
}

let pageRequest = 0;
async function loadPages() {
  state.pagesLoaded = true;
  message("dataset-error", "");
  try {
    const data = await api("/api/dataset/pages");
    const pages = data.pages || [];
    $("page-select").replaceChildren();
    if (!pages.length) {
      $("page-select").append(element("option", "", "No pages available"));
      $("page-empty").textContent = "The Arshasb dataset has not been prepared. See the project setup instructions.";
    } else {
      for (const page of pages) {
        const option = element("option", "", `Page ${page.id} · ${page.word_count ?? "?"} words`);
        option.value = String(page.id);
        $("page-select").append(option);
      }
      await loadPage(pages[0].id);
    }
    await refreshAnnotationCount();
  } catch (error) {
    state.pagesLoaded = false;
    $("page-empty").textContent = "Dataset could not be loaded.";
    message("dataset-error", error.message, true);
  }
}

async function loadPage(id) {
  const request = ++pageRequest;
  $("page-empty").textContent = "Loading page…";
  $("page-empty").hidden = false;
  $("page-canvas").hidden = true;
  $("save-annotation").disabled = true;
  $("page-crop-preview").hidden = true;
  showWord(null);
  message("dataset-error", "");
  try {
    const page = await api(`/api/dataset/pages/${encodeURIComponent(id)}`);
    if (request !== pageRequest) return;
    state.page = page;
    pageCrop.words = Array.isArray(page.words) ? page.words : [];
    $("page-canvas").hidden = false;
    await pageCrop.load(page.image_url);
    if (request !== pageRequest) return;
    $("page-empty").hidden = true;
    $("annotation-group").value = `arshasb-${page.id}`;
    updatePageInstruction();
  } catch (error) {
    if (request !== pageRequest) return;
    $("page-canvas").hidden = true;
    $("page-empty").textContent = "This page could not be opened.";
    message("dataset-error", error.message, true);
  }
}

$("page-select").addEventListener("change", (event) => loadPage(event.target.value));

function showWord(word) {
  $("word-empty").hidden = Boolean(word);
  $("word-result").hidden = !word;
  if (!word) return;
  $("word-text").textContent = word.text;
  $("word-meta").textContent = `Word ${word.index} · line ${word.line}`;
}

function updatePageInstruction() {
  $("page-instruction").textContent = state.pageMode === "words" ? "Click a word to inspect its provided annotation." : "Drag around one complete character, including its dots.";
}

function setPageMode(mode) {
  state.pageMode = mode;
  $("mode-words").classList.toggle("active", mode === "words");
  $("mode-crop").classList.toggle("active", mode === "crop");
  $("mode-words").setAttribute("aria-pressed", String(mode === "words"));
  $("mode-crop").setAttribute("aria-pressed", String(mode === "crop"));
  $("page-canvas-stage").classList.toggle("cropping", mode === "crop");
  pageCrop.reset();
  updatePageInstruction();
}

$("mode-words").addEventListener("click", () => setPageMode("words"));
$("mode-crop").addEventListener("click", () => setPageMode("crop"));
$("reset-page-crop").addEventListener("click", () => pageCrop.reset());

$("annotation-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.page || !pageCrop.crop) { message("annotation-feedback", "Draw a complete character crop on the page first.", true); return; }
  const button = $("save-annotation");
  busy(button, true, "Save annotation ＋", "Saving…");
  message("annotation-feedback", "");
  try {
    await api("/api/annotations", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ page_id: state.page.id, bbox: pageCrop.crop, character: $("annotation-character").value, reviewer: $("annotation-reviewer").value.trim(), group_id: $("annotation-group").value.trim(), split: $("annotation-split").value })
    });
    message("annotation-feedback", "Character annotation saved. Labels are reviewed training data; they are not model predictions.");
    toast("Character annotation saved.");
    refreshAnnotationCount();
  } catch (error) { message("annotation-feedback", error.message, true); }
  finally { busy(button, false, "Save annotation ＋", ""); }
});

async function refreshAnnotationCount() {
  try {
    const data = await api("/api/annotations");
    $("annotation-count").textContent = `${numeric(data.total)} saved character annotation${data.total === 1 ? "" : "s"} · export includes reviewed labels.`;
  } catch { $("annotation-count").textContent = "Saved annotation count is currently unavailable."; }
}

function modelName(name) {
  if (!name) return "Model";
  if (name === "cnn") return "CNN";
  if (name === "hog") return "HOG + SVM";
  if (name === "mobilenet") return "MobileNetV2";
  return String(name).replaceAll("_", " ");
}

async function loadEvaluation() {
  state.evaluationLoaded = true;
  const button = $("refresh-evaluation");
  busy(button, true, "Refresh report ↻", "Loading…");
  message("evaluation-error", "");
  try {
    const report = await api("/api/evaluation");
    renderEvaluation(report);
  } catch (error) {
    state.evaluationLoaded = false;
    $("evaluation-content").replaceChildren();
    const empty = element("div", "card eval-empty");
    empty.append(element("h2", "", "No evaluation report available."), element("p", "", "Train and evaluate the models using the project commands, then refresh this report. Metrics appear only after they have been measured."));
    $("evaluation-content").append(empty);
    message("evaluation-error", error.message, true);
  } finally { busy(button, false, "Refresh report ↻", ""); }
}

$("refresh-evaluation").addEventListener("click", loadEvaluation);

function renderEvaluation(report) {
  const container = $("evaluation-content");
  container.replaceChildren();
  const models = report.models || {};
  const entries = Object.entries(models);
  if (!entries.length) {
    const empty = element("div", "card eval-empty");
    empty.append(element("h2", "", "No model measurements yet."), element("p", "", "Train and evaluate a model to create its measured validation and test report."));
    container.append(empty);
    return;
  }
  const selected = report.selected_model;
  const heldOut = models[selected]?.test;
  const summary = element("div", "evaluation-summary");
  summary.append(stat(modelName(selected), "Selected using validation data"), stat(percent(heldOut?.accuracy), "Selected model · test accuracy"), stat(percent(heldOut?.macro_f1), "Selected model · test macro F1"));
  container.append(summary);
  const comparison = element("section", "card report-card");
  comparison.append(element("span", "step-label", "MEASURED RESULTS"), element("h2", "", "Model comparison"), element("p", "field-hint", "Validation selects the model. The test set measures the selected model on held-out examples. Scores below describe only this report's data domain."));
  const scroll = element("div", "table-scroll");
  const table = element("table", "comparison-table");
  const caption = element("caption", "visually-hidden", "Measured model accuracy, macro F1, precision and recall by evaluation split.");
  table.append(caption);
  const head = document.createElement("thead");
  const row = document.createElement("tr");
  for (const title of ["MODEL", "SPLIT", "ACCURACY", "MACRO F1", "PRECISION", "RECALL", "EXAMPLES"]) { const th = element("th", "", title); th.scope = "col"; row.append(th); }
  head.append(row);
  table.append(head);
  const body = document.createElement("tbody");
  for (const [key, model] of entries) {
    for (const split of ["validation", "test"]) {
      const metrics = model[split];
      if (!metrics) continue;
      const tr = element("tr", key === selected ? "selected" : "");
      const name = element("td", "", modelName(key));
      if (key === selected) name.append(element("span", "table-tag", "SELECTED"));
      tr.append(name, element("td", "", split === "test" ? "Test" : "Validation"), element("td", "", percent(metrics.accuracy)), element("td", "", percent(metrics.macro_f1)), element("td", "", percent(metrics.macro_precision)), element("td", "", percent(metrics.macro_recall)), element("td", "", numeric(metrics.count)));
      body.append(tr);
    }
  }
  table.append(body);
  scroll.append(table);
  comparison.append(scroll);
  if (finite(heldOut?.accuracy_ci95?.[0]) && finite(heldOut?.accuracy_ci95?.[1])) comparison.append(element("p", "field-hint", `Selected test accuracy, 95% interval: ${percent(heldOut.accuracy_ci95[0])}–${percent(heldOut.accuracy_ci95[1])}. A finite test set cannot establish perfect accuracy on every future image.`));
  container.append(comparison);

  const dataCard = element("section", "card report-card");
  dataCard.append(element("span", "step-label", "KNOW THE DATA"), element("h2", "", "Where these results apply"), element("p", "field-hint", report.data?.domain || "The report does not describe its data domain."));
  for (const split of ["train", "validation", "test"]) {
    const count = report.data?.counts?.[split];
    if (finite(count)) dataCard.append(element("span", "report-pill", `${split === "train" ? "Training" : split === "test" ? "Test" : "Validation"}: ${numeric(count)}`));
  }
  const cameraMetrics = heldOut?.by_source?.camera;
  dataCard.append(element("span", "report-pill", finite(cameraMetrics?.count)
    ? `Camera test: ${numeric(cameraMetrics.count)} images · ${percent(cameraMetrics.accuracy)} accuracy`
    : "Unseen camera images: not evaluated"));
  const timing = report.deployment_latency_ms;
  if (finite(timing?.p95)) {
    dataCard.append(element("p", "field-hint", `Selected model response: ${timing.median.toFixed(1)} ms median · ${timing.p95.toFixed(1)} ms p95 over ${numeric(timing.count)} validation inputs. Includes preprocessing and prediction; excludes capture, upload, and network time.`));
  }
  const limitations = element("ul", "limitations-list");
  for (const limit of report.limitations || []) limitations.append(element("li", "", typeof limit === "string" ? limit : JSON.stringify(limit)));
  limitations.append(element("li", "", "Manual Arshasb page annotations and synthetic letter results do not measure recognition accuracy on new handwriting or phone photographs."));
  dataCard.append(limitations);
  container.append(dataCard);

  const matrices = [];
  for (const [key, model] of entries) for (const split of ["test", "validation"]) if (Array.isArray(model[split]?.confusion_matrix)) matrices.push({ key, split, metrics: model[split] });
  if (matrices.length) renderConfusionSection(container, matrices, selected);
  if (Array.isArray(heldOut?.per_class)) {
    const card = element("section", "card report-card");
    card.append(element("span", "step-label", "EVERY CLASS COUNTS"), element("h2", "", "Per-letter test results"));
    const wrapper = element("div", "table-scroll");
    const table = element("table", "comparison-table");
    const head = document.createElement("thead");
    const header = document.createElement("tr");
    for (const title of ["LETTER", "NAME", "PRECISION", "RECALL", "F1", "EXAMPLES"]) {
      const cell = element("th", "", title); cell.scope = "col"; header.append(cell);
    }
    head.append(header);
    const body = document.createElement("tbody");
    for (const item of heldOut.per_class) {
      const row = document.createElement("tr");
      const glyph = element("th", "persian", item.character); glyph.scope = "row"; glyph.lang = "fa";
      row.append(glyph, element("td", "", item.name), element("td", "", percent(item.precision)), element("td", "", percent(item.recall)), element("td", "", percent(item.f1)), element("td", "", numeric(item.support)));
      body.append(row);
    }
    table.append(head, body); wrapper.append(table); card.append(wrapper); container.append(card);
  }
}

function renderConfusionSection(container, matrices, selected) {
  const card = element("section", "card report-card");
  const header = element("div", "confusion-controls");
  const heading = element("div");
  heading.append(element("span", "step-label", "LOOK AT THE MISTAKES"), element("h2", "", "Confusion matrix"));
  const label = element("label", "", "Model & split");
  const select = document.createElement("select");
  matrices.forEach((entry, index) => { const option = element("option", "", `${modelName(entry.key)} · ${entry.split}`); option.value = String(index); select.append(option); });
  const defaultIndex = matrices.findIndex((entry) => entry.key === selected && entry.split === "test");
  select.value = String(Math.max(0, defaultIndex));
  label.append(select);
  header.append(heading, label);
  const scroll = element("div", "table-scroll");
  card.append(header, element("p", "field-hint", "Rows are true letters; columns are predicted letters. Darker cells contain more examples."), scroll, element("p", "matrix-caption", "Scroll horizontally to inspect every letter. A diagonal cell is a correct prediction."));
  function draw() {
    const entry = matrices[Number(select.value)];
    const matrix = entry.metrics.confusion_matrix;
    const letters = entry.metrics.per_class || state.letters;
    const nameFor = (index) => letters[index]?.character || state.letters[index]?.character || String(index);
    const maximum = Math.max(1, ...matrix.flat().filter(finite));
    const table = element("table", "matrix-table");
    const caption = element("caption", "visually-hidden", `${modelName(entry.key)} ${entry.split} confusion matrix. Rows are true labels, columns predictions.`);
    table.append(caption);
    const thead = document.createElement("thead");
    const first = document.createElement("tr");
    const corner = element("th", "", "↓ / →"); corner.scope = "col"; first.append(corner);
    matrix.forEach((_, index) => { const th = element("th", "", nameFor(index)); th.scope = "col"; th.lang = "fa"; first.append(th); });
    thead.append(first);
    table.append(thead);
    const tbody = document.createElement("tbody");
    matrix.forEach((values, trueIndex) => {
      const tr = document.createElement("tr");
      const th = element("th", "", nameFor(trueIndex)); th.scope = "row"; th.lang = "fa"; tr.append(th);
      values.forEach((value, predictedIndex) => {
        const td = element("td", trueIndex === predictedIndex ? "diagonal" : "", String(value));
        const density = finite(value) ? value / maximum : 0;
        td.style.backgroundColor = `rgba(19,121,110,${(density * .78 + .035).toFixed(3)})`;
        if (density > .6) td.style.color = "#fff";
        td.title = `True ${nameFor(trueIndex)} → predicted ${nameFor(predictedIndex)}: ${value}`;
        tr.append(td);
      });
      tbody.append(tr);
    });
    table.append(tbody);
    scroll.replaceChildren(table);
  }
  select.addEventListener("change", draw);
  draw();
  container.append(card);
}

loadOverview();
