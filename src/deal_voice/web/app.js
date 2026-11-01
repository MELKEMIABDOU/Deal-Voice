"use strict";

const $ = (id) => document.getElementById(id);
const state = { documents: [], selected: new Set(), ready: false, remoteBusy: false, localBusy: false, limits: { files: 5, upload_mb: 100, audio_minutes: 120 } };
const allowedExtensions = new Set(["pdf", "mp3", "wav", "m4a", "flac"]);

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = String(text);
  return item;
}

function notice(text, success = false) {
  $("notice").textContent = text;
  $("notice").className = success ? "notice success" : "notice";
  $("notice").hidden = !text;
}

function apiMessage(body, fallback) {
  if (typeof body?.detail === "string") return body.detail;
  if (Array.isArray(body?.detail)) return "La demande n’est pas valide. Vérifiez les fichiers ou la question.";
  return fallback;
}

async function request(path, options = {}) {
  let response;
  try { response = await fetch(path, { credentials: "same-origin", ...options }); }
  catch { throw new Error("La connexion au serveur a été interrompue. Actualisez les documents avant de relancer un import."); }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(apiMessage(body, response.status === 409 ? "Une opération est déjà en cours. Réessayez dans quelques instants." : "La demande n’a pas abouti. Veuillez réessayer."));
  }
  return response.status === 204 ? null : response.json();
}

function locked() { return state.localBusy || state.remoteBusy || !state.ready; }

function updateControls() {
  const disabled = locked();
  $("file-input").disabled = disabled;
  $("drop-zone").setAttribute("aria-disabled", String(disabled));
  $("select-all").disabled = disabled || !state.documents.length;
  $("select-all").checked = state.documents.length > 0 && state.selected.size === state.documents.length;
  $("select-all").indeterminate = state.selected.size > 0 && state.selected.size < state.documents.length;
  $("delete-documents").disabled = disabled || !state.selected.size;
  $("refresh-documents").disabled = disabled;
  $("question").disabled = disabled;
  $("send-question").disabled = disabled || !state.selected.size || !$("question").value.trim();
  document.querySelectorAll(".document-row input").forEach((checkbox) => { checkbox.disabled = disabled; });
  $("selection-summary").textContent = state.selected.size ? `${state.selected.size} document${state.selected.size > 1 ? "s" : ""} sélectionné${state.selected.size > 1 ? "s" : ""} pour votre prochaine question.` : "Sélectionnez un document pour commencer.";
  const count = state.documents.length;
  $("document-count").textContent = `${count} document${count > 1 ? "s" : ""} disponible${count > 1 ? "s" : ""} · ${state.selected.size} sélectionné${state.selected.size > 1 ? "s" : ""}`;
}

function renderDocuments() {
  const list = $("document-list");
  list.replaceChildren();
  const term = $("document-filter").value.trim().toLocaleLowerCase("fr");
  const filtered = state.documents.filter((item) => item.source.toLocaleLowerCase("fr").includes(term));
  if (!filtered.length) list.append(node("p", "document-empty", state.documents.length ? "Aucun document ne correspond à votre recherche." : "Votre espace est prêt à accueillir ses premiers documents."));
  for (const item of filtered) {
    const selected = state.selected.has(item.document_id);
    const row = node("label", `document-row${selected ? " selected" : ""}`);
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selected;
    checkbox.disabled = locked();
    checkbox.setAttribute("aria-label", `Sélectionner ${item.source}`);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.selected.add(item.document_id);
      else state.selected.delete(item.document_id);
      row.classList.toggle("selected", checkbox.checked);
      updateControls();
    });
    const description = node("span", "document-description");
    description.append(node("span", "document-name", item.source), node("span", "document-kind", item.file_type === "audio" ? "Échange audio" : item.file_type === "pdf" ? "Document PDF" : "Document"));
    row.append(checkbox, node("span", "file-symbol", item.file_type === "audio" ? "AUD" : item.file_type === "pdf" ? "PDF" : "DOC"), description);
    list.append(row);
  }
  updateControls();
}

async function loadDocuments() {
  const data = await request("/api/v1/documents");
  state.documents = data.documents;
  const available = new Set(state.documents.map((item) => item.document_id));
  state.selected = new Set([...state.selected].filter((id) => available.has(id)));
  renderDocuments();
}

async function loadStatus() {
  try {
    const data = await request("/api/v1/status");
    const wasBusy = state.remoteBusy;
    state.ready = Boolean(data.ready);
    state.remoteBusy = Boolean(data.busy);
    if (data.limits) Object.assign(state.limits, data.limits);
    $("connection-dot").className = `connection-dot ${state.ready ? (state.remoteBusy ? "busy" : "online") : "offline"}`;
    $("connection-label").textContent = !state.ready ? "Service en préparation" : state.remoteBusy ? "Traitement en cours" : "Service connecté";
    $("model-label").textContent = data.model ? `Modèle · ${data.model}` : "Moteur documentaire";
    $("upload-limits").textContent = `Jusqu’à ${state.limits.files} fichiers · ${state.limits.upload_mb} Mo par fichier`;
    $("retry-connection").hidden = state.ready;
    if (wasBusy && !state.remoteBusy && !state.localBusy) await loadDocuments().catch(() => {});
  } catch {
    state.ready = false;
    $("connection-dot").className = "connection-dot offline";
    $("connection-label").textContent = "Service indisponible";
    $("model-label").textContent = "Vérifiez que le serveur est démarré.";
    $("retry-connection").hidden = false;
  }
  updateControls();
}

async function connect() {
  await loadStatus();
  if (state.ready && !state.remoteBusy) {
    try { await loadDocuments(); }
    catch (error) { notice(error.message); }
  }
}

function uploadFile(file, position, total) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `/api/v1/documents?filename=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.responseType = "json";
    $("upload-label").textContent = `${position}/${total} · ${file.name}`;
    $("upload-bar").value = 0;
    $("upload-stage").textContent = "Transfert du fichier…";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) $("upload-bar").value = event.loaded / event.total * 100;
    };
    xhr.upload.onload = () => {
      $("upload-bar").removeAttribute("value");
      $("upload-stage").textContent = "Analyse et indexation du document…";
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
      else reject(new Error(apiMessage(xhr.response, "L’import a échoué. Vérifiez le fichier et réessayez.")));
    };
    xhr.onerror = () => reject(new Error("La connexion a été interrompue pendant l’import. Actualisez la liste avant de réessayer."));
    xhr.send(file);
  });
}

async function waitForImport(job) {
  if (!job?.job_id) throw new Error("Le serveur n’a pas confirmé l’import. Actualisez les documents avant de réessayer.");
  $("upload-bar").removeAttribute("value");
  $("upload-stage").textContent = "Transcription ou extraction, puis indexation… Vous pouvez laisser cette page ouverte.";
  for (;;) {
    const result = await request(`/api/v1/jobs/${encodeURIComponent(job.job_id)}`);
    if (result.status === "completed" && result.document) return result.document;
    if (result.status === "failed") throw new Error(result.error || "L’import a échoué.");
    if (result.status !== "processing") throw new Error("L’état de l’import n’est plus disponible. Actualisez les documents.");
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}

async function upload(files) {
  if (locked() || !files.length) return;
  if (files.length > state.limits.files) return notice(`Choisissez au maximum ${state.limits.files} fichiers par import.`);
  for (const file of files) {
    if (!allowedExtensions.has(file.name.split(".").pop().toLowerCase())) return notice(`Format non pris en charge : ${file.name}. Choisissez un PDF ou un fichier audio.`);
    if (file.size > state.limits.upload_mb * 1024 * 1024) return notice(`${file.name} dépasse la limite de ${state.limits.upload_mb} Mo.`);
    if (!file.size) return notice(`${file.name} est vide.`);
  }
  notice("");
  state.localBusy = true;
  updateControls();
  $("upload-progress").hidden = false;
  const imported = [];
  let failure = null;
  try {
    for (let index = 0; index < files.length; index++) {
      const result = await waitForImport(await uploadFile(files[index], index + 1, files.length));
      imported.push(result.document_id);
    }
  } catch (error) { failure = error.message; }
  finally {
    try {
      await loadDocuments();
      const available = new Set(state.documents.map((item) => item.document_id));
      imported.filter((id) => available.has(id)).forEach((id) => state.selected.add(id));
      renderDocuments();
    } catch (error) { failure ||= error.message; }
    state.localBusy = false;
    $("upload-progress").hidden = true;
    $("file-input").value = "";
    await loadStatus();
  }
  if (failure) notice(`${imported.length ? `${imported.length} document(s) importé(s). ` : ""}${failure}`);
  else notice(`${imported.length} fichier${imported.length > 1 ? "s" : ""} traité${imported.length > 1 ? "s" : ""}. Vous pouvez poser votre question.`, true);
}

function inlineText(parent, text) {
  text.split(/(\*\*[^*]+\*\*)/g).forEach((part) => {
    parent.append(part.startsWith("**") && part.endsWith("**") ? node("strong", "", part.slice(2, -2)) : document.createTextNode(part));
  });
}

function tableCells(line) {
  return line.trim().replace(/^\|/, "").replace(/(?<!\\)\|$/, "").split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, "|"));
}

function renderAnswer(parent, text) {
  const lines = text.split("\n");
  for (let index = 0; index < lines.length;) {
    const line = lines[index];
    if (!line.trim()) { index++; continue; }
    if (line.trim().startsWith("|") && lines[index + 1] && tableCells(lines[index + 1]).every((cell) => /^:?-{3,}:?$/.test(cell))) {
      const headers = tableCells(line);
      const rows = [];
      let end = index + 2;
      while (end < lines.length && lines[end].trim().startsWith("|")) rows.push(tableCells(lines[end++]));
      if (rows.every((row) => row.length === headers.length) && tableCells(lines[index + 1]).length === headers.length) {
        const table = node("table");
        const head = node("thead");
        const heading = node("tr");
        headers.forEach((cell) => { const th = node("th"); th.scope = "col"; inlineText(th, cell); heading.append(th); });
        head.append(heading);
        const body = node("tbody");
        rows.forEach((cells) => { const row = node("tr"); cells.forEach((cell) => { const td = node("td"); inlineText(td, cell); row.append(td); }); body.append(row); });
        table.append(head, body);
        const wrapper = node("div", "table-scroll"); wrapper.append(table); parent.append(wrapper);
      } else parent.append(node("pre", "", lines.slice(index, end).join("\n")));
      index = end;
    } else if (line.trim().startsWith("```")) {
      let end = index + 1;
      while (end < lines.length && !lines[end].trim().startsWith("```")) end++;
      parent.append(node("pre", "", lines.slice(index + 1, end).join("\n")));
      index = end + 1;
    } else {
      const heading = line.match(/^#{1,6}\s+(.+)/);
      const paragraph = node(heading ? "h3" : "p", /^\s*(?:[-*]|\d+\.)\s/.test(line) ? "list-line" : "");
      inlineText(paragraph, heading ? heading[1] : line);
      parent.append(paragraph);
      index++;
    }
  }
}

function sourceCaption(source, index) {
  const parts = [`${index + 1}. ${source.source || "Document"}`];
  if (source.page != null) parts.push(`page ${source.page}`);
  if (source.section) parts.push(source.section);
  if (source.start_time != null && source.end_time != null) parts.push(`${Number(source.start_time).toFixed(1)}–${Number(source.end_time).toFixed(1)} s`);
  return parts.join(" · ");
}

function showResponse(card, response) {
  card.replaceChildren();
  card.removeAttribute("aria-busy");
  const statuses = { answer: "Réponse proposée", insufficient: "Information non établie", unverified: "Information non établie", clarify: "Précision nécessaire", context_limit: "Périmètre à préciser", invalid_plan: "Réponse non établie", invalid_evidence: "Réponse non établie" };
  const header = node("div", "answer-heading");
  const label = node("div", "answer-label");
  label.append(node("span", "", "✦"), node("span", "", statuses[response.generation_status] || "Réponse"));
  header.append(label);
  if (response.omitted_evidence_count > 0) header.append(node("span", "partial-badge", "Contexte partiel"));
  const copy = node("button", "text-button copy-button", "Copier"); copy.type = "button";
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(response.answer); copy.textContent = "Copié"; }
    catch { copy.textContent = "Copie indisponible"; }
  });
  header.append(copy);
  const content = node("div", "answer-content");
  renderAnswer(content, response.answer || "Aucune réponse n’a été renvoyée.");
  card.append(header, content);
  const sources = response.retrieved_chunks || [];
  if (sources.length) {
    const wrapper = node("div", "sources");
    wrapper.append(node("p", "sources-title", `PASSAGES RETROUVÉS · ${sources.length}`));
    if (response.omitted_evidence_count > 0) wrapper.append(node("p", "sources-note", "Certains passages retrouvés n’ont pas été inclus dans le contexte de la réponse."));
    sources.forEach((source, index) => {
      const details = node("details", "source");
      details.append(node("summary", "", sourceCaption(source, index)));
      if (source.content_kind === "table") details.append(node("p", "source-note", source.table_complete === true ? "Bloc extrait entier · la couverture du PDF reste à vérifier." : "Bloc de tableau extrait · la couverture du PDF reste à vérifier."));
      details.append(node("pre", "source-content", source.content || ""));
      wrapper.append(details);
    });
    card.append(wrapper);
  }
}

function scrollConversation() {
  if (window.matchMedia("(max-width: 760px)").matches) $("messages").lastElementChild?.scrollIntoView({ block: "start" });
  else $("conversation-area").scrollTop = $("conversation-area").scrollHeight;
}

async function sendQuestion(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (locked() || !question || !state.selected.size) return;
  notice("");
  state.localBusy = true;
  const selected = [...state.selected];
  $("welcome").hidden = true;
  $("clear-conversation").disabled = false;
  const user = node("article", "message message-user");
  user.append(node("p", "", question), node("small", "", `${selected.length} document${selected.length > 1 ? "s" : ""} sélectionné${selected.length > 1 ? "s" : ""}`));
  const assistant = node("article", "message message-assistant");
  assistant.setAttribute("aria-busy", "true");
  const pending = node("div", "pending");
  pending.append(node("span", "spinner"), node("span", "", "Recherche et préparation de la réponse…"));
  assistant.append(pending);
  $("messages").append(user, assistant);
  $("question").value = "";
  $("question").style.height = "auto";
  updateControls();
  scrollConversation();
  try {
    const response = await request("/api/v1/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: question, top_k: 5, document_ids: selected }) });
    showResponse(assistant, response);
  } catch (error) {
    assistant.replaceChildren(node("p", "answer-error", error.message));
    assistant.removeAttribute("aria-busy");
    const retry = node("button", "text-button", "Reprendre la question"); retry.type = "button";
    retry.addEventListener("click", () => { $("question").value = question; updateControls(); $("question").focus(); });
    assistant.append(retry);
  } finally {
    state.localBusy = false;
    await loadStatus();
    scrollConversation();
    if (!locked()) $("question").focus();
  }
}

$("file-input").addEventListener("change", (event) => upload([...event.target.files]));
$("drop-zone").addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); if (!locked()) $("file-input").click(); } });
$("drop-zone").addEventListener("click", (event) => { if (locked()) event.preventDefault(); });
["dragenter", "dragover"].forEach((type) => $("drop-zone").addEventListener(type, (event) => { event.preventDefault(); if (!locked()) $("drop-zone").classList.add("drag-over"); }));
["dragleave", "drop"].forEach((type) => $("drop-zone").addEventListener(type, () => $("drop-zone").classList.remove("drag-over")));
$("drop-zone").addEventListener("drop", (event) => { event.preventDefault(); upload([...event.dataTransfer.files]); });
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
$("document-filter").addEventListener("input", renderDocuments);
$("select-all").addEventListener("change", (event) => { state.selected = event.target.checked ? new Set(state.documents.map((item) => item.document_id)) : new Set(); renderDocuments(); });
$("refresh-documents").addEventListener("click", async () => { try { await loadDocuments(); notice(""); } catch (error) { notice(error.message); } });
$("retry-connection").addEventListener("click", connect);
$("question-form").addEventListener("submit", sendQuestion);
$("question").addEventListener("input", () => { $("question").style.height = "auto"; $("question").style.height = `${Math.min($("question").scrollHeight, 180)}px`; updateControls(); });
$("question").addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!$("send-question").disabled) $("question-form").requestSubmit(); } });
document.querySelectorAll(".suggestion").forEach((button) => button.addEventListener("click", () => { $("question").value = button.dataset.question; updateControls(); $("question").focus(); }));
$("clear-conversation").addEventListener("click", () => { if (state.localBusy) return; $("messages").replaceChildren(); $("welcome").hidden = false; $("clear-conversation").disabled = true; notice(""); });
$("delete-documents").addEventListener("click", async () => {
  if (locked() || !state.selected.size) return;
  const ids = [...state.selected];
  if (!window.confirm(`Supprimer ${ids.length} document(s), leur index et leurs fichiers ? Cette action est définitive.`)) return;
  state.localBusy = true;
  updateControls();
  let failure = null;
  try { for (const id of ids) await request(`/api/v1/documents/${encodeURIComponent(id)}`, { method: "DELETE" }); }
  catch (error) { failure = error.message; }
  finally { try { await loadDocuments(); } catch (error) { failure ||= error.message; } state.localBusy = false; await loadStatus(); }
  notice(failure || "La liste des documents a été mise à jour.", !failure);
});

connect();
setInterval(() => { if (!document.hidden && !state.localBusy) loadStatus(); }, 15000);
