const kbUpload = document.getElementById("kbUpload");
const rebuildBtn = document.getElementById("rebuildBtn");
const clearBtn = document.getElementById("clearBtn");
const checkBtn = document.getElementById("checkBtn");
const inputText = document.getElementById("inputText");
const highlighted = document.getElementById("highlighted");
const claims = document.getElementById("claims");
const hint = document.getElementById("hint");
const downloadBtn = document.getElementById("downloadBtn");
const copyPlainBtn = document.getElementById("copyPlainBtn");
const openaiToggle = document.getElementById("openaiToggle");

let lastResult = null;

function setHint(message, tone = "") {
  hint.textContent = message;
  hint.className = tone ? `hint ${tone}` : "hint";
}

async function fetchStatus() {
  const resp = await fetch("/api/kb/status");
  if (!resp.ok) return;
  const data = await resp.json();
  document.getElementById("kbCount").textContent = `${data.file_count} files`;
  document.getElementById("kbChunks").textContent = data.chunk_count;
  document.getElementById("kbModel").textContent = data.embedding_model || "—";
  document.getElementById("kbIndexed").textContent = data.last_indexed || "—";
}

function renderHighlights(text, spans) {
  if (!spans || spans.length === 0) {
    highlighted.textContent = text;
    return;
  }
  const sorted = spans.slice().sort((a, b) => a.start - b.start);
  let cursor = 0;
  const parts = [];
  for (const span of sorted) {
    parts.push(escapeHtml(text.slice(cursor, span.start)));
    const labelClass = span.label === "SUPPORTED" ? "supported" : span.label === "CONTRADICTED" ? "contradicted" : "nei";
    parts.push(
      `<mark class="${labelClass}">${escapeHtml(text.slice(span.start, span.end))}</mark>`
    );
    cursor = span.end;
  }
  parts.push(escapeHtml(text.slice(cursor)));
  highlighted.innerHTML = parts.join("");
}

function renderClaims(spans) {
  claims.innerHTML = "";
  spans.forEach((span, idx) => {
    const card = document.createElement("div");
    card.className = "claim-card";
    const label = span.label;
    const conf = span.confidence.toFixed(2);
    card.innerHTML = `
      <h3>Claim ${idx + 1}: ${escapeHtml(span.claim)}</h3>
      <div class="claim-meta">
        <span>Label: ${label}</span>
        <span>Confidence: ${conf}</span>
        <span>Evidence: ${span.evidence.length}</span>
      </div>
    `;
    span.evidence.forEach((ev) => {
      const evDiv = document.createElement("div");
      evDiv.className = "evidence";
      evDiv.innerHTML = `
        <small>${escapeHtml(ev.source_file)} · ${escapeHtml(ev.chunk_id)} · score ${ev.score.toFixed(2)}</small>
        <div>${escapeHtml(ev.text)}</div>
      `;
      card.appendChild(evDiv);
    });
    claims.appendChild(card);
  });
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

kbUpload.addEventListener("change", async (event) => {
  const files = event.target.files;
  if (!files.length) return;
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));
  const resp = await fetch("/api/kb/upload", { method: "POST", body: formData });
  if (!resp.ok) {
    setHint("Upload failed.");
    return;
  }
  const data = await resp.json();
  setHint(`Uploaded ${data.count} files. Rebuild index to apply.`);
  await fetchStatus();
});

rebuildBtn.addEventListener("click", async () => {
  setHint("Rebuilding index...");
  const resp = await fetch("/api/kb/rebuild", { method: "POST" });
  if (!resp.ok) {
    setHint("Index rebuild failed.");
    return;
  }
  await fetchStatus();
  setHint("Index rebuilt.");
});

clearBtn.addEventListener("click", async () => {
  const resp = await fetch("/api/kb/clear", { method: "DELETE" });
  if (!resp.ok) {
    setHint("Clear failed.");
    return;
  }
  highlighted.textContent = "";
  claims.innerHTML = "";
  await fetchStatus();
  setHint("KB cleared.");
});

checkBtn.addEventListener("click", async () => {
  const text = inputText.value.trim();
  if (!text) {
    setHint("Please paste some text.");
    return;
  }
  setHint("Checking...");
  const mode = openaiToggle.checked ? "openai" : "local";
  const resp = await fetch("/api/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, top_k: 5, mode, return_debug: false })
  });
  if (!resp.ok) {
    const err = await resp.json();
    setHint(err.detail || "Check failed.");
    return;
  }
  const data = await resp.json();
  lastResult = data;
  renderHighlights(data.input_text, data.spans);
  renderClaims(data.spans);
  setHint("Done.");
});

downloadBtn.addEventListener("click", () => {
  if (!lastResult) {
    setHint("No results to download.");
    return;
  }
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "fact_check_results.json";
  a.click();
  URL.revokeObjectURL(url);
});

copyPlainBtn.addEventListener("click", async () => {
  const text = inputText.value;
  await navigator.clipboard.writeText(text);
  setHint("Copied plain text.");
});

fetchStatus();
