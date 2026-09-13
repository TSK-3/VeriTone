const $ = (id) => document.getElementById(id);
const state = { results: [] };
const fileInput = $("audio-file");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  $("file-name").textContent = file ? file.name : "Choose a WAV segment";
  $("analyze-button").disabled = !file;
  $("form-message").textContent = file ? `${(file.size / 1024).toFixed(1)} KB ready for in-memory analysis.` : "Choose an audio segment to begin.";
});

$("analyze-button").addEventListener("click", async () => {
  const file = fileInput.files[0]; if (!file) return;
  const callId = $("call-id").value.trim() || "unnamed-call";
  const query = new URLSearchParams({ start_s: $("start-time").value || "0", feature_only_logging: $("feature-only").checked });
  const similarity = $("similarity").value; if (similarity) query.set("speaker_similarity", similarity);
  setLoading(true);
  try {
    const response = await fetch(`/v1/calls/${encodeURIComponent(callId)}/segments?${query}`, { method: "POST", headers: { "Content-Type": "audio/wav" }, body: file });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Analysis request failed");
    const result = {...body.segment, running_risk_score: body.risk_score / 100, evidence_segments: body.evidence_segments, alert: !["pass", "warn"].includes(body.action), recommended_action: body.recommended_action};
    state.results.push(result); render(result);
    $("form-message").textContent = `✅ Analysis complete — risk ${body.risk_score}/100 (${body.action}). Only the scores entered the audit trail.`;
    $("result-title").textContent = `Clip risk · ${body.risk_score}%`;
  } catch (error) { $("form-message").textContent = error.message; $("api-state").textContent = "ERROR"; }
  finally { setLoading(false); }
});

// --- live call panel: polls /v1/live every second -----------------------------
const liveState = { sessionId: null, alerts: 0, transcriptShown: 0, pollTimer: null };

function highlightTriggers(text, triggers) {
  let html = text;
  (triggers || []).forEach((w) => { html = html.replace(new RegExp(`(${w})`, "ig"), '<mark>$1</mark>'); });
  return html;
}

function renderBarsLive(contributions) {
  const box = $("live-bars"); box.innerHTML = "";
  Object.entries(contributions || {}).forEach(([name, value]) => {
    const item = document.createElement("div"); item.className = "bar-item";
    item.innerHTML = `${name.replaceAll("_", " ")}<div class="bar-track"><div class="bar-fill" style="width:${Math.round(value * 100)}%"></div></div>`;
    box.append(item);
  });
}

function renderTranscript(transcript) {
  const box = $("transcript"); if (!transcript) return;
  box.innerHTML = "";
  transcript.forEach((t) => {
    const line = document.createElement("p"); line.className = `transcript-line src-${t.source}`;
    line.innerHTML = `<strong>${t.source === "spoof" || t.source === "simulated" ? "caller" : t.source}</strong> ${highlightTriggers(t.text, t.triggers)}${t.risk != null ? `<span class="line-risk">risk ${t.risk}</span>` : ""}`;
    box.append(line);
  });
  box.scrollTop = box.scrollHeight;
}

function renderSms(alerts, configured) {
  $("sms-mode").textContent = configured ? "Twilio SMS live" : "console mode";
  const box = $("sms-list"); if (!alerts || !alerts.length) return;
  box.innerHTML = "";
  alerts.slice(-4).forEach((a) => {
    const card = document.createElement("div"); card.className = `sms-card ${a.kind}`;
    card.innerHTML = `<strong>✉ ${a.kind} · risk ${a.risk}${a.words && a.words.length ? ` · “${a.words.join("`, `")}”` : ""}</strong><small>${a.body.replaceAll("\n", " · ")}</small><span class="sms-status ${a.sms_status}">${a.sms_status}</span>`;
    box.append(card);
  });
}

async function pollLive() {
  try {
    const res = await fetch("/v1/live"); if (!res.ok) return;
    const data = await res.json();
    const active = (data.calls || []).find((c) => c.status === "live") || (data.calls || [])[0];
    if (!active) return;
    liveState.sessionId = active.session_id;
    $("live-label").textContent = active.label || active.session_id;
    $("live-state").textContent = active.status === "live" ? "LIVE" : "ENDED";
    $("live-dot").classList.toggle("on", active.status === "live");
    const risk = active.risk_score || 0;
    $("live-risk").textContent = risk;
    $("live-gauge").style.color = risk >= 70 ? "var(--danger)" : risk >= 45 ? "var(--amber)" : "var(--teal)";
    $("live-action").textContent = active.recommended_action || active.action || "";
    $("live-evidence").textContent = `${active.evidence_segments || 0} segments analyzed · Tier1 ${active.signals?.tier1 ?? "—"}% / Tier2 ${active.signals?.tier2 ?? "—"}%`;
    const triggers = await fetch(`/v1/live/${encodeURIComponent(active.session_id)}`).then((r) => r.ok ? r.json() : null);
    if (!triggers) return;
    $("live-triggers").textContent = triggers.triggers?.length ? `Trigger words: ${[...new Set(triggers.triggers)].join(", ")}` : "No trigger words heard";
    $("live-peak").textContent = `Peak risk: ${triggers.peak_risk ?? risk}`;
    renderBarsLive(active.signals?.contributions);
    renderTranscript(triggers.transcript);
    renderSms(triggers.alerts, data.sms_configured);
  } catch { /* server restarting during demo — keep last frame */ }
}

$("demo-scenario-button").addEventListener("click", async () => {
  $("demo-scenario-button").disabled = true;
  $("demo-scenario-button").innerHTML = "⏳ Scam call in progress…";
  try {
    const res = await fetch("/v1/demo/scenario", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: `scam-demo-${Date.now()}` }) });
    const body = await res.json();
    liveState.sessionId = body.session_id;
    $("result-title").textContent = "Live simulated call running";
  } catch (e) { $("demo-note").textContent = `Failed to start: ${e.message}`; }
  setTimeout(() => { $("demo-scenario-button").disabled = false; $("demo-scenario-button").innerHTML = "▶ Run simulated scam call"; }, 16000);
});

$("demo-sms-button").addEventListener("click", async () => {
  try {
    const res = await fetch("/v1/demo/sms", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: liveState.sessionId }) });
    const body = await res.json();
    $("demo-note").textContent = `Test SMS: ${body.sms_status}${body.error ? ` — ${body.error}` : ""}`;
    pollLive();
  } catch (e) { $("demo-note").textContent = `SMS test failed: ${e.message}`; }
});

$("demo-scamcall-button").addEventListener("click", async () => {
  $("demo-scamcall-button").disabled = true;
  $("demo-scamcall-button").innerHTML = "⏳ Scam in progress…";
  try {
    // 1) Guaranteed live analysis: real cloned-voice WAVs through the full pipeline.
    const sid = `scam-play-${Date.now()}`;
    await fetch("/v1/demo/scenario", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sid }) });
    // 2) Real Twilio call in parallel: cloned voice dials the customer's phone.
    let callInfo = "";
    try {
      const res = await fetch("/v1/twilio/scam-call", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
      const body = await res.json();
      callInfo = ` 📞 Twilio call placed to ${body.to} — answer & press any key.`;
    } catch (e) { callInfo = ` 📞 Twilio call failed: ${e.message}`; }
    $("demo-note").textContent = `Live cloned-voice analysis running.${callInfo}`;
  } catch (e) { $("demo-note").textContent = `Failed: ${e.message}`; }
  setTimeout(() => { $("demo-scamcall-button").disabled = false; $("demo-scamcall-button").innerHTML = "📞 Call me — cloned voice"; }, 16000);
});

pollLive();
liveState.pollTimer = setInterval(pollLive, 1000);
function setLoading(loading){$("analyze-button").disabled=loading||!fileInput.files[0];$("analyze-button").innerHTML=loading?"Analyzing…":"Analyze segment <span>→</span>";$("api-state").textContent=loading?"PROCESSING":"LIVE";}
function render(result){const risk=Math.round((result.running_risk_score ?? result.combined_risk_score)*100);$("risk-score").textContent=risk;$("gauge").style.color=risk>=70?"var(--danger)":"var(--teal)";$("risk-label").textContent=result.evidence_segments<3?`Collecting evidence (${result.evidence_segments}/3)`:risk>=70?"Elevated synthetic-voice risk":risk>=45?"Review recommended":"Low synthetic-voice risk";$("result-title").textContent=`Call risk · ${risk}%`;$("tier1-score").textContent=`${Math.round(result.tier1.score*100)}%`;$("tier1-latency").textContent=`${result.tier1.latency_ms} ms`;$("tier2-score").textContent=`${Math.round(result.tier2.score*100)}%`;$("tier2-confidence").textContent=`${Math.round(result.tier2.confidence*100)}% confidence`;const c=result.consistency_check;$("consistency").textContent=c.flag.replaceAll("_"," ");$("consistency-copy").textContent=c.similarity_score==null?"No reference available":`${Math.round(c.similarity_score*100)}% similarity to known voice`;$("alert-copy").textContent=result.recommended_action||"Risk crossed your configured operating threshold.";$("alert-box").classList.toggle("hidden",!result.alert);$("api-state").textContent="LIVE";renderChart();renderFeatures(result.feature_breakdown);renderBars(result.tier2.encoder_contributions);}
function renderChart(){const chart=$("chart");chart.innerHTML="";state.results.slice(-12).forEach(r=>{const b=document.createElement("div");b.className=`chart-bar ${r.combined_risk_score>=.7?"high":""}`;b.style.height=`${Math.max(8,r.combined_risk_score*88)}px`;b.title=`${Math.round(r.combined_risk_score*100)}% risk`;chart.append(b)});$("segment-count").textContent=`${state.results.length} analyzed`;}
function renderFeatures(features){const box=$("features");box.innerHTML="";$("feature-status").textContent=features?"DISPLAY ONLY":"PRIVATE MODE";if(!features){box.innerHTML='<p class="muted">Feature detail omitted by feature-only audit mode.</p>';return}Object.entries(features).forEach(([name,value])=>{const item=document.createElement("div");item.className="feature";item.innerHTML=`<span>${name.replaceAll("_"," ")}</span><strong>${value}</strong>`;box.append(item)});}
function renderBars(values){const box=$("contribution-bars");box.innerHTML="";Object.entries(values).forEach(([name,value])=>{const item=document.createElement("div");item.className="bar-item";item.innerHTML=`${name.replaceAll("_"," ")}<div class="bar-track"><div class="bar-fill" style="width:${value*100}%"></div></div>`;box.append(item)});}
