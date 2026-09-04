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
    state.results.push(body); render(body);
    $("form-message").textContent = "Analysis complete. Only the resulting scores were added to the audit trail.";
  } catch (error) { $("form-message").textContent = error.message; $("api-state").textContent = "ERROR"; }
  finally { setLoading(false); }
});

$("demo-button").addEventListener("click", () => { const demo = { combined_risk_score:.82, tier1:{score:.76,latency_ms:43}, tier2:{score:.85,confidence:.91,latency_ms:117,encoder_contributions:{wav2vec2_xlsr:.81,wavlm_large:.88,rawnet3:.86}}, consistency_check:{flag:"inconsistent",similarity_score:.41}, feature_breakdown:{prosody_irregularity:"high",spectral_artifacts:"high",breathing_pattern:"absent",background_noise_consistency:"inconsistent"}, alert:true }; state.results.push(demo); render(demo); $("result-title").textContent="Sample cloned-voice alert"; });
function setLoading(loading){$("analyze-button").disabled=loading||!fileInput.files[0];$("analyze-button").innerHTML=loading?"Analyzing…":"Analyze segment <span>→</span>";$("api-state").textContent=loading?"PROCESSING":"LIVE";}
function render(result){const risk=Math.round((result.running_risk_score ?? result.combined_risk_score)*100);$("risk-score").textContent=risk;$("gauge").style.color=risk>=70?"var(--danger)":"var(--teal)";$("risk-label").textContent=result.evidence_segments<3?`Collecting evidence (${result.evidence_segments}/3)`:risk>=70?"Elevated synthetic-voice risk":risk>=45?"Review recommended":"Low synthetic-voice risk";$("result-title").textContent=`Call risk · ${risk}%`;$("tier1-score").textContent=`${Math.round(result.tier1.score*100)}%`;$("tier1-latency").textContent=`${result.tier1.latency_ms} ms`;$("tier2-score").textContent=`${Math.round(result.tier2.score*100)}%`;$("tier2-confidence").textContent=`${Math.round(result.tier2.confidence*100)}% confidence`;const c=result.consistency_check;$("consistency").textContent=c.flag.replaceAll("_"," ");$("consistency-copy").textContent=c.similarity_score==null?"No reference available":`${Math.round(c.similarity_score*100)}% similarity to known voice`;$("alert-box").classList.toggle("hidden",!result.alert);$("api-state").textContent="LIVE";renderChart();renderFeatures(result.feature_breakdown);renderBars(result.tier2.encoder_contributions);}
function renderChart(){const chart=$("chart");chart.innerHTML="";state.results.slice(-12).forEach(r=>{const b=document.createElement("div");b.className=`chart-bar ${r.combined_risk_score>=.7?"high":""}`;b.style.height=`${Math.max(8,r.combined_risk_score*88)}px`;b.title=`${Math.round(r.combined_risk_score*100)}% risk`;chart.append(b)});$("segment-count").textContent=`${state.results.length} analyzed`;}
function renderFeatures(features){const box=$("features");box.innerHTML="";$("feature-status").textContent=features?"DISPLAY ONLY":"PRIVATE MODE";if(!features){box.innerHTML='<p class="muted">Feature detail omitted by feature-only audit mode.</p>';return}Object.entries(features).forEach(([name,value])=>{const item=document.createElement("div");item.className="feature";item.innerHTML=`<span>${name.replaceAll("_"," ")}</span><strong>${value}</strong>`;box.append(item)});}
function renderBars(values){const box=$("contribution-bars");box.innerHTML="";Object.entries(values).forEach(([name,value])=>{const item=document.createElement("div");item.className="bar-item";item.innerHTML=`${name.replaceAll("_"," ")}<div class="bar-track"><div class="bar-fill" style="width:${value*100}%"></div></div>`;box.append(item)});}
