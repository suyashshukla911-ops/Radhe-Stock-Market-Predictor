const $=id=>document.getElementById(id);
const state={symbol:"SENSEX",interval:"5",lastAnalysis:null,lastAnalysisAt:0,lastInstrumentKey:null,busy:false,predictionCadenceMs:300000,marketOpen:false};

function fmt(v,d=2){return Number.isFinite(Number(v))?Number(v).toLocaleString("en-IN",{minimumFractionDigits:d,maximumFractionDigits:d}):"—"}
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
async function getJSON(url){const r=await fetch(url,{cache:"no-store"});const j=await r.json();if(!r.ok)throw new Error(j.detail||"Request failed");return j}

function candleLabel(start,end,sec,open){
  if(!start||!end)return "—";
  const s=new Date(start).toLocaleTimeString("en-IN",{hour:"2-digit",minute:"2-digit"});
  const e=new Date(end).toLocaleTimeString("en-IN",{hour:"2-digit",minute:"2-digit"});
  return open?`Current candle ${s}–${e}${sec!=null?` · ${sec}s left`:""}`:`Last completed candle ${s}–${e}`;
}
function drawChart(rows,pred){
  if(!window.Plotly)return;
  if(!rows?.length){Plotly.purge("chart");return}
  const trace={x:rows.map(r=>r.time),open:rows.map(r=>r.open),high:rows.map(r=>r.high),low:rows.map(r=>r.low),close:rows.map(r=>r.close),type:"candlestick",name:"Market",increasing:{line:{color:"#15966b"}},decreasing:{line:{color:"#d64550"}}};
  const last=rows[rows.length-1].time;
  const layout={paper_bgcolor:"#ffffff",plot_bgcolor:"#ffffff",margin:{l:54,r:18,t:10,b:38},dragmode:"pan",uirevision:`${state.symbol}-${state.interval}`,xaxis:{rangeslider:{visible:false},gridcolor:"#e9edf2",color:"#5b6b7d",fixedrange:false},yaxis:{gridcolor:"#e9edf2",color:"#5b6b7d",fixedrange:false},font:{color:"#17324d"},showlegend:false};
  const shapes=[];
  if(pred){
    [["predicted_close","#2563eb"],["predicted_high","#15966b"],["predicted_low","#d64550"]].forEach(([k,c])=>{
      const y=Number(pred[k]); if(Number.isFinite(y))shapes.push({type:"line",x0:last,x1:last,y0:y,y1:y,line:{color:c,dash:"dot",width:1}});
    });
  }
  layout.shapes=shapes;
  Plotly.react("chart",[trace],layout,{displaylogo:false,responsive:true,displayModeBar:false,doubleClick:"reset",scrollZoom:true});
}

function clearPrediction(){
  state.lastAnalysis=null; state.lastAnalysisAt=0;
  ["current","next-close","next-high","next-low","eod-close","signal","up-p","down-p","model-status","model-age","probability","title","timestamp","candle-label","candle-clock"].forEach(id=>{if($(id))$(id).textContent="—"});
  $("validation").innerHTML="";
}
function renderValidation(v){
  const items=[
    ["Direction accuracy",`${v.direction_accuracy}%`],
    ["Selective precision",`${v.signal_precision}%`],
    ["Wilson lower bound",`${v.precision_lower_bound}%`],
    ["Signal count",v.signal_count],
    ["Coverage",`${v.signal_coverage}%`],
    ["Brier",v.calibrated_brier],
    ["Threshold",v.threshold],
    ["Training rows",v.train_rows]
  ];
  $("validation").innerHTML=items.map(([a,b])=>`<div class="val"><span>${esc(a)}</span><b>${esc(b)}</b></div>`).join("");
}
function renderNews(n){
  const rows=(n?.display_articles||[]).slice(0,10);
  const toi=n?.toi||{};
  $("news-score").textContent=`${rows.length} top stories`;
  $("news-score").className="pill good";
  $("news-summary").innerHTML=[["Top stories",rows.length],["Source","Times of India"],["Updated",toi.fetched_at?new Date(toi.fetched_at).toLocaleTimeString("en-IN")||"—":"—"],["Coverage",`${toi.sections_ok||0}/${toi.sections_total||0} sections`]].map(([a,b])=>`<div><b>${esc(b)}</b><span>${esc(a)}</span></div>`).join("");
  $("news-list").innerHTML=rows.map((a,i)=>{
    const when=a.scraped_at||"";
    return `<article class="news-item"><div class="news-rank">${i+1}</div><div><a href="${esc(a.url||"#")}" target="_blank" rel="noopener">${esc(a.title||"")}</a><div class="news-meta"><span>${esc(a.section||"Markets")}</span><span>${when?new Date(when).toLocaleString("en-IN"):"Latest"}</span></div></div></article>`;
  }).join("")||`<div class="muted">Top Times of India headlines are loading. Please check again shortly.</div>`;
  $("news-quota").textContent=toi.fetched_at?`Latest Times of India update: ${new Date(toi.fetched_at).toLocaleString("en-IN")}`:"Waiting for the latest Times of India headlines…";
  $("news-download").href=n?.download_url||"/api/news/download";
}
function renderAnalysis(d){
  state.lastAnalysis=d;
  const p=d.prediction,v=p.validation;
  $("title").textContent=`${d.instrument.name} · ${d.interval}m`;
  $("timestamp").textContent=`Model input: ${new Date(d.last_timestamp).toLocaleString("en-IN")}`;
  $("current").textContent=fmt(d.current_price);
  $("price-label").textContent=d.market_state.market_status==="OPEN"?"LIVE PRICE":"LAST MARKET PRICE";
  $("next-close").textContent=fmt(p.predicted_close);
  $("next-high").textContent=fmt(p.predicted_high);
  $("next-low").textContent=fmt(p.predicted_low);
  $("eod-close").textContent=fmt(p.predicted_eod_close);
  $("signal").textContent=p.signal;
  $("signal-note").textContent=p.signal==="REVALIDATE"?"Validation gate not statistically sufficient":"Signal eligible";
  $("gate-title").textContent=v.gate_90_passed?"SELECTIVE SIGNAL GATE PASSED":"REVALIDATE";
  $("gate-copy").textContent=v.gate_90_passed?`Selective validation supports ${v.signal_precision}% precision with a ${v.precision_lower_bound}% Wilson lower bound.`:`Prediction remains visible, but the measured selective precision (${v.signal_precision}%) and Wilson lower bound (${v.precision_lower_bound}%) do not support a trusted BUY/SELL.`;
  const up=p.probability_up;
  $("probability").textContent=`UP ${(up*100).toFixed(1)}% · DOWN ${((1-up)*100).toFixed(1)}%`;
  $("probability").className="pill "+(up>=.65?"good":up<=.35?"bad":"warn");
  $("up-p").textContent=(up*100).toFixed(1)+"%";
  $("down-p").textContent=((1-up)*100).toFixed(1)+"% down";
  $("model-status").textContent=v.gate_90_passed?"SIGNAL ELIGIBLE":"REVALIDATE";
  $("model-age").textContent=`Model age: ${Math.round(d.market_state.model_cache_age_seconds)}s · retrain ${d.market_state.model_retrain_minutes}m`;
  $("policy-box").textContent=`Refresh policy: UI ${d.refresh_policy.browser_seconds}s · prediction ${d.refresh_policy.prediction_seconds}s · news ${d.refresh_policy.news_minutes}m · TOI ${d.refresh_policy.toi_minutes}m.`;
  renderValidation(v); renderNews(d.news); drawChart(d.candles,p); state.lastAnalysisAt=Date.now();
}
function renderLive(d){
  const status=d.market_data?.market_status||"UNKNOWN";
  state.marketOpen=status==="OPEN";
  const live=Boolean(d.tick&&Number.isFinite(Number(d.tick.price)));
  const lastCandle=d.current_candle;

  if(live)$("current").textContent=fmt(d.tick.price);
  else if(d.market_data?.last_market_price!=null)$("current").textContent=fmt(d.market_data.last_market_price);

  $("price-label").textContent=status==="OPEN"?"LIVE PRICE":state.interval==="1d"?"PREVIOUS SESSION CLOSE":"LAST CANDLE CLOSE";
  $("status").textContent=status==="OPEN"?(d.market_data.stream_connected?"● MARKET OPEN · LIVE DATA":"○ MARKET OPEN · RECOVERING LIVE DATA"):`● MARKET ${status} · LAST COMPLETED SESSION`;
  $("status").className=status==="OPEN"?"status live":"status closed";

  const age=d.market_data.last_update_age_seconds;
  $("feed-age").textContent=age==null?"—":`${age.toFixed(1)}s`;

  if(lastCandle){
    if(state.interval==="1d"){
      const day=new Date(lastCandle.start).toLocaleDateString("en-IN",{day:"2-digit",month:"short",year:"numeric"});
      $("candle-clock").textContent=`Previous session · ${day}`;
      $("candle-label").textContent=`Previous completed session · ${day}`;
      $("title").textContent=`${d.symbol} · 1D`;
    }else{
      $("candle-clock").textContent=candleLabel(lastCandle.start,lastCandle.end,d.seconds_to_candle_close,lastCandle.forming);
      $("candle-label").textContent=candleLabel(lastCandle.start,lastCandle.end,d.seconds_to_candle_close,lastCandle.forming);
      $("title").textContent=`${d.symbol} · ${state.interval}m`;
    }
  }

  if(d.tick?.trade_time)$("timestamp").textContent=status==="OPEN"?`Live ${new Date(d.tick.trade_time).toLocaleTimeString("en-IN")}`:`Last market trade ${new Date(d.tick.trade_time).toLocaleString("en-IN")}`;

  if(d.candles?.length)drawChart(d.candles, state.marketOpen && state.lastAnalysis ? state.lastAnalysis.prediction : null);

  if(status!=="OPEN"){
    // Never carry live prediction numbers into after-hours mode.
    state.lastAnalysis=null; state.lastAnalysisAt=0;
    if(state.interval==="1d"){
      $("price-label").textContent="PREVIOUS SESSION CLOSE";
      $("current").textContent=d.market_data?.last_market_price!=null?fmt(d.market_data.last_market_price):"—";
      $("timestamp").textContent=lastCandle?.start?`Previous session ${new Date(lastCandle.start).toLocaleDateString("en-IN")}`:"Previous session";
      $("next-close").textContent="—"; $("next-high").textContent="—"; $("next-low").textContent="—"; $("eod-close").textContent="—";
      $("signal").textContent="HISTORY"; $("signal-note").textContent="Previous completed session";
      $("model-status").textContent="DAILY HISTORY"; $("model-age").textContent="Select an intraday timeframe during market hours for prediction";
      $("up-p").textContent="—"; $("down-p").textContent="—"; $("probability").textContent="HISTORY";
      $("policy-box").textContent="Daily history view. Select 1m–60m while the market is open for a model forecast.";
      $("validation").innerHTML="";
    }else{
      $("price-label").textContent="LAST CANDLE CLOSE";
      if(lastCandle){
        $("current").textContent=fmt(lastCandle.close);
        $("next-close").textContent=fmt(lastCandle.close);
        $("next-high").textContent=fmt(lastCandle.high);
        $("next-low").textContent=fmt(lastCandle.low);
        $("eod-close").textContent=fmt(lastCandle.close);
        $("next-close-label").textContent="last completed candle close";
        $("next-high-label-title").textContent="LAST CANDLE HIGH";
        $("next-low-label-title").textContent="LAST CANDLE LOW";
        $("next-close-label-title").textContent="LAST CANDLE CLOSE";
        $("eod-label-title").textContent="LAST SESSION CLOSE";
      }
      $("signal").textContent="HISTORY"; $("signal-note").textContent="Last completed session";
      $("up-p").textContent="—"; $("down-p").textContent="—"; $("probability").textContent="MARKET CLOSED";
      $("model-status").textContent="MARKET CLOSED"; $("model-age").textContent="Prediction available during market hours";
      $("policy-box").textContent="Live price/candles remain visible. Next-candle predictions are enabled only while the market is open.";
      $("validation").innerHTML="";
    }
  }else{
    $("next-close-label-title").textContent="NEXT CANDLE CLOSE";
    $("next-high-label-title").textContent="NEXT CANDLE HIGH";
    $("next-low-label-title").textContent="NEXT CANDLE LOW";
    $("eod-label-title").textContent="END-OF-DAY OUTLOOK";
    $("next-close-label").textContent="model estimate for selected timeframe";
  }
}
async function initialLive(){
  const d=await getJSON(`/api/live?symbol=${encodeURIComponent(state.symbol)}&interval=${state.interval}`);
  state.lastInstrumentKey=d.instrument_key; renderLive(d);
}
async function analyze(force=false){
  if(state.busy)return;
  if(state.interval==="1d"){await initialLive();return;}
  if(!state.marketOpen){
    $("status").textContent="● MARKET CLOSED · predictions resume at 09:15 IST";
    $("status").className="status closed";
    return;
  }
  state.busy=true;
  $("analyze").disabled=true; $("retrain").disabled=true;
  $("status").textContent=force?`RETRAINING ${state.symbol}…`:`ANALYZING ${state.symbol} ${state.interval}m…`;
  try{
    const d=await getJSON(`/api/analyze?symbol=${encodeURIComponent(state.symbol)}&interval=${encodeURIComponent(state.interval)}&retrain=${force}`);
    state.lastInstrumentKey=d.instrument.instrument_key;
    renderAnalysis(d);
    $("status").textContent=d.market_state.market_status==="OPEN"?(d.market_state.stream_connected?"● LIVE · UPSTOX STREAM":"○ LIVE · REST RECOVERY"):"● MARKET CLOSED · LAST SESSION";
    $("status").className=d.market_state.market_status==="OPEN"?"status live":"status closed";
  }catch(e){ $("status").textContent=`MODEL: ${e.message}`; $("status").className="status error"; }
  finally{state.busy=false;$("analyze").disabled=false;$("retrain").disabled=false}
}
async function pulse(){
  try{
    const d=await getJSON("/api/market-pulse?limit=30");
    $("pulse-grid").innerHTML=(d.results||[]).slice(0,20).map(x=>`<button class="pulse-item" data-symbol="${esc(x.symbol)}"><b>${esc(x.symbol)}</b><span>${esc(x.name)}</span><strong class="${Number(x.change_pct)>=0?"up":"down"}">${x.change_pct==null?"—":(Number(x.change_pct)>=0?"+":"")+Number(x.change_pct).toFixed(2)+"%"}</strong><small>${x.price==null?"—":fmt(x.price)}</small></button>`).join("");
    document.querySelectorAll(".pulse-item").forEach(b=>b.addEventListener("click",async()=>{state.symbol=b.dataset.symbol;$("symbol").value=state.symbol;clearPrediction();$("status").textContent=`LOADING ${state.symbol}…`;window.scrollTo({top:0,behavior:"smooth"});await initialLive().catch(()=>{});if(state.interval!=="1d" && state.marketOpen)await analyze(false)}));
  }catch(e){$("pulse-grid").innerHTML=`<div class="muted">${esc(e.message)}</div>`}
}
async function newsRefresh(){
  try{const d=await getJSON(`/api/news?symbol=${encodeURIComponent(state.symbol)}&name=${encodeURIComponent(state.symbol)}`);renderNews(d)}catch(e){$("news-quota").textContent=`Market news unavailable: ${e.message}`}
}
async function manualNewsRefresh(){
  const btn=$("news-refresh");
  if(!btn)return;
  btn.disabled=true; btn.textContent="REFRESHING…";
  try{
    const r=await getJSON("/api/news/refresh");
    await newsRefresh();
    $("news-quota").textContent=`Latest Times of India update: ${r.fetched_at?new Date(r.fetched_at).toLocaleString("en-IN"):"just now"}`;
  }catch(e){$("news-quota").textContent=`News refresh failed: ${e.message}`}
  finally{btn.disabled=false;btn.textContent="REFRESH NEWS"}
}
async function loadUniverse(query=""){
  const button=$("universe-search-btn"), status=$("search-status");
  try{
    button.disabled=true; button.textContent="SEARCHING…"; status.textContent=query?`Searching for “${query}”…`:"Loading listed securities…";
    const url=query?`/api/stocks?q=${encodeURIComponent(query)}&limit=50`:`/api/universe`;
    const d=await getJSON(url);
    if(!query){$("universe-badge").textContent=`DIRECTORY ${(d.total||0).toLocaleString("en-IN")}`;renderUniverse(d.results||[]);status.textContent=`Showing ${(d.results||[]).length.toLocaleString("en-IN")} of ${(d.total||0).toLocaleString("en-IN")} listed securities.`;}
    else {renderUniverse(d.results||[]);status.textContent=`${(d.results||[]).length} result(s) found for “${query}”. Click a row to load it.`;}
  }catch(e){$("universe-results").innerHTML=`<div class="muted">${esc(e.message)}</div>`;status.textContent=`Search failed: ${e.message}`}
  finally{button.disabled=false;button.textContent="SEARCH"}
}
function renderUniverse(rows){
  $("universe-results").innerHTML=(rows||[]).slice(0,60).map(x=>`<button class="stock-row" data-symbol="${esc(x.symbol)}"><b>${esc(x.symbol)}</b><span>${esc(x.name||"")}</span><small>${esc(x.exchange||"")}</small></button>`).join("")||`<div class="muted">No securities matched.</div>`;
  document.querySelectorAll(".stock-row").forEach(b=>b.addEventListener("click",async()=>{state.symbol=b.dataset.symbol;$("symbol").value=state.symbol;$("search-status").textContent=`Selected ${state.symbol}. Loading market data…`;clearPrediction();window.scrollTo({top:0,behavior:"smooth"});await initialLive().catch(()=>{});if(state.interval!=="1d" && state.marketOpen)await analyze(false)}));
}
async function boot(){
  try{
    const h=await getJSON("/api/health");
    state.predictionCadenceMs=Math.max(120000,Number(h.prediction_refresh_seconds||300)*1000); state.interval=$("interval").value;
    $("status").textContent=h.upstox_connected?"● UPSTOX READY":"○ UPSTOX TOKEN MISSING";
    const p=await getJSON("/api/priority");
    $("priority-list").innerHTML=p.stocks.map(s=>`<option value="${esc(s.symbol)}">${esc(s.name)}</option>`).join("");
    await initialLive();
    if(state.marketOpen) await analyze(false);
    await newsRefresh();
    await pulse();
    await loadUniverse();
  }catch(e){
    $("status").textContent=e.message;$("status").className="status error";
  }
}

async function refreshAll(){
  try{
    await initialLive();
    const now=Date.now();
    if(!state.lastAnalysis||now-(state.lastAnalysisAt||0)>=state.predictionCadenceMs){await analyze(false);state.lastAnalysisAt=now}
    await newsRefresh();
  }catch(e){}
}

$("analyze").addEventListener("click",()=>analyze(false));
$("retrain").addEventListener("click",()=>analyze(true));
$("interval").addEventListener("change",async()=>{state.interval=$("interval").value;clearPrediction();$("retrain").disabled=state.interval==="1d";$("status").textContent=`LOADING ${state.interval.toUpperCase()}…`;await initialLive();if(state.interval!=="1d" && state.marketOpen)await analyze(false)});
$("symbol").addEventListener("change",async()=>{state.symbol=($("symbol").value||"SENSEX").trim();clearPrediction();$("status").textContent=`LOADING ${state.symbol.toUpperCase()}…`;await initialLive();if(state.interval!=="1d" && state.marketOpen)await analyze(false)});
$("symbol").addEventListener("keydown",e=>{if(e.key==="Enter")$("symbol").dispatchEvent(new Event("change"))});
$("pulse").addEventListener("click",pulse);
$("universe-search-btn").addEventListener("click",()=>loadUniverse($("universe-search").value.trim()));
$("universe-search").addEventListener("keydown",e=>{if(e.key==="Enter")$("universe-search-btn").click()});

boot();
setInterval(async()=>{await initialLive().catch(()=>{})},60000);
setInterval(async()=>{if(state.marketOpen && !state.busy && Date.now()-state.lastAnalysisAt>=state.predictionCadenceMs){await analyze(false).catch(()=>{})}},60000);
setInterval(async()=>{await newsRefresh().catch(()=>{})},600000);
setInterval(async()=>{await pulse().catch(()=>{})},300000);
