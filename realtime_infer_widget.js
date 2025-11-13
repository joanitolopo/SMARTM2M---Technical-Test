(function(){
  // CONFIG
  const WS_URL = "ws://localhost:8000/ws/infer"; // change if server remote
  const WAIT_MS_AFTER_CLICK = 800; // wait after click for animation (tune if needed)
  const LABEL_KEYS = ['front_left','front_right','rear_left','rear_right','hood'];
  const LABEL_PRETTY = {
    'front_left': 'Front Left',
    'front_right': 'Front Right',
    'rear_left': 'Rear Left',
    'rear_right': 'Rear Right',
    'hood': 'Hood'
  };
  const SEND_THROTTLE_MS = 300; // avoid double sends in quick succession

  // Create / inject widget container
  if (document.getElementById('ai-infer-widget-inline')) {
    console.log("Widget already present.");
    return;
  }

  const style = document.createElement('style');
  style.innerHTML = `
  #ai-infer-widget-inline{
    position: fixed; right: 12px; top: 12px; width:320px;
    background: rgba(255,255,255,0.98); border-radius:10px; padding:12px;
    box-shadow:0 8px 30px rgba(0,0,0,0.12); z-index:2147483647;
    font-family: system-ui, Arial, sans-serif;
  }
  #ai-infer-widget-inline h4{margin:0 0 8px 0; font-size:15px}
  #ai-infer-widget-inline .ws-status{font-size:12px;color:#444;margin-bottom:8px}
  #ai-infer-widget-inline .row{display:flex;justify-content:space-between;align-items:center;padding:6px 8px;border-radius:8px;margin-bottom:6px;background:#fafafa}
  .badge-open{background:#d1fae5;color:#065f46;padding:6px 10px;border-radius:999px;font-weight:700}
  .badge-closed{background:#fee2e2;color:#7f1d1d;padding:6px 10px;border-radius:999px;font-weight:700}
  #ai-infer-widget-inline .controls{display:flex;gap:8px;margin-bottom:8px}
  #ai-infer-widget-inline button{padding:6px 8px;font-size:13px;cursor:pointer;border-radius:6px}
  #ai-infer-widget-inline small{display:block;color:#666;margin-top:6px}
  `;
  document.head.appendChild(style);

  const widget = document.createElement('div');
  widget.id = 'ai-infer-widget-inline';
  widget.innerHTML = `
    <h4>AI realtime detected state</h4>
    <div class="ws-status">WS: <span id="ws-conn-inline">disconnected</span></div>
    <div class="controls">
      <button id="ai-start-inline">Connect</button>
      <button id="ai-stop-inline" disabled>Disconnect</button>
      <button id="ai-capture-inline">Capture</button>
    </div>
    <div id="ai-infer-rows"></div>
    <small>Will auto-capture after you press control buttons on the page.</small>
  `;
  document.body.appendChild(widget);

  // build rows
  const rowsRoot = document.getElementById('ai-infer-rows');
  function buildRows(){
    rowsRoot.innerHTML = '';
    LABEL_KEYS.forEach(k=>{
      const r = document.createElement('div');
      r.className = 'row';
      r.id = 'row-'+k;
      r.innerHTML = `<div>${LABEL_PRETTY[k]}</div><div id="val-${k}" class="badge-closed">unknown</div>`;
      rowsRoot.appendChild(r);
    });
  }
  buildRows();

  // WebSocket management
  let ws = null;
  let wsConnected = false;
  function setWsStatus(text){
    const el = document.getElementById('ws-conn-inline');
    if(el) el.textContent = text;
  }

  function connectWS(){
    if(ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    try{
      ws = new WebSocket(WS_URL);
    }catch(e){
      console.error("WS connect error:", e);
      setWsStatus('error');
      return;
    }
    setWsStatus('connecting...');
    ws.onopen = ()=>{ wsConnected = true; setWsStatus('connected'); document.getElementById('ai-start-inline').disabled = true; document.getElementById('ai-stop-inline').disabled = false; };
    ws.onclose = ()=>{ wsConnected = false; setWsStatus('disconnected'); document.getElementById('ai-start-inline').disabled = false; document.getElementById('ai-stop-inline').disabled = true; };
    ws.onerror = (e)=>{ console.warn("WS error", e); setWsStatus('error'); };
    ws.onmessage = (ev)=>{ try { const j = JSON.parse(ev.data); if(j.result) updateRows(j.result); } catch(e){ console.warn("WS msg parse", e); } };
  }
  function closeWS(){ if(ws){ try{ ws.close() }catch(e){} } wsConnected=false; setWsStatus('disconnected'); document.getElementById('ai-start-inline').disabled=false; document.getElementById('ai-stop-inline').disabled=true; }

  document.getElementById('ai-start-inline').addEventListener('click', ()=> connectWS());
  document.getElementById('ai-stop-inline').addEventListener('click', ()=> closeWS());
  document.getElementById('ai-capture-inline').addEventListener('click', ()=> sendCurrentCanvas());

  // update UI rows from server result (expects {key:{prob,pred}, ...})
  function updateRows(result){
    LABEL_KEYS.forEach(k=>{
      const el = document.getElementById('val-'+k);
      if(!el) return;
      const rec = result[k];
      if(!rec){ el.textContent = 'na'; el.className='badge-closed'; return; }
      const prob = (rec.prob!==undefined)? Number(rec.prob).toFixed(3) : 'na';
      const pred = rec.pred;
      if(pred===1 || pred==='1' || pred===true){ el.textContent = `OPEN (${prob})`; el.className='badge-open'; }
      else { el.textContent = `CLOSED (${prob})`; el.className='badge-closed'; }
    });
  }

  // helper: find canvas and overlay buttons
  function findCanvas(){ let c=document.querySelector('#root canvas'); if(!c) c=document.querySelector('canvas'); return c; }
  function findControlButtons(){
    // try to find those five buttons (match by text)
    const all = Array.from(document.querySelectorAll('button'));
    const map = {};
    all.forEach(b=>{
      const t = b.innerText.trim();
      if(!t) return;
      if(t.includes('Front Left')) map['front_left']=b;
      if(t.includes('Front Right')) map['front_right']=b;
      if(t.includes('Rear Left')) map['rear_left']=b;
      if(t.includes('Rear Right')) map['rear_right']=b;
      if(t.includes('Hood')) map['hood']=b;
    });
    return map;
  }

  // hide overlays siblings of canvas ancestor temporarily
  function hideOverlays(){
    try{
      const canvas = findCanvas(); if(!canvas) return null;
      let anc = canvas.parentElement;
      while(anc && anc !== document.body){
        if(anc.childElementCount > 1) break;
        anc = anc.parentElement;
      }
      if(!anc) anc = canvas.parentElement || document.body;
      const changed = [];
      for(const child of Array.from(anc.children)){
        if(child.contains(canvas)) continue;
        child.setAttribute('data-old-display', child.style.display || '');
        child.setAttribute('data-old-visibility', child.style.visibility || '');
        child.setAttribute('data-old-pointer', child.style.pointerEvents || '');
        child.style.display = 'none'; child.style.visibility='hidden'; child.style.pointerEvents='none';
        changed.push(child);
      }
      // hide absolute overlays too
      for(const el of anc.querySelectorAll('*')){
        if(el.contains(canvas)) continue;
        const cs = window.getComputedStyle(el);
        if(cs && cs.position === 'absolute'){
          el.setAttribute('data-old-display', el.style.display || '');
          el.setAttribute('data-old-visibility', el.style.visibility || '');
          el.setAttribute('data-old-pointer', el.style.pointerEvents || '');
          el.style.display='none'; el.style.visibility='hidden'; el.style.pointerEvents='none';
        }
      }
      return true;
    }catch(e){
      console.warn("hideOverlays error", e);
      return false;
    }
  }
  function restoreOverlays(){
    try{
      const elems = document.querySelectorAll('[data-old-display]');
      for(const el of elems){
        el.style.display = el.getAttribute('data-old-display') || '';
        el.style.visibility = el.getAttribute('data-old-visibility') || '';
        el.style.pointerEvents = el.getAttribute('data-old-pointer') || '';
        el.removeAttribute('data-old-display'); el.removeAttribute('data-old-visibility'); el.removeAttribute('data-old-pointer');
      }
    }catch(e){}
  }

  // capture canvas as dataURL
  function captureCanvasDataURL(){
    const canvas = findCanvas();
    if(!canvas) return null;
    try{
      hideOverlays();
      // toDataURL is synchronous
      const d = canvas.toDataURL('image/png');
      restoreOverlays();
      return d;
    }catch(e){
      try{ restoreOverlays(); }catch(_){}
      console.warn("captureCanvasDataURL failed", e);
      return null;
    }
  }

  // throttle sending
  let lastSendTs = 0;
  function sendImageDataURL(dataUrl){
    if(!wsConnected || !ws || ws.readyState !== WebSocket.OPEN){
      console.warn("WS not open - not sending");
      return false;
    }
    const now = Date.now();
    if(now - lastSendTs < SEND_THROTTLE_MS) return false;
    lastSendTs = now;
    const payload = JSON.stringify({ image: dataUrl });
    ws.send(payload);
    return true;
  }

  // public capture & send
  function sendCurrentCanvas(){
    const durl = captureCanvasDataURL();
    if(!durl){ console.warn("No canvas to capture"); return false; }
    const ok = sendImageDataURL(durl);
    if(!ok) console.warn("Send blocked or failed");
    return ok;
  }

  // Listen to control button clicks and auto-capture after WAIT_MS_AFTER_CLICK
  function attachClickListeners(){
    const buttons = findControlButtons();
    if(!buttons || Object.keys(buttons).length === 0){
      console.warn("No control buttons found to attach listeners");
      return;
    }
    console.log("Attaching listeners to control buttons:", Object.keys(buttons));
    Object.entries(buttons).forEach(([key, btn])=>{
      // avoid multiple listeners
      btn.removeEventListener('__ai_data_listener__', dummy);
      const handler = (ev) => {
        // Set a timeout to let animation progress, then capture
        setTimeout(()=>{
          const d = captureCanvasDataURL();
          if(!d){ console.warn("capture after click failed"); return; }
          sendImageDataURL(d);
        }, WAIT_MS_AFTER_CLICK);
      };
      // store handler so we can remove later if needed
      btn.__ai_data_listener__ = handler;
      btn.addEventListener('click', handler);
    });
  }
  function dummy(){}

  setInterval(()=>{ try{ attachClickListeners(); } catch(e){} }, 2000);

  window.AI_INFER = { sendCurrentCanvas };

  console.log("AI realtime widget injected. Click 'Connect' and then interact with page controls. Widget listens to button clicks and auto-captures after click.");
})();
