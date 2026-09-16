/* Runtime-backed workflow. Legacy illustration arrays never stand in for a run. */
const computedFronts = {};
const computedBaselines = {};
const completedTaskIds = new Set();
const originalRenderNsga = renderNsga2Table;
const originalRenderSurr = renderSurrInputs;
const originalDrawPareto = drawPareto;
const originalDrawParetoNice = drawParetoNice;

renderSurrInputs = function() {
  originalRenderSurr();
  const bounds=S.runtimeBounds?.[String(S.surrCond)];if(!bounds)return;
  for(const item of SURR_PARAMS[S.surrCond].items){
    const key=['T','I','Q'].includes(item.id)?`${item.id}_${S.surrCond===2?'B':'A'}`:item.id;
    const range=bounds[key],slider=el('pr_'+item.id);if(!range||!slider)continue;
    slider.min=range[0];slider.max=range[1];slider.value=Math.min(range[1],Math.max(range[0],Number(slider.value)));
    surrSlider(item.id,slider.value);
  }
};

renderNsga2Table = function () {
  const rows = NSGA_DATA[S.nsga2Cond || 1] || [];
  if (!rows.length) {
    el('nsga2-table').innerHTML = '<tr><td colspan="16">Run NSGA-II to calculate a feasible Pareto set for this condition.</td></tr>';
    return;
  }
  originalRenderNsga();
};

function showTaskFailure(message, type='nsga2') {
  const button = el(type === 'rl' ? 'rl-btn-s' : 'nsga2-btn');
  if (button) { button.disabled = false; button.textContent = type === 'rl' ? 'Local CLI training required' : 'Launch NSGA-II Optimization'; }
  if (type === 'nsga2') updateNsga2Prog(0, message);
  showToast('Task not completed', message, 't-err');
}

handleNsga2Done = function (data) {
  if (completedTaskIds.has(data.task_id)) return;
  if (data.success === false || !data.results || !Array.isArray(data.front)) {
    showTaskFailure(data.error || 'The task did not return a computed result set.');
    return;
  }
  completedTaskIds.add(data.task_id);
  for (const [key, rows] of Object.entries(data.results)) {
    const condition = Number(key.replace('cond', ''));
    NSGA_DATA[condition] = rows.map(row => ({...row,
      T: row.T === undefined ? undefined : +row.T.toFixed(2),
      I: row.I === undefined ? undefined : +row.I.toFixed(0),
      Q: row.Q === undefined ? undefined : +row.Q.toFixed(2),
      TA: row.TA === undefined ? undefined : +row.TA.toFixed(2),
      IA: row.IA === undefined ? undefined : +row.IA.toFixed(0),
      QA: row.QA === undefined ? undefined : +row.QA.toFixed(2),
      TB: row.TB === undefined ? undefined : +row.TB.toFixed(2),
      IB: row.IB === undefined ? undefined : +row.IB.toFixed(0),
      QB: row.QB === undefined ? undefined : +row.QB.toFixed(2),
      Cu_out: +row.Cu_out.toFixed(3), As_out: +row.As_out.toFixed(3),
      _raw: row, _cond: condition}));
    computedFronts[condition] = data.front;
    computedBaselines[condition] = data.baseline;
  }
  updateNsga2Prog(100, `Computed ${data.front.length} retained candidates · ${data.evaluations} evaluations · ${Number(data.runtime_seconds).toFixed(2)} s`);
  el('nsga2-btn').disabled = false; el('nsga2-btn').textContent = 'Launch NSGA-II Optimization';
  renderNsga2Table(); drawParetoNice('pareto1','cu-as'); drawParetoNice('pareto2','e-profit');
  showToast('NSGA-II completed', 'Computed results saved locally. Select a scenario and apply it to Overview.', 't-ok');
};

handleRLDone = function (data) {
  showTaskFailure(data.error || 'Use the local research CLI for PPO training and verified saved policies for inference.', 'rl');
};

pollTask = async function (id, type) {
  for (let attempt=0; attempt<120; attempt++) {
    await sleep(500);
    const task = await API(`/api/task/${id}`);
    if (task.status === 'completed') {
      if (type === 'nsga2') handleNsga2Done({task_id:id,success:true,...task.result});
      else handleRLDone({task_id:id,...task.result});
      return;
    }
    if (['failed','error','not_found'].includes(task.status) || task.error) {
      showTaskFailure(task.error || task.message || 'Task unavailable',type); return;
    }
    if (type === 'nsga2') updateNsga2Prog(task.progress || 5,'Computing the requested NSGA-II search…');
  }
  showTaskFailure('The browser stopped waiting. Check Experiment Records before starting another run.',type);
};

function drawComputedPareto(id, xkey, ykey, xlabel, ylabel) {
  const canvas = el(id); if (!canvas) return;
  const width = canvas.offsetWidth || 420, height = canvas.offsetHeight || 220;
  const dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(width*dpr); canvas.height=Math.round(height*dpr);
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,width,height);ctx.fillStyle='#6a6860';ctx.font='14px sans-serif';
  const rows=computedFronts[S.nsga2Cond||1]||[];
  if (!rows.length) {ctx.fillText('Run optimization to populate this chart.',20,45);return;}
  const xs=rows.map(r=>r[xkey]),ys=rows.map(r=>r[ykey]);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const px=x=>62+(x-x0)/(x1-x0||1)*(width-88);
  const py=y=>height-46-(y-y0)/(y1-y0||1)*(height-76);
  ctx.strokeStyle='#d8d5cc';ctx.beginPath();ctx.moveTo(62,24);ctx.lineTo(62,height-46);ctx.lineTo(width-20,height-46);ctx.stroke();
  ctx.fillStyle='#2563a8';for(const row of rows){ctx.beginPath();ctx.arc(px(row[xkey]),py(row[ykey]),3.5,0,2*Math.PI);ctx.fill();}
  ctx.fillStyle='#3a3830';ctx.textAlign='center';ctx.fillText(xlabel,width/2,height-8);
  ctx.save();ctx.translate(15,height/2);ctx.rotate(-Math.PI/2);ctx.fillText(ylabel,0,0);ctx.restore();
  ctx.textAlign='left';ctx.fillText(x0.toFixed(2),62,height-28);ctx.textAlign='right';ctx.fillText(x1.toFixed(2),width-20,height-28);
  ctx.fillText(y0.toFixed(2),57,height-43);ctx.fillText(y1.toFixed(2),57,29);
}

drawParetoNice = function (id,type) {
  if (S.runtimeReady) return type==='cu-as'
    ? drawComputedPareto(id,'Cu_out','As_out','Cu_out (g/L)','As_out (g/L)')
    : drawComputedPareto(id,'E','profit','Energy (kWh)','Profit (10⁴ CNY)');
  return originalDrawParetoNice(id,type);
};
drawPareto = function(id,xi,yi) {
  if(S.runtimeReady) return drawParetoNice(id,xi===0?'cu-as':'e-profit');
  return originalDrawPareto(id,xi,yi);
};

function showComputedConstraints(result) {
  if (!result.constraints) return;
  const labels={C1_Cu:'C₁ Cu_out ≤ 8.0 g/L',C2_As:'C₂ As_out ≥ 4.0 g/L',
    C3_J:'C₃ J ≤ 340 A/m²',C4_V:'C₄ V_cell ≤ 2.5 V',C5_CuAs:'C₅ Cu/As ≤ 0.80'};
  el('ov-constraints').replaceChildren();
  for(const [key,value] of Object.entries(result.constraints)) {
    const row=document.createElement('div');row.className='cst-item';
    const dot=document.createElement('div');dot.className='cst-dot';dot.style.background=value.ok?'var(--green)':'var(--red)';
    const name=document.createElement('div');name.className='cst-name';name.textContent=labels[key]||key;
    const measured=document.createElement('div');measured.className='cst-val';
    measured.style.color=value.ok?'var(--green)':'var(--red)';measured.textContent=value.val===null?'Invalid':Number(value.val).toFixed(3);
    row.append(dot,name,measured);el('ov-constraints').append(row);
  }
}

updateKPIs = function(cu, arsenic, energy, profit, source) {
  const values=[cu,arsenic,energy,profit];
  const ids=['cu','as','e','p'], units=['g/L','g/L','kWh','×10⁴ CNY'];
  S.kpiSrc=source;
  const baseline=S.kpiBaseline;
  const refs=baseline?[baseline.Cu_out,baseline.As_out,baseline.E,baseline.profit]:null;
  values.forEach((value,index)=>{
    const parent=el('kpi-'+ids[index]);parent.replaceChildren(document.createTextNode(Number(value).toFixed(index===2?0:2)));
    const unit=document.createElement('span');unit.className='kpi-unit';unit.textContent=units[index];parent.append(unit);
    const label=document.querySelectorAll('#kpi-row .kpi-label')[index];
    const delta=el('kpi-'+ids[index]+'-delta');
    if(refs&&source==='nsga2'){
      label.textContent=`Same-input baseline: ${Number(refs[index]).toFixed(2)} ${units[index]}`;
      const direction=[-1,1,-1,1][index];
      delta.textContent=Math.abs(refs[index])>1e-12?`${(direction*(value-refs[index])/Math.abs(refs[index])*100).toFixed(1)}% ${['Cu reduction','As retention','energy reduction','profit change'][index]}`:'Baseline is zero';
    }else{
      label.textContent=S.runtimeMode==='synthetic_demo'?'Synthetic scenario · not plant measurements':'Local model output';
      delta.textContent=source==='initial'?'Initial scenario':'Computed result';
    }
  });
  el('src-badge').textContent=`Source: ${S.runtimeMode==='synthetic_demo'?'SYNTHETIC DEMO · ':''}${source}`;
};

function applyComputedOverview(result, source) {
  const condition=result.condition||S.surrCond||1;
  S.overviewResults??={};S.overviewResults[condition]={result,source};
  S.ovCond=condition;
  document.querySelectorAll('#ov-cond-btns .seg-btn').forEach((button,index)=>button.classList.toggle('active',index===condition-1));
  renderOvFlow(condition);
  const p=result.updated_params||result.params||result;
  const units=condition===3?['A','B']:[condition===1?'A':'B'];
  const pairs=[['Cu_in (g/L)',p.Cu_in],['t (h)',p.t]];
  const labels={T:'°C',I:'A',Q:'m³/h'};
  units.forEach(unit=>['T','I','Q'].forEach(key=>pairs.push([`${key}_${unit} (${labels[key]})`,p[`${key}_${unit}`]??p[`${key}${unit}`]??p[key]])));
  el('ov-params').replaceChildren();
  for(const [key,value] of pairs){
    const box=document.createElement('div');box.className='stat-box';
    const name=document.createElement('div');name.className='stat-lbl';name.textContent=key;
    const val=document.createElement('div');val.style.fontSize='15px';val.textContent=value==null?'—':Number(value).toLocaleString('en-US',{maximumFractionDigits:2});
    box.append(name,val);el('ov-params').append(box);
  }
  S.kpiBaseline=source==='nsga2'?computedBaselines[condition]:null;
  updateKPIs(result.Cu_out,result.As_out,result.E,result.profit,source);
  showComputedConstraints(result);gotoTab('overview');
}

ovSetCond = async function(condition) {
  const saved=S.overviewResults?.[condition];
  if(saved){applyComputedOverview(saved.result,saved.source);return;}
  const params=S.runtimeDefaults?.[String(condition)];
  if(!params)return;
  const response=await POST('/api/surrogate/predict',{condition,params});
  if(response.status!=='success'){showToast('Prediction unavailable',response.message||response.error,'t-warn');return;}
  applyComputedOverview({...response.result,params},'initial');
};

function fillSurrogateFromRecord(record) {
  const condition={three_stage:1,four_stage:2,serial:3}[record.mode];
  if(!condition)return;
  S.surrCond=condition;renderSurrInputs();
  document.querySelectorAll('#surr-cond-btns .seg-btn').forEach((button,index)=>button.classList.toggle('active',index===condition-1));
  for(const field of SURR_PARAMS[condition].items){
    const key=field.id;
    const value=key==='Cu_in'?record.cu_in:key==='t'?record.duration:
      key.startsWith('T')?record.temperature:key.startsWith('I')?record.current:record.flow;
    const slider=el('pr_'+key);slider.value=value;surrSlider(key,slider.value);
  }
}

sendToSurrogate = function() {
  fillSurrogateFromRecord({mode:el('m-mode').value,cu_in:+el('m-cu').value,
    temperature:+el('m-temp').value,current:+el('m-current').value,
    flow:+el('m-flow').value,duration:+el('m-dur').value});
  gotoTab('surrogate');
};

surrFill = async function() {
  const response=await API('/api/data/list?limit=1');
  if(!response.data?.length){showToast('No stored records','Import a validated record first.','t-warn');return;}
  fillSurrogateFromRecord(response.data[0]);
  showToast('Inputs loaded','Using the latest local record.','t-ok');
};

applyNsga2Row = function(index) {
  const display=(NSGA_DATA[S.nsga2Cond]||[])[index];if(!display)return;
  const row=display._raw||display;
  if (!row.feasible) {showToast('Selection blocked','The candidate did not pass all five constraints.','t-err');return;}
  applyComputedOverview(row,'nsga2');
  showToast('Candidate applied to the dashboard', `${display.scene} · ${row.model_source}. No plant control command was sent.`, 't-ok');
};

applySurrToOverview = function () {
  if(!S.prediction)return;
  applyComputedOverview(S.prediction,'prediction');
};

exportExp = async function(format) {
  if(!_selExp){showToast('Select an experiment first','','t-warn');return;}
  const response=await POST('/api/export',{experiment_id:_selExp.experiment_id,format});
  if(response.status!=='success'){showToast('Export failed',response.message||response.error,'t-err');return;}
  const link=document.createElement('a');link.href=response.url;link.download=response.filename;link.click();
  showToast('Computed candidates exported', `${response.rows} rows downloaded.`, 't-ok');
};

runPreprocess = async function () {
  const report=await POST('/api/data/preprocess',{});
  el('pp-status').textContent=report.status==='success'
    ? `Validated ${report.checked} stored records; ${report.invalid} invalid. No dataset was changed.`
    : report.message||report.error;
};

async function inferLocalPolicy() {
  const result=await POST('/api/rl/predict',{condition:S.surrCond,params:getSurrParams(),weights:[0.25,0.25,0.25,0.25]});
  if(result.status!=='success'){showToast('Policy unavailable',result.message||result.error,'t-warn');return;}
  const r=result.result;
  if(!r.feasible){showToast('Candidate rejected','The policy proposal violates the configured process envelope.','t-warn');return;}
  applyComputedOverview(r,'rl');
  showToast('Saved policy evaluated','Normalized action increments decoded and all five constraints checked.','t-ok');
}

function openComputedOptimization() {
  S.nsga2Cond=S.surrCond;
  document.querySelectorAll('#nsga2-cond-btns .seg-btn').forEach((button,index)=>button.classList.toggle('active',index===S.nsga2Cond-1));
  gotoTab('nsga2');renderNsga2Table();
}

async function refreshResearchModelStatus(attempt=0) {
  const status=await API('/api/health');
  if(status.status!=='ok')return;
  el('hd-model-txt').textContent=status.surrogate_status==='ready'?`${status.models_loaded} ready`:status.surrogate_status;
  el('hd-model-dot').className='sdot '+(status.surrogate_status==='ready'?'dot-ok':status.model_load_finished?'dot-er':'dot-ld');
  el('surr-model-src').textContent=`${status.models_loaded} local surrogate files · ${status.rl_models_loaded} local policies`;
  el('local-policy-btn').disabled=!status.model_load_finished||status.rl_models_loaded<1||status.models_loaded<3;
  if(!status.model_load_finished&&attempt<60)setTimeout(()=>refreshResearchModelStatus(attempt+1),500);
}

async function initializeReleaseWorkflow() {
  const health=await API('/api/health');
  if(health.status!=='ok'){el('runtime-notice').textContent='Backend unavailable. Start run_demo.py to use the verified local workflow.';return;}
  S.runtimeMode=health.mode;S.runtimeReady=true;
  S.runtimeDefaults=health.defaults;
  S.runtimeBounds=health.bounds;
  document.body.dataset.runtimeMode=health.mode;
  const demo=health.mode==='synthetic_demo';
  el('runtime-notice').textContent=demo
    ? 'SYNTHETIC DEMO · Inputs and predictions are generated for software demonstration. NSGA-II runs on demand; no industrial data or trained research weights are loaded.'
    : 'LOCAL RESEARCH · Predictions require your private model assets. Historical layout illustrations are not live experimental results. Optimization/training use the local research CLI.';
  for(const condition of [1,2,3])NSGA_DATA[condition]=[];
  renderNsga2Table();
  renderSurrInputs();
  // Unverified hard-coded comparison plots and training animations are not runtime evidence.
  for(const name of ['compare','saferl','operation'])document.querySelector(`#nav [data-panel="${name}"]`).hidden=true;
  for(const id of ['wf-rl','wf-advisory']){
    const step=el(id);step.hidden=true;
    if(step.previousElementSibling?.tagName==='SPAN')step.previousElementSibling.hidden=true;
  }
  document.querySelector('#panel-overview>.g3').hidden=true;
  document.querySelector('#panel-overview>.card:last-child').hidden=true;
  document.querySelector('#panel-surrogate>.g3').hidden=true;
  document.querySelectorAll('#nsga2-constraints input').forEach(input=>input.disabled=true);
  document.querySelectorAll('#panel-data input[type="checkbox"]').forEach(input=>{input.disabled=true;input.checked=false;});
  document.querySelectorAll('#panel-data .itab')[2].hidden=true;
  document.querySelectorAll('#nsga2-run-mode-btns button')[1].hidden=true;
  el('local-policy-btn').disabled=demo||health.rl_models_loaded<1;
  el('nsga2-btn').disabled=!demo;
  el('hd-model-txt').textContent=demo?'Synthetic demo':health.surrogate_status;
  el('hd-model-dot').className='sdot '+(demo?'dot-ld':health.models_loaded?'dot-ok':'dot-er');
  el('surr-model-src').textContent=demo?'Explicit synthetic formula · no trained-model accuracy claimed':`${health.models_loaded} local surrogate files available`;
  document.querySelectorAll('#panel-overview .kpi-label').forEach(label=>label.textContent=demo?'Synthetic scenario · not plant measurements':'Awaiting a verified local prediction');
  if(demo){
    const result=await POST('/api/surrogate/predict',{condition:1,params:health.defaults['1']});
    if(result.status==='success'){
      applyComputedOverview({...result.result,params:health.defaults['1']},'initial');
    }
  }else{
    for(const id of ['kpi-cu','kpi-as','kpi-e','kpi-p'])el(id).textContent='—';
    for(const id of ['kpi-cu-delta','kpi-as-delta','kpi-e-delta','kpi-p-delta'])el(id).textContent='Run local model prediction';
    el('ov-constraints').textContent='Constraint evaluation awaits a model prediction.';
    refreshResearchModelStatus();
  }
  drawParetoNice('pareto1','cu-as');drawParetoNice('pareto2','e-profit');
}
