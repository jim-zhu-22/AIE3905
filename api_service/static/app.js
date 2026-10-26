'use strict';
const $=id=>document.getElementById(id);
const state={mode:'sample',modules:[],result:null,job:null,token:0,historyOffset:0,preview:null,jobs:new Map(),terminalSeen:new Map(),streamConnected:false,liveGeneration:0};
const names={queued:'排队中',running:'处理中',completed:'已完成',partial:'部分完成',failed:'失败',cancelled:'已取消',ok:'完成',no_target:'无目标',ready:'可用',disabled:'未启用',unavailable:'待接入',unknown:'未知',not_requested:'未请求',waiting:'待发送',uploading:'上传中',skipped:'跳过',error:'失败'};
function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function badge(status){return el('span',names[status]||status,'badge '+(['partial','unknown','skipped','disabled','unavailable'].includes(status)?'warning':['failed','error'].includes(status)?'error':'neutral'));}
let toastTimer;
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,6500);}
async function api(path,options){const response=await fetch(path,options);let data;try{data=await response.json();}catch{throw new Error('服务返回了无法读取的内容，请检查是否已启动。');}if(!response.ok){const error=data.error;const issue=new Error((error?.message||'请求失败')+(error?.request_id?'（编号 '+error.request_id.slice(0,10)+'）':''));issue.status=response.status;issue.code=error?.code;issue.retryAfter=Number(response.headers.get('Retry-After')||0);throw issue;}return data;}
function changeView(id){if(!['workbench','history','modules','guide'].includes(id))id='workbench';for(const view of document.querySelectorAll('.view'))view.hidden=view.id!==id;for(const item of document.querySelectorAll('.nav-item'))item.classList.toggle('active',item.dataset.view===id);$('viewLabel').textContent={workbench:'工作台',history:'任务记录',modules:'模块接入',guide:'使用说明'}[id];if(location.hash!=='#'+id)history.replaceState(null,'','#'+id);if(id==='history')loadHistory().catch(error=>toast(error.message));}
for(const button of document.querySelectorAll('[data-view]'))button.onclick=()=>changeView(button.dataset.view);
window.addEventListener('hashchange',()=>changeView(location.hash.slice(1)));
function setMode(mode){state.mode=mode;$('startJob').textContent=mode==='upload'?'开始 / 继续连续处理':'生成任务结果 ↗';$('submitHint').textContent=mode==='upload'?'开始后可继续添加图片；每张处理完后自动显示结果。':'每次处理独立编号，结果可查询、可下载。';$('sampleInputs').hidden=mode!=='sample';$('uploadInputs').hidden=mode!=='upload';$('sampleMode').classList.toggle('selected',mode==='sample');$('uploadMode').classList.toggle('selected',mode==='upload');}
$('sampleMode').onclick=()=>setMode('sample');$('uploadMode').onclick=()=>setMode('upload');
async function refreshModules(){const data=await api('/v1/modules');state.modules=data.modules;$('serviceStatus').textContent=data.workspace_available?'工作区已连接':'结果读取模式';$('serviceStatus').className='badge '+(data.workspace_available?'':'warning');$('modelNotice').textContent=data.modules.find(m=>m.id==='segmentation')?.state==='ready'?'模型已就绪，可以处理新图片。':'模型尚未启用。可先使用已有分割包；新图片提交会给出明确提示。';const envLink=el('a',' 本地 CPU/GPU 环境准备 ↗');envLink.href='/v1/environment-guide';envLink.target='_blank';envLink.rel='noopener';$('modelNotice').append(envLink);$('moduleChoices').replaceChildren();$('moduleCards').replaceChildren();for(const module of data.modules){const card=el('article',undefined,'panel module-card');const header=el('div',undefined,'card-head');header.append(el('h2',module.title),badge(module.state));card.append(header,el('p',module.reason||(module.id==='segmentation'?'复用仓库中的目标检测与实例分割程序。':'已配置模块，任务处理时调用其真实输出。')),el('small','接入方式：'+module.adapter));$('moduleCards').append(card);if(module.id!=='segmentation'){const row=el('label',undefined,'module-choice');const input=document.createElement('input');input.type='checkbox';input.value=module.id;input.disabled=module.state!=='ready';row.append(input,el('span',module.title),badge(module.state));$('moduleChoices').append(row);}}if(data.config_error)toast(data.config_error);}
async function refreshSamples(selected){const data=await api('/v1/examples');$('sampleSelect').replaceChildren(...data.examples.map(item=>new Option(item.sample_id,item.sample_id)));if(!data.examples.length)$('sampleSelect').append(new Option('暂无样例，请先导入成果包',''));if(selected)$('sampleSelect').value=selected;}
function selectedModules(){return [...$('moduleChoices').querySelectorAll('input:checked')].map(input=>input.value);}
function resetResult(){state.result=null;$('imageStage').hidden=true;$('emptyState').hidden=false;$('resultStats').hidden=true;$('resultActions').hidden=true;$('instancePanel').hidden=true;$('warningBox').hidden=true;$('resultTitle').textContent='结果预览';$('showOriginal').disabled=true;}
$('rgbFile').onchange=()=>{const files=[...$('rgbFile').files];if(!files.length)return;try{const added=uploadQueue.add(files);$('fileLabel').textContent='已添加 '+added+' 张，可继续选择';}catch(error){toast(error.message);}finally{$('rgbFile').value='';}};
async function submit(){
  if(state.mode==='upload'){
    try{
      if(!$('prompt').value.trim())throw new Error('请填写目标描述。');
      const pending=uploadQueue.items.filter(item=>item.status==='waiting');
      if(!pending.length&&!uploadQueue.enabled)throw new Error('请先添加需要处理的图片。');
      const depth=$('depthFile').files[0]||null;
      if(depth&&pending.length!==1)throw new Error('一张深度图只能与一张 RGB 配对。批量 RGB 请先清空深度图，或通过 API 逐张配对提交。');
      const unit=$('depthUnit').value;
      uploadQueue.start({prompt:$('prompt').value,modules:selectedModules(),depth,depthTargetKey:depth?pending[0].id:null,
        meta:{depth_unit:unit,alignment:$('alignment').value,depth_scale_to_m:unit==='mm'?0.001:unit==='m'?1:null}});
    }catch(error){toast(error.message);}return;
  }
  $('startJob').disabled=true;
  try{const id=$('sampleSelect').value;if(!id)throw new Error('请先选择或导入一份已有成果。');
    const job=await api('/v1/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sample_id:id,modules:selectedModules()})});
    resetResult();changeView('workbench');state.job=job;state.token++;await followJob(job,state.token);
  }catch(error){toast(error.message);}finally{$('startJob').disabled=false;}
}
$('startJob').onclick=submit;$('trySample').onclick=()=>{setMode('sample');if([...$('sampleSelect').options].some(o=>o.value==='frame_211'))$('sampleSelect').value='frame_211';submit();};
async function followJob(job,token){if(token!==state.token)return;state.job=job;$('jobProgress').hidden=!['queued','running'].includes(job.status);$('progressText').textContent=job.stage+(job.total?' · '+job.done+'/'+job.total:'');$('cancelJob').hidden=job.status!=='queued';if(job.result_ref){const result=await api(job.result_ref);if(token===state.token)renderResult(result);return;}if(['failed','cancelled'].includes(job.status)){toast(job.error||names[job.status]);return;}setTimeout(async()=>{if(token!==state.token)return;try{await followJob(await api('/v1/jobs/'+job.id),token);}catch(error){$('jobProgress').hidden=true;toast(error.message);}},850);}
$('cancelJob').onclick=async()=>{if(!state.job)return;try{const job=await api('/v1/jobs/'+state.job.id+'/cancel',{method:'POST'});state.token++;await followJob(job,state.token);}catch(error){toast(error.message);}};
function renderResult(result){state.result=result;$('emptyState').hidden=true;$('jobProgress').hidden=true;$('imageStage').hidden=false;$('resultStats').hidden=false;$('resultActions').hidden=false;$('instancePanel').hidden=false;$('resultTitle').textContent=result.frame_id;$('resultImage').src=result.overlay_ref;$('sourceBadge').textContent=result.execution.new_inference?'新图片推理':'已有分割成果 · 未重新推理';$('showOriginal').disabled=!result.image_ref;$('showOverlay').classList.add('selected');$('showOriginal').classList.remove('selected');$('statInstances').textContent=result.instance_count;$('statSize').textContent=result.image_size.join(' × ');$('statModules').textContent=Object.keys(result.stages).filter(key=>key!=='segmentation').length;$('statState').textContent=names[result.status]||result.status;$('downloadZip').href=result.download_ref;$('reportLink').href=result.report_ref;$('instanceSearch').value='';renderInstances();if(result.instances.length)selectInstance(result.instances[0]);else $('instanceDetail').replaceChildren(el('p','没有匹配的物体。'));const incomplete=Object.entries(result.stages).filter(([key,value])=>key!=='segmentation'&&value.status!=='ok');$('warningBox').hidden=!incomplete.length;if(incomplete.length)$('warningBox').textContent='部分属性未完成：'+incomplete.map(([key])=>state.modules.find(module=>module.id===key)?.title||key).join('、')+'。选择实例可查看具体原因；有效分割结果已保留。';}
function renderInstances(){if(!state.result)return;const query=$('instanceSearch').value.toLowerCase();$('instanceRows').replaceChildren();for(const item of state.result.instances.filter(item=>(item.instance_id+' '+item.object_label).toLowerCase().includes(query))){const row=el('tr');row.dataset.instance=item.instance_id;const open=el('button',item.instance_id,'text-button');open.onclick=()=>selectInstance(item);const first=el('td');first.append(open);row.append(first,el('td',item.object_label),el('td',item.det_score.toFixed(3)));const status=el('td');const values=Object.values(item.attributes);status.append(badge(!values.length?'not_requested':values.every(value=>value.status==='ok')?'ok':'partial'));row.append(status);row.onclick=()=>selectInstance(item);$('instanceRows').append(row);}}
$('instanceSearch').oninput=renderInstances;
function selectInstance(item){for(const row of $('instanceRows').children)row.classList.toggle('selected',row.dataset.instance===item.instance_id);const detail=$('instanceDetail');detail.replaceChildren(el('h3',item.instance_id+' · '+item.object_label));const image=document.createElement('img');image.src=item.mask_ref;image.alt=item.instance_id+' 二值 mask';detail.append(image);const link=el('a','打开 mask ↗');link.href=item.mask_ref;link.target='_blank';link.rel='noopener';detail.append(link,el('p','像素框：'+item.bbox_xyxy.join(', ')));if(!Object.keys(item.attributes).length)detail.append(el('p','本任务未请求属性模块。'));for(const [key,value] of Object.entries(item.attributes)){const line=el('div',undefined,'attribute-line');line.append(el('strong',state.modules.find(module=>module.id===key)?.title||key),document.createTextNode(' '),badge(value.status));if(value.label)line.append(el('p',value.label+(value.score!==null?' · '+value.score.toFixed(3):'')));if(value.reason)line.append(el('p',value.reason));detail.append(line);}}
$('showOverlay').onclick=()=>{if(!state.result)return;$('resultImage').src=state.result.overlay_ref;$('showOverlay').classList.add('selected');$('showOriginal').classList.remove('selected');};$('showOriginal').onclick=()=>{if(!state.result?.image_ref)return;$('resultImage').src=state.result.image_ref;$('showOriginal').classList.add('selected');$('showOverlay').classList.remove('selected');};$('showJson').onclick=()=>{if(!state.result)return;$('jsonContent').textContent=JSON.stringify(state.result,null,2);$('jsonDialog').showModal();};$('closeJson').onclick=()=>$('jsonDialog').close();
async function loadHistory(){const data=await api('/v1/jobs?limit=20&offset='+state.historyOffset);$('historyRows').replaceChildren();for(const job of data.jobs){const row=el('tr');const title=el('td',job.frame_id);title.append(el('small',job.id.slice(0,18)+'…'));const time=new Date(job.created_at).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});const status=el('td');status.append(badge(job.status));const action=el('td');const button=el('button','查看','text-button');button.onclick=async()=>{$('liveFollow').checked=false;state.liveGeneration++;changeView('workbench');state.token++;resetResult();try{await followJob(await api('/v1/jobs/'+job.id),state.token);}catch(error){toast(error.message);}};action.append(button);row.append(title,el('td',job.source==='sample'?'已有成果':'新图片'),el('td',time),status,action);$('historyRows').append(row);}if(!data.jobs.length){const row=el('tr');const cell=el('td','还没有任务，在工作台生成第一份结果。');cell.colSpan=5;row.append(cell);$('historyRows').append(row);}$('historyCount').textContent='共 '+data.total+' 个任务';$('historyPrev').disabled=state.historyOffset===0;$('historyNext').disabled=state.historyOffset+20>=data.total;}
$('refreshHistory').onclick=()=>loadHistory().catch(error=>toast(error.message));$('historyPrev').onclick=()=>{state.historyOffset=Math.max(0,state.historyOffset-20);loadHistory().catch(error=>toast(error.message));};$('historyNext').onclick=()=>{state.historyOffset+=20;loadHistory().catch(error=>toast(error.message));};
$('importBundle').onclick=async()=>{const id=$('bundleId').value.trim(),file=$('bundleFile').files[0];if(!/^[A-Za-z0-9_-]{1,100}$/.test(id)||!file){toast('请填写字母、数字或下划线组成的样例名称，并选择 ZIP。');return;}$('importBundle').disabled=true;try{const body=new FormData();body.append('sample_id',id);body.append('bundle',file);const data=await api('/v1/bundles/import',{method:'POST',body});await refreshSamples(data.sample_id);toast('导入完成：'+data.instance_count+' 个实例。');}catch(error){toast(error.message);}finally{$('importBundle').disabled=false;}};
changeView(location.hash.slice(1)||'workbench');Promise.all([refreshModules(),refreshSamples()]).then(connectEvents).catch(error=>{$('serviceStatus').textContent='连接失败';toast(error.message);});

const isTerminal=job=>['completed','partial','failed','cancelled'].includes(job.status);
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
function renderQueue(queue){
  $('uploadQueue').hidden=!queue.items.length;
  const done=queue.items.filter(item=>['completed','partial'].includes(item.status)).length;
  const waiting=queue.items.filter(item=>item.status==='waiting').length;
  $('queueSummary').textContent=`${queue.enabled?'连续接收中':'发送已暂停'} · 完成 ${done}/${queue.items.length} · 待发 ${waiting}`;
  $('pauseUploads').disabled=!queue.enabled;$('queueRows').replaceChildren();
  for(const item of queue.items){const row=el('div',undefined,'queue-row');const info=el('div');info.append(el('strong',item.file.name));
    if(item.error)info.append(el('small',item.error));else if(item.job)info.append(el('small',item.job.stage));
    const actions=el('div',undefined,'queue-actions');actions.append(badge(item.status));
    if(item.job){const view=el('button','查看','text-button');view.onclick=async()=>{$('liveFollow').checked=false;state.liveGeneration++;state.token++;changeView('workbench');try{await followJob(await api('/v1/jobs/'+item.job.id),state.token);}catch(error){toast(error.message);}};actions.append(view);}
    if(['failed','cancelled'].includes(item.status)){const retry=el('button','重新排队','text-button');retry.onclick=()=>queue.retry(item.id);actions.append(retry);}
    if(item.status==='waiting'&&!item.job){const remove=el('button','移除','text-button');remove.onclick=()=>queue.remove(item.id);actions.append(remove);}row.append(info,actions);$('queueRows').append(row);
  }
}
async function displayFinished(job){
  if(!$('liveFollow').checked||$('workbench').hidden||!job.result_ref)return;
  const generation=++state.liveGeneration;
  try{const result=await api(job.result_ref);if(generation===state.liveGeneration&&$('liveFollow').checked)renderResult(result);}
  catch(error){toast(error.message);}
}
function receiveJob(job,{snapshot=false}={}){
  const previous=state.jobs.get(job.id);state.jobs.set(job.id,job);uploadQueue.update(job);
  if(isTerminal(job)){
    const first=state.terminalSeen.get(job.id)!==job.status;state.terminalSeen.set(job.id,job.status);
    if(first&&!snapshot){if($('liveFollow').checked&&!$('workbench').hidden)state.token++;if(state.job?.id===job.id)$('jobProgress').hidden=true;if(job.result_ref)displayFinished(job);else if(!previous||previous.status!==job.status)toast(job.error||names[job.status]);}
  }else if($('liveFollow').checked&&!$('workbench').hidden){
    state.job=job;$('jobProgress').hidden=false;$('progressText').textContent=job.frame_id+' · '+job.stage+(job.total?' · '+job.done+'/'+job.total:'');$('cancelJob').hidden=job.status!=='queued';
  }
}
const uploadQueue=new ImageQueue({
  submit:async item=>{
    const options=item.options,body=new FormData();body.append('rgb',item.file);
    body.append('frame_id',item.file.name.replace(/\.[^.]+$/,'').replace(/[^A-Za-z0-9_-]/g,'_').slice(0,80)||'uploaded_frame');
    body.append('prompt',options.prompt);body.append('modules',JSON.stringify(options.modules));body.append('meta',JSON.stringify(options.meta));
    if(options.depth&&options.depthTargetKey===item.id)body.append('depth',options.depth);
    return api('/v1/perceive',{method:'POST',headers:{'Idempotency-Key':item.key},body});
  },
  wait:async job=>{
    let current=job;while(!isTerminal(current)){await sleep(500);current=state.jobs.get(job.id)||current;
      if(!isTerminal(current))current=await api('/v1/jobs/'+job.id);receiveJob(current);}
    return current;
  },onChange:renderQueue,onJob:job=>receiveJob(job)
});
$('pauseUploads').onclick=()=>uploadQueue.pause();$('clearUploads').onclick=()=>uploadQueue.clearFinished();
$('liveFollow').onchange=()=>{state.liveGeneration++;if($('liveFollow').checked){const latest=[...state.jobs.values()].filter(job=>job.result_ref).sort((a,b)=>b.updated_at.localeCompare(a.updated_at))[0];if(latest)displayFinished(latest);}};
function receiveSnapshot(data){for(const job of data.jobs)receiveJob(job,{snapshot:true});const latest=data.jobs.filter(job=>job.result_ref).sort((a,b)=>b.updated_at.localeCompare(a.updated_at))[0];if(latest&&!state.result)displayFinished(latest);}
function connectEvents(){
  const source=new EventSource('/v1/events');
  source.onopen=()=>{state.streamConnected=true;$('eventStatus').textContent='结果通知已连接';$('eventStatus').className='badge';};
  source.addEventListener('snapshot',event=>receiveSnapshot(JSON.parse(event.data)));
  source.addEventListener('job',event=>receiveJob(JSON.parse(event.data)));
  source.onerror=()=>{state.streamConnected=false;$('eventStatus').textContent='重连中 · 自动查询备用';$('eventStatus').className='badge warning';};
  const fallback=setInterval(async()=>{if(state.streamConnected||document.hidden)return;try{const data=await api('/v1/jobs?limit=30');for(const job of data.jobs.reverse())receiveJob(job);}catch{}},2000);
  window.addEventListener('pagehide',()=>{source.close();clearInterval(fallback);},{once:true});
}
