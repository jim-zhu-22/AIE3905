/* Read-only walkthrough of persisted pipeline artifacts. No simulated inference. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const node = (tag, text, cls) => {
    const item = document.createElement(tag);
    if (text !== undefined) item.textContent = text;
    if (cls) item.className = cls;
    return item;
  };
  const stages = [
    ['input', '查看原图', '本次分析的画面'],
    ['segmentation', '识别物体', '找到位置与轮廓'],
    ['roi', '查看特写', '逐个观察物体'],
    ['material', '分析材质', '查看材质候选'],
    ['depth', '深度信息', '附加数据'],
    ['result', '汇总与导出', '获取完整结果']
  ];
  const filenames = ['preprocess.json', 'segmentation.json', 'material_input.json', 'material_result.json', 'result.json'];
  const labels = {ok:'完成', partial:'部分完成', unknown:'未提供结论', skipped:'跳过', error:'失败', no_match:'无目标', not_run:'未运行', unconfirmed:'未标定', aligned:'已声明配准', reference_only:'仅保存引用', legacy_statistics_only:'历史统计引用'};
  const demo = {result:null, docs:{}, errors:[], step:0, instance:null, token:0, snapshot:null};
  const offline = window.PIPELINE_DEMO_DATA;
  const display = value => value === undefined || value === null ? '未提供' : Array.isArray(value) ? value.join(' × ') : String(value);
  const status = value => labels[value] || display(value);
  const score = value => typeof value === 'number' ? value.toFixed(3) : '未提供';
  const friendly = value => ({packet:'包装件',carton:'纸箱',box:'箱体',bag:'袋装物',bottle:'瓶装物',plastic:'塑料',paper:'纸类',metal:'金属',wood:'木质',fabric:'织物',unknown:'未提供'}[value] || value || '未提供');
  const objectName = item => item?'物体 '+String(demo.result.instances.findIndex(i=>i.instance_id===item.instance_id)+1).padStart(2,'0'):'未选择';
  const candidateCount = result => result.instances.filter(i=>i.attributes.material?.values?.top1_label).length;
  const recordName = (job,index=0) => (job.created_at?new Date(job.created_at).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'Asia/Shanghai'}):'演示记录 '+(index+1))+' · '+(job.source==='upload'?'原图分析':'基于已有识别结果');
  const message = text => {for(const id of ['flowMessage','taskResultMessage']){byId(id).textContent=text;byId(id).hidden=!text;}};
  const resultRoute = ()=>/^#result\/(job_[a-zA-Z0-9_-]+)$/.exec(location.hash)?.[1];
  async function readJson(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error('文件读取失败（HTTP '+response.status+'）');
    return response.json();
  }
  function artifact(path) {
    if (!path || typeof path !== 'string') return null;
    if (demo.snapshot) return demo.snapshot.files[path] || null;
    if (!demo.result?.pipeline_result_ref || path.startsWith('/') || path.includes('\\') || path.split('/').includes('..') || path.includes(':')) return null;
    return demo.result.pipeline_result_ref.replace(/result\.json$/, '') + path.split('/').map(encodeURIComponent).join('/');
  }
  function image(parent, url, caption, crop=false) {
    const figure=node('figure',undefined,crop?'flow-figure crop':'flow-figure');
    if (url) {
      const img=node('img');img.src=url;img.alt=caption;
      img.onerror=()=>figure.replaceChildren(node('p','图像不可用：'+caption,'flow-placeholder'));
      figure.append(img);
    } else figure.append(node('p','未提供该图像','flow-placeholder'));
    figure.append(node('figcaption',caption));parent.append(figure);
  }
  function fact(key,value) {
    const row=node('div',undefined,'flow-fact');row.append(node('span',key),node('strong',display(value)));byId('flowFacts').append(row);
  }
  function file(path,title,parent='flowFiles') {
    const url=artifact(path);if (!url) return;
    const link=node('a',title||path,'secondary small');link.href=url;
    if(demo.snapshot||parent==='flowDownloads') link.download=path.split('/').pop();
    else {link.target='_blank';link.rel='noopener';}
    byId(parent).append(link);
  }
  function currentItems() {
    const result=demo.result;
    const item=result.instances.find(i=>i.instance_id===demo.instance);
    const roi=demo.docs['material_input.json']?.items?.find(i=>i.instance_id===demo.instance);
    const material=demo.docs['material_result.json']?.items?.find(i=>i.instance_id===demo.instance);
    const final=demo.docs['result.json']?.instances?.find(i=>i.instance_id===demo.instance);
    return {result,item,roi,material,final};
  }
  function render() {
    if(!demo.result)return;
    if(demo.step===4&&!window.demoPreferences.get().depth)demo.step=0;
    const {result,item,roi,material,final}=currentItems();
    const [key,title]=stages[demo.step];
    const pre=demo.docs['preprocess.json'],seg=demo.docs['segmentation.json'];
    const number=demo.step===5?5:demo.step+1;
    byId('flowStep').textContent=key==='depth'?'附加信息':'步骤 '+number+' / 5';
    byId('flowTitle').textContent=title;byId('flowPosition').textContent=key==='depth'?'深度信息':number+' / 5';
    byId('flowPrev').disabled=demo.step===0;byId('flowNext').disabled=demo.step===5;
    byId('flowInstance').disabled=!result.instances.length;
    byId('flowInstance').previousElementSibling.textContent=resultRoute()?'选择目标，查看输出数据':'选择一个物体，查看完整分析过程';
    byId('flowInstance').parentElement.hidden=key==='input'||key==='depth';
    for(const button of byId('flowRail').children){const selected=button.dataset.key===key;button.classList.toggle('selected',selected);button.setAttribute('aria-current',selected?'step':'false');}
    byId('flowDepth').classList.toggle('selected',key==='depth');
    byId('flowVisual').replaceChildren();byId('flowFacts').replaceChildren();byId('flowFiles').replaceChildren();byId('flowDownloads').replaceChildren();byId('flowTechnical').open=false;
    let narrative,input,output,boundary,data;
    const visual=byId('flowVisual');
    const original=demo.snapshot?artifact('rgb.png'):result.image_ref;
    const overlay=demo.snapshot?artifact('overlay.jpg'):result.overlay_ref;
    const mask=demo.snapshot?artifact(item?.mask_path):item?.mask_ref;
    const secondary=(url,caption)=>{const details=node('details',undefined,'flow-secondary');details.append(node('summary',caption));image(details,url,caption,true);visual.append(details);};
    if(key==='input') {
      narrative='从原始画面开始，查看本次分析使用的图片。';input='原始图片';output='可逐个分析的画面';
      image(visual,original,'本次分析的原图');
      fact('图片尺寸',result.image_size.join(' × ')+' 像素');fact('处理方式',result.execution.new_inference?'从原图重新分析':'基于已有物体识别结果');
      fact('原始图片',original?'已提供':'未提供');fact('深度信息',result.depth_ref?'已保存':result.depth.depth_path?'有引用，未附文件':'未提供');
      boundary=original?'本页回看已完成的真实结果，切换步骤不会重新运行模型。':'本记录未附原始图片，仍可查看已有的物体识别结果。';
      data=pre||{execution:result.execution,image_size:result.image_size,depth:result.depth};file('preprocess.json');file('rgb.png','原始图片');
    } else if(key==='segmentation') {
      narrative='把画面中的物体逐个圈出来，标出各自的位置和轮廓。';input=result.execution.new_inference?'原始画面':'已有识别结果';output=result.instance_count+' 个物体的位置与轮廓';
      image(visual,overlay,'识别结果 · 色块表示物体范围');if(item)secondary(mask,'查看所选物体的轮廓图');
      fact('检出物体',result.instance_count+' 个');fact('当前选择',objectName(item));
      if(item){fact('预测类别',friendly(item.object_label));fact('检测评分',score(item.det_score));}
      fact('结果来源',result.execution.new_inference?'本次重新分析':'复用已有识别结果');
      boundary='轮廓与类别为模型识别结果，可能包含背景误检，需结合画面复核。';data=seg?{...seg,instances:seg.instances.filter(i=>i.instance_id===demo.instance)}:item||{instance_count:0};file('segmentation.json');if(item)file(item.mask_path,'物体轮廓文件');
    } else if(key==='roi') {
      narrative='从原图中取出选定物体的特写，作为材质分析的输入。';input='原图 + 所选物体的位置';output='单个物体的特写图片';
      image(visual,artifact(roi?.crop_path),'所选物体的特写',true);if(roi?.crop_mask_path)secondary(artifact(roi.crop_mask_path),'查看物体轮廓');
      fact('当前选择',objectName(item));fact('特写状态',status(roi?.status));fact('图片尺寸',roi?.crop_size?roi.crop_size.join(' × ')+' 像素':null);fact('背景保留',roi?.background_mode==='original'?'保留原图背景':roi?.background_mode);
      boundary='当前按物体外框截取，特写可能包含少量背景；尚未标定真实接触区域。';data=roi||{status:'unavailable',reason:'本任务没有可读取的特写数据'};file('material_input.json');file(roi?.crop_path,'物体特写');file(roi?.crop_mask_path,'轮廓文件');
    } else if(key==='material') {
      narrative='根据物体特写给出材质预测，保留各类别的模型评分。';input='物体特写';output='材质预测 + 模型评分';
      image(visual,artifact(roi?.crop_path),'用于材质分析的物体特写',true);
      const attr=item?.attributes.material,prediction=material||attr,scores=material?.scores||attr?.values?.scores;
      if(scores&&typeof scores==='object'){
        const chart=node('div',undefined,'flow-scores');chart.append(node('h3','各材质的模型评分'));
        for(const [label,value] of Object.entries(scores).filter(([,v])=>typeof v==='number').sort((a,b)=>b[1]-a[1])){const row=node('div',undefined,'flow-score');const bar=node('progress');bar.max=1;bar.value=value;row.append(node('span',friendly(label)),bar,node('strong',score(value)));chart.append(row);}visual.append(chart);
      }
      fact('当前选择',objectName(item));fact('材质预测',friendly(material?.top1_label||attr?.values?.top1_label));fact('模型评分',score(prediction?.score));
      boundary=prediction?.status==='error'?'这件物体的材质分析未完成，已有识别结果已保留。':prediction?.status==='skipped'||!prediction?'本记录未产生材质预测，可继续查看其他有效结果。':'材质为模型预测，评分反映模型倾向，不等于实际准确率。';data=prediction||{status:'skipped',reason:'material_not_requested'};file('material_result.json');
    } else if(key==='depth') {
      narrative='深度信息与识别结果一起保存，供后续尺寸分析使用。';input='可选的深度数据';output='深度记录与可用状态';
      const card=node('div',undefined,'flow-depth-card');card.append(node('span','附加信息','eyebrow'),node('h3',result.depth_ref?'已保存深度数据':result.depth.depth_path?'已记录深度引用':'未提供深度数据'),node('p','当前展示保存状态'),node('strong','物理尺寸：尚未计算'));visual.append(card);
      fact('深度数据',result.depth_ref?'已保存':result.depth.depth_path?'仅有引用':'未提供');fact('图片尺寸',result.depth.depth_size);fact('深度单位',status(result.depth.depth_unit));fact('图像对齐',status(result.depth.alignment));fact('尺寸计算','尚未处理');
      boundary='深度单位、图像对齐与处理方法确认后，才能用于可靠的尺寸计算。';data=result.depth;file('preprocess.json');if(result.depth_ref)file('depth.png','深度文件');
    } else {
      narrative='这是整条视觉链路的最终输出：每个目标的类别、位置、轮廓与材质预测，供机器臂系统的任务规划模块读取。';input='物体识别 + 材质分析';output='提供给机器臂系统的视觉感知数据（JSON）';
      image(visual,overlay,'完整画面的识别结果');
      const canonical=demo.docs['result.json'];data=canonical?{...canonical,instances:canonical.instances.filter(i=>i.instance_id===demo.instance)}:{status:'unavailable',reason:'旧任务未提供标准结果文件'};
      const list=node('div',undefined,'customer-result-list'),table=node('table'),head=node('tr');for(const text of ['目标 ID','类别','像素框 xyxy','材质预测','评分'])head.append(node('th',text));const thead=node('thead');thead.append(head);table.append(thead);const body=node('tbody');
      for(const current of canonical?.instances||result.instances){const prediction=current.material||current.attributes?.material,row=node('tr');row.append(node('td',current.instance_id),node('td',friendly(current.object_label)),node('td',current.bbox_xyxy.join(', ')),node('td',friendly(prediction?.top1_label||prediction?.values?.top1_label)),node('td',score(prediction?.score)));body.append(row);}table.append(body);list.append(table);visual.append(list);
      fact('输出格式','JSON · '+(canonical?.schema_version||result.schema_version));fact('目标数量',result.instance_count+' 个');
      if(item){const selected=final||item;fact('当前目标 ID',selected.instance_id);fact('目标类别',friendly(selected.object_label));fact('位置（像素）',selected.bbox_xyxy.join(', '));fact('轮廓文件',selected.mask_path);fact('材质预测',friendly(selected.material?.top1_label||item.attributes.material?.values?.top1_label));fact('模型评分',score(selected.material?.score??item.attributes.material?.score));file(selected.mask_path,'当前目标轮廓','flowDownloads');}
      boundary=(result.status==='no_match'?'本次未检出目标。':'')+'坐标为原图像素，格式为左上角 x、y 和右下角 x、y。机械臂坐标、抓取位姿与运动指令由后续标定和规划模块生成。材质为模型预测。';
      file('result.json','下载视觉感知 JSON','flowDownloads');file('report.md','下载报告','flowDownloads');
      if(!demo.snapshot){const link=node('a','下载完整结果包','primary');link.href=result.download_ref;byId('flowDownloads').prepend(link);}else fact('完整结果包','随演示文件一同提供');
      for(const name of filenames)file(name);
    }
    byId('flowNarrative').textContent=narrative;byId('flowInput').textContent=input;byId('flowOutput').textContent=output;byId('flowBoundary').textContent=boundary;
    byId('flowTrace').textContent='任务编号：'+result.run_id+' · 图像：'+result.frame_id+' · 物体编号：'+(demo.instance||'无');
    byId('flowJson').textContent=JSON.stringify(data,(key,value)=>key.toLowerCase().includes('sha256')?undefined:value,2);
  }

  async function use(result,snapshot=null,step=0) {
    const token=++demo.token;demo.result=result;demo.snapshot=snapshot;demo.docs={};demo.errors=[];demo.step=step;demo.instance=result.instances[0]?.instance_id||null;
    byId('flowBody').hidden=true;message('正在读取该任务各阶段的真实文件…');
    const outcomes=await Promise.allSettled(filenames.map(async name=>{
      const url=artifact(name);
      if(!snapshot&&!url)throw new Error('任务未提供标准文件');
      const data=snapshot?snapshot.docs[name]:await readJson(url);
      if(!data)throw new Error('文件不存在');
      if(data.run_id!==result.run_id||data.frame_id!==result.frame_id)throw new Error('任务标识不一致');
      return [name,data];
    }));
    if(token!==demo.token)return;
    outcomes.forEach((outcome,index)=>{if(outcome.status==='fulfilled')demo.docs[outcome.value[0]]=outcome.value[1];else demo.errors.push(filenames[index]);});
    message(demo.errors.length?'部分阶段文件不可用：'+demo.errors.join('、')+'。缺失步骤将明确显示未提供。':'');
    byId('flowBody').hidden=false;
    if(![...byId('flowJob').options].some(option=>option.value===result.run_id))byId('flowJob').append(new Option('当前分析记录',result.run_id));
    byId('flowJob').value=result.run_id;
    byId('flowInstance').replaceChildren(...result.instances.map(item=>new Option(objectName(item)+' · '+friendly(item.object_label),item.instance_id)));
    if(!result.instances.length)byId('flowInstance').append(new Option('无目标实例',''));
    byId('flowMode').textContent=offline?'离线演示 · 真实结果回看':'真实结果回看';
    byId('flowSummary').replaceChildren();
    for(const [label,value] of [['图片尺寸',result.image_size.join(' × ')],['检出物体',result.instance_count+' 个'],['材质候选',candidateCount(result)+' 个']]){
      const card=node('div');card.append(node('small',label),node('strong',value));byId('flowSummary').append(card);
    }
    byId('flowTrace').textContent='run_id: '+result.run_id+'  ·  frame_id: '+result.frame_id+'  ·  当前实例: '+(demo.instance||'无')+(result.execution.source_run_id?'  ·  来源任务: '+result.execution.source_run_id:'');
    render();
  }
  async function refresh(preferredId) {
    byId('flowRefresh').disabled=true;byId('taskRefresh').disabled=true;
    try {
      const page=offline?{jobs:offline.snapshots.map(s=>({id:s.result.run_id,frame_id:s.result.frame_id,status:s.result.status==='partial'?'partial':'completed',source:s.result.execution.new_inference?'upload':'sample',snapshot:s})),total:offline.snapshots.length}:await readJson('/v1/jobs?limit=100&offset=0');
      renderTasks(page);
      const tasks=page.jobs.filter(job=>job.result_ref||job.snapshot);
      byId('flowJob').replaceChildren(...tasks.map((job,index)=>new Option(recordName(job,index),job.id)));
      demo.tasks=tasks;
      window.demoPreferences.records(tasks.map((job,index)=>({id:job.id,label:recordName(job,index)})));
      if(resultRoute()){await openTaskResult(resultRoute());return;}
      if(window.demoUpload?.pending()&&typeof preferredId!=='string')return;
      if(!tasks.length){++demo.token;demo.result=null;byId('flowBody').hidden=true;message(offline?'暂无已保存成果。':'还没有分析结果，上传第一张图片即可开始。');return;}
      const requested=typeof preferredId==='string'?preferredId:demo.result?.run_id||window.demoPreferences.get().record;
      const chosen=tasks.find(t=>t.id===requested)||tasks[0];byId('flowJob').value=chosen.id;
      await choose(chosen.id);
    } catch(error){message('演示数据读取失败：'+error.message);byId('taskMessage').textContent='刷新失败，已有记录可能不是最新状态。请稍后重试。';}finally{byId('flowRefresh').disabled=false;byId('taskRefresh').disabled=false;}
  }
  function renderTasks(page){
    byId('taskScope').textContent=offline?'离线文件保存的任务成果，可打开查看完整流程。':'后台任务的最近记录，点击“刷新状态”获取最新进度。';
    byId('taskMessage').textContent=offline?'共 '+page.jobs.length+' 条已保存记录':'显示最近 '+page.jobs.length+' 条 / 共 '+page.total+' 条 · 更新于 '+new Date().toLocaleTimeString('zh-CN',{hour12:false});
    byId('taskCards').replaceChildren();
    const states={queued:['等待处理','任务已接收，等待开始。'],running:['处理中','正在分析图片，请稍后刷新查看。'],completed:['已完成','分析已结束，可以查看和导出成果。'],partial:['处理已结束','已生成分析结果，可打开本任务的图片与输出。'],failed:['未完成','本次处理未成功，请联系项目人员检查。'],cancelled:['已取消','本次任务已取消。']};
    for(const [index,job] of page.jobs.entries()){
      const [title,description]=states[job.status]||['状态暂不可读','请刷新后再查看。'];
      const card=node('article',undefined,'panel customer-task-card');
      card.append(node('span',title,'badge '+(['partial','queued','running'].includes(job.status)?'warning':job.status==='failed'?'error':'')),node('h2',recordName(job,index)),node('p',description));
      if(job.status==='running'&&job.total>0)card.append(node('p','属性分析：'+job.done+' / '+job.total+' 项','hint'));
      if(job.result_ref||job.snapshot){const link=node('a','查看结果 →','secondary');link.href='#result/'+job.id;card.append(link);}
      byId('taskCards').append(card);
    }
    if(!page.jobs.length)byId('taskCards').append(node('p','暂无任务记录。','hint'));
  }
  async function choose(id) {
    const job=demo.tasks?.find(t=>t.id===id);if(!job)return;
    const selection=++demo.token;byId('flowBody').hidden=true;message('正在打开所选任务…');
    try{const result=job.snapshot?job.snapshot.result:await readJson(job.result_ref);if(selection===demo.token)await use(result,job.snapshot);}
    catch(error){if(selection===demo.token)message('任务读取失败：'+error.message);}
  }
  async function openTaskResult(id){
    const selection=++demo.token;byId('flowBody').hidden=true;byId('taskResultName').textContent='正在读取所选任务…';message('正在读取任务成果…');
    try{
      const job=offline?demo.tasks?.find(t=>t.id===id):await readJson('/v1/jobs/'+encodeURIComponent(id));
      if(!job||(!job.result_ref&&!job.snapshot))throw new Error('该任务没有可查看的成果，请返回任务记录。');
      const result=job.snapshot?job.snapshot.result:await readJson(job.result_ref);
      if(selection!==demo.token||resultRoute()!==id)return;
      byId('taskResultName').textContent=recordName(job,demo.tasks?.findIndex(t=>t.id===id)||0);
      await use(result,job.snapshot,5);
    }catch(error){if(selection===demo.token){byId('taskResultName').textContent='任务成果暂不可用';message(error.message);}}
  }
  for(const [index,[key,title,caption]] of stages.entries()) {
    if(key==='depth')continue;
    const button=node('button',undefined,'flow-node');button.dataset.key=key;
    button.append(node('span',String(index===5?5:index+1).padStart(2,'0'),'flow-node-number'),node('strong',title),node('small',caption));button.onclick=()=>{demo.step=index;render();};byId('flowRail').append(button);
  }
  byId('flowDepth').onclick=()=>{demo.step=4;render();};
  byId('flowPrev').onclick=()=>{demo.step=demo.step===5?3:demo.step===4?0:Math.max(0,demo.step-1);render();};
  byId('flowNext').onclick=()=>{demo.step=demo.step===3||demo.step===4?5:Math.min(stages.length-1,demo.step+1);render();};
  byId('flowInstance').onchange=()=>{demo.instance=byId('flowInstance').value;byId('flowTrace').textContent='run_id: '+demo.result.run_id+'  ·  frame_id: '+demo.result.frame_id+'  ·  当前实例: '+demo.instance;render();};
  byId('flowJob').onchange=()=>choose(byId('flowJob').value);byId('flowRefresh').onclick=refresh;
  byId('taskRefresh').onclick=refresh;
  window.addEventListener('demo-settings-change',render);
  window.addEventListener('hashchange',()=>{if(resultRoute())openTaskResult(resultRoute());else{++demo.token;if(location.hash==='#flow')refresh();}});
  byId('flowPresent').onclick=()=>{const active=document.body.classList.toggle('flow-present');byId('flowPresent').textContent=active?'退出专注演示':'专注演示';byId('flowPresent').setAttribute('aria-pressed',String(active));};
  window.pipelineDemo={use,refresh,clear:()=>{++demo.token;demo.result=null;byId('flowBody').hidden=true;message('本次图片尚未生成结果。');}};
  refresh();
})();
