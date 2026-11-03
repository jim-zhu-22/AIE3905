/* Starts real local inference and follows the returned durable job. */
(()=>{
  'use strict';
  const id=name=>document.getElementById(name),key='aie3905.active-upload.v1';
  if(window.PIPELINE_DEMO_DATA){id('uploadPanel').hidden=true;id('uploadOffline').hidden=false;return;}
  let ready=false,modules=[],busy=false,jobId=null,timer=null,preview=null,pending=false;
  const note=text=>{id('uploadMessage').textContent=text;id('uploadMessage').hidden=!text;};
  const remember=value=>{try{if(value)sessionStorage.setItem(key,value);else sessionStorage.removeItem(key);}catch{/* Task is still available in durable history. */}};
  async function api(url,options){
    const response=await fetch(url,options);let data;
    try{data=await response.json();}catch{throw new Error('服务响应无法读取，请检查连接。');}
    if(!response.ok){const error=new Error(data.error?.message||'请求未成功，请检查图片或稍后重试。');error.rejected=true;error.code=data.error?.code;throw error;}
    return data;
  }
  function lock(value){
    busy=value;
    for(const name of ['uploadRgb','uploadDepth','clearDepth'])id(name).disabled=value;
    id('uploadStart').disabled=value||!ready;
    id('uploadStart').textContent=value?'正在分析…':'开始分析 →';
  }
  async function connect(){
    id('uploadConnect').disabled=true;
    try{
      const data=await api('/v1/modules');
      ready=data.workspace_available&&data.modules.some(m=>m.id==='segmentation'&&m.state==='ready');
      modules=data.modules.filter(m=>m.id!=='segmentation'&&m.state==='ready').map(m=>m.id);
      const titles=data.modules.filter(m=>modules.includes(m.id)).map(m=>m.title);
      id('uploadService').textContent=ready?'服务已就绪 · 物体识别'+(titles.length?' + '+titles.join(' + '):'；材质分析尚未启用'):'分析服务尚未就绪，请联系项目人员启动模型；已有结果仍可查看。';
      if(ready&&!modules.includes('material')&&titles.length)id('uploadService').textContent+='（材质分析尚未启用）';
    }catch{ready=false;id('uploadService').textContent='未连接到分析服务，请确认本机服务已启动。';}
    finally{id('uploadConnect').hidden=ready;id('uploadConnect').disabled=false;lock(busy);}
  }
  function progress(title,text,working){
    id('uploadProgress').hidden=false;id('uploadProgressTitle').textContent=title;id('uploadProgressText').textContent=text;id('uploadBusy').hidden=!working;
  }
  function paused(error){
    progress('进度查询暂时中断','任务可能仍在后台处理。点击“继续查询”恢复查看，不会重复创建任务。',false);
    id('uploadResume').hidden=false;note(error.message);lock(true);
  }
  async function follow(){
    if(!jobId)return;
    clearTimeout(timer);id('uploadResume').hidden=true;
    try{
      const job=await api('/v1/jobs/'+encodeURIComponent(jobId));
      if(job.result_ref){
        progress('分析已结束，正在读取成果','即将展示本次图片的分析结果。',true);
        const result=await api(job.result_ref);
        if(!location.hash.startsWith('#result/'))await window.pipelineDemo.use(result);
        pending=false;
        await window.pipelineDemo.refresh(job.id);
        progress('分析已结束','本次检出 '+result.instance_count+' 个物体。下方可逐步查看，并在最后汇总导出。',false);
        remember(null);jobId=null;lock(false);note('');
        id('uploadPanel').open=false;id('uploadSummary').textContent='继续上传下一张图片';return;
      }
      if(['failed','cancelled'].includes(job.status)){
        progress(job.status==='failed'?'本次分析未完成':'任务已取消','请在任务记录中查看本次任务；可重新选择图片后提交。',false);
        remember(null);jobId=null;lock(false);note('');return;
      }
      const stage=job.status==='queued'?'等待处理':job.stage==='目标检测与分割'?'识别物体':(job.stage||'分析图片').split(' · ')[0];
      progress(stage,job.total>0?'属性分析进度：'+job.done+' / '+job.total+' 项。完成后自动展示结果。':'正在处理本次图片，完成后自动展示结果。',true);
      timer=setTimeout(follow,900);
    }catch(error){
      if(error.code==='job_not_found'){
        remember(null);jobId=null;pending=false;lock(false);id('uploadResume').hidden=true;
        progress('原任务已不可用','请重新选择图片开始分析，或查看任务记录中的其他成果。',false);
      }else paused(error);
    }
  }
  id('uploadRgb').onchange=()=>{
    id('uploadDepth').value='';
    id('uploadPlaceholder').textContent='选择图片后在这里预览';
    if(preview)URL.revokeObjectURL(preview);preview=null;
    id('uploadPreview').hidden=true;id('uploadPreview').removeAttribute('src');id('uploadPlaceholder').hidden=false;note('');
    const file=id('uploadRgb').files[0];if(!file)return;
    if(file.size>15*1024*1024){id('uploadRgb').value='';note('图片不能超过 15 MiB。');return;}
    preview=URL.createObjectURL(file);id('uploadPreview').src=preview;id('uploadPreview').hidden=false;id('uploadPlaceholder').hidden=true;
    pending=true;window.pipelineDemo.clear();id('uploadProgress').hidden=true;id('uploadSummary').textContent='图片已选好，开始分析';
  };
  id('uploadPreview').onerror=()=>{id('uploadPreview').hidden=true;id('uploadPlaceholder').hidden=false;id('uploadPlaceholder').textContent='图片无法预览，请换一张 PNG 或 JPG。';id('uploadRgb').value='';note('所选文件无法作为图片读取。');};
  id('clearDepth').onclick=()=>{id('uploadDepth').value='';};
  id('uploadForm').onsubmit=async event=>{
    event.preventDefault();if(busy)return;
    const rgb=id('uploadRgb').files[0],depth=id('uploadDepth').files[0];
    if(!rgb){note('请先选择一张现场图片。');return;}
    if(depth?.size>15*1024*1024||rgb.size+(depth?.size||0)>22*1024*1024-65536){note('图片总大小过大，请使用较小的文件；每张最大 15 MiB，请求总大小需小于 22 MiB。');return;}
    lock(true);note('');await connect();
    if(!ready){lock(false);return;}
    pending=true;window.pipelineDemo.clear();
    const body=new FormData();body.append('rgb',rgb);if(depth)body.append('depth',depth);
    body.append('prompt','box. bag. bottle. carton. packet.');body.append('modules',JSON.stringify(modules));
    body.append('frame_id','upload_'+Date.now());
    progress('正在提交图片','正在创建本次分析任务。',true);
    try{
      const job=await api('/v1/perceive',{method:'POST',body});jobId=job.id;remember(jobId);await follow();
    }catch(error){
      lock(false);progress(error.rejected?'图片未提交成功':'提交响应中断',error.rejected?error.message:'连接中断，任务可能已创建。请先查看任务记录，确认后再提交，避免重复分析。',false);
    }
  };
  id('uploadConnect').onclick=connect;
  id('uploadResume').onclick=()=>{note('');follow();};
  window.demoUpload={pending:()=>pending};
  try{const saved=sessionStorage.getItem(key);if(/^job_[a-zA-Z0-9_-]+$/.test(saved||'')){jobId=saved;pending=true;lock(true);window.pipelineDemo.clear();follow();}}catch{/* History remains available. */}
  connect();
})();
