/* Sequential admission: finish one image before submitting the next. */
(function(root){
  'use strict';
  const terminal=new Set(['completed','partial','failed','cancelled']);
  const newKey=()=>globalThis.crypto?.randomUUID?.()||('image_'+Date.now()+'_'+Math.random().toString(36).slice(2));
  class ImageQueue {
    constructor({submit,wait,onChange=()=>{},onJob=()=>{},maxItems=100}){
      this.submit=submit;this.wait=wait;this.onChange=onChange;this.onJob=onJob;
      this.maxItems=maxItems;this.items=[];this.enabled=false;this.busy=false;this.options=null;
    }
    changed(){this.onChange(this);}
    add(files){
      let added=0;
      for(const file of files){
        if(this.items.length>=this.maxItems)throw new Error('本次列表最多保留 '+this.maxItems+' 张图片，请先清理已完成项。');
        const signature=[file.name,file.size,file.lastModified].join('|');
        if(this.items.some(item=>item.signature===signature&&!terminal.has(item.status)))continue;
        const item={id:newKey(),key:newKey(),file,signature,status:'waiting',job:null,error:null,
          options:this.enabled?structuredClone(this.options):null};
        if(file.size>15*1024*1024){item.status='failed';item.error='图片超过 15 MiB。';}
        else if(!/\.(png|jpe?g)$/i.test(file.name)){item.status='failed';item.error='请选择 PNG 或 JPEG 图片。';}
        this.items.push(item);added++;
      }
      this.changed();if(this.enabled)this.pump();return added;
    }
    start(options){
      this.options=structuredClone(options);this.enabled=true;
      for(const item of this.items)if(!item.options&&!terminal.has(item.status))item.options=structuredClone(options);
      this.changed();return this.pump();
    }
    pause(){this.enabled=false;this.changed();}
    clearFinished(){this.items=this.items.filter(item=>!terminal.has(item.status));this.changed();}
    remove(id){this.items=this.items.filter(item=>item.id!==id||item.status!=='waiting'||item.job);this.changed();}
    retry(id){const item=this.items.find(item=>item.id===id);if(!item||!terminal.has(item.status))return;
      item.status='waiting';item.error=null;item.job=null;item.key=newKey();
      if(this.enabled&&!item.options)item.options=structuredClone(this.options);this.changed();if(this.enabled)this.pump();}
    update(job){const item=this.items.find(item=>item.job?.id===job.id);if(!item)return;
      item.job=job;item.status=job.status;item.error=job.error;this.changed();}
    async pump(){
      if(this.busy)return;this.busy=true;
      try{
        while(this.enabled){
          const item=this.items.find(item=>item.status==='waiting');if(!item)break;
          item.options??=structuredClone(this.options);item.error=null;this.changed();
          try{
            if(!item.job){item.status='uploading';this.changed();item.job=await this.submit(item);}
            item.status=item.job.status;this.onJob(item.job);this.changed();
            const result=terminal.has(item.job.status)?item.job:await this.wait(item.job);
            item.job=result;item.status=result.status;item.error=result.error;this.onJob(result);this.changed();
          }catch(error){
            item.error=error.message;
            if(!error.status||error.status===429||error.status===503||error.status>=500){
              // A lost response is ambiguous. Keep the key/job to make a deliberate retry safe.
              item.status='waiting';this.enabled=false;
            }else{item.status='failed';}
            this.changed();
          }
        }
      }finally{this.busy=false;this.changed();}
    }
  }
  if(typeof module!=='undefined'&&module.exports)module.exports={ImageQueue};
  else root.ImageQueue=ImageQueue;
})(globalThis);
