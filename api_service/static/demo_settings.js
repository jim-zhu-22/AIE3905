/* Browser-scoped presentation preferences. Never modifies model configuration. */
(()=>{
  'use strict';
  const id=name=>document.getElementById(name);
  const key='aie3905.presentation.v1';
  const defaults={title:'从一张图片，到清晰的识别结果',record:'',depth:true,technical:false};
  const normalize=value=>({
    title:typeof value?.title==='string'&&value.title.trim()?value.title.trim().slice(0,60):defaults.title,
    record:typeof value?.record==='string'&&/^job_[a-zA-Z0-9_-]+$/.test(value.record)?value.record:'',
    depth:typeof value?.depth==='boolean'?value.depth:true,
    technical:typeof value?.technical==='boolean'?value.technical:false
  });
  let settings={...defaults};
  try{settings=normalize(JSON.parse(localStorage.getItem(key)));}
  catch{ id('settingsMessage').textContent='未能读取已保存设置，当前使用默认展示。'; }
  function apply(){
    id('demoHeading').textContent=settings.title;
    document.title=settings.title+' · AIE';
    document.body.classList.toggle('hide-demo-depth',!settings.depth);
    id('flowTechnical').hidden=!settings.technical;
    window.dispatchEvent(new Event('demo-settings-change'));
  }
  function fill(){
    id('settingTitle').value=settings.title;
    id('settingRecord').value=settings.record;
    id('settingDepth').checked=settings.depth;
    id('settingTechnical').checked=settings.technical;
  }
  function save(next){
    try{localStorage.setItem(key,JSON.stringify(next));}
    catch{id('settingsMessage').textContent='保存失败：浏览器未允许本地存储。设置未更改。';return;}
    settings=next;apply();fill();id('settingsMessage').textContent='已保存。展示内容已生效，默认记录将在下次打开时使用。';
  }
  id('demoSettings').onsubmit=event=>{event.preventDefault();save(normalize({title:id('settingTitle').value,record:id('settingRecord').value,depth:id('settingDepth').checked,technical:id('settingTechnical').checked}));};
  id('settingsReset').onclick=()=>save({...defaults});
  window.demoPreferences={get:()=>({...settings}),records:jobs=>{
    const selected=id('settingRecord').value||settings.record;
    id('settingRecord').replaceChildren(new Option('最新可查看记录',''),...jobs.map(job=>new Option(job.label,job.id)));
    if(selected&&!jobs.some(job=>job.id===selected))id('settingRecord').append(new Option('原默认记录不可用，将使用最新记录',selected));
    id('settingRecord').value=selected;
  }};
  fill();apply();
})();
