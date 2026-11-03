/* Customer navigation: upload flow and a separate, focused task result view. */
(()=>{
  const show=()=>{
    const requested=location.hash.slice(1);
    const view=/^result\/job_[a-zA-Z0-9_-]+$/.test(requested)?'result':['flow','tasks','settings','guide'].includes(requested)?requested:'flow';
    if(view!=='result'&&location.hash!=='#'+view)history.replaceState(null,'','#'+view);
    const host=document.getElementById(view==='result'?'taskResultHost':'flowResultHost');
    host.append(document.getElementById('flowBody'));
    document.querySelectorAll('.view').forEach(section=>section.hidden=section.id!==view);
    document.querySelectorAll('[data-view]').forEach(button=>{
      const selected=button.dataset.view===(view==='result'?'tasks':view);
      button.classList.toggle('selected',selected);
      button.setAttribute('aria-current',selected?'page':'false');
    });
    if(view!=='flow'){
      document.body.classList.remove('flow-present');
      const present=document.getElementById('flowPresent');present.textContent='专注演示';present.setAttribute('aria-pressed','false');
    }
  };
  document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{location.hash=button.dataset.view;});
  window.addEventListener('hashchange',show);show();
})();
