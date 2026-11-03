/* Offline handoff support. Uses the existing viewer's schema and coordinates. */
function validateAnnotationImport(data, availableFrames) {
  if (!data || data.schema_version !== 2 || !Array.isArray(data.objects)) throw new Error('请选择本工具导出的版本 2 JSON 标注文件。');
  const frameMap = new Map(availableFrames.map(f => [f.id, f]));
  const seen = new Set();
  const materials = ['', 'paper', 'plastic', 'metal', 'wood', 'fabric', 'unknown'];
  function validBox(b, f) {
    return Array.isArray(b) && b.length === 4 && b.every(Number.isInteger) && b[0] >= 0 && b[1] >= 0 && b[2] <= f.width && b[3] <= f.height && b[2] > b[0] && b[3] > b[1];
  }
  const clean = data.objects.map((r, i) => {
    const f = r && frameMap.get(r.frame_id);
    const fail = () => { throw new Error('第 ' + (i + 1) + ' 条标注的图片、编号、材质或坐标不符合当前数据集，未导入任何记录。'); };
    if (!f || !/^object_\d{1,9}$/.test(r.object_id) || seen.has(r.object_id) || !materials.includes(r.material) || !validBox(r.object_bbox_xyxy, f)) fail();
    if (!Array.isArray(r.image_size) || r.image_size[0] !== f.width || r.image_size[1] !== f.height || r.coordinate_convention !== 'original_image_pixels_xyxy_half_open') fail();
    if (r.contact_bbox_xyxy != null) {
      const c = r.contact_bbox_xyxy, b = r.object_bbox_xyxy;
      if (!validBox(c, f) || c[0] < b[0] || c[1] < b[1] || c[2] > b[2] || c[3] > b[3]) fail();
    }
    seen.add(r.object_id);
    if(r.mask_rle){
      const codec=typeof module!=='undefined'?require('./annotation_masks.js'):{decodeMask,maskBounds};
      const bounds=codec.maskBounds(codec.decodeMask(r.mask_rle,f.width,f.height),f.width,f.height);
      if(!bounds || bounds.some((v,j)=>v!==r.object_bbox_xyxy[j])) fail();
    }
    const extra={mask_rle:r.mask_rle||null,auto_source_id:typeof r.auto_source_id==='string'?r.auto_source_id.slice(0,300):'',sequence_id:typeof r.sequence_id==='string'?r.sequence_id.slice(0,200):'',model_material:typeof r.model_material==='string'?r.model_material:'',model_confidence:Number.isFinite(r.model_confidence)?r.model_confidence:null,model_scores:r.model_scores&&typeof r.model_scores==='object'?r.model_scores:{},material_model_version:typeof r.material_model_version==='string'?r.material_model_version:''};
    for(const kind of ['mask','material']){
      const valid=r[kind+'_review_status']==='confirmed' && typeof r[kind+'_reviewed_by']==='string' && r[kind+'_reviewed_by'].trim() && typeof r[kind+'_reviewed_at']==='string' && Number.isFinite(Date.parse(r[kind+'_reviewed_at'])) && (kind==='mask'?!!r.mask_rle:!!r.material);
      extra[kind+'_review_status']=valid?'confirmed':'needs_review';extra[kind+'_reviewed_by']=valid?r[kind+'_reviewed_by'].slice(0,80):'';extra[kind+'_reviewed_at']=valid?r[kind+'_reviewed_at']:'';
    }
    return {object_id:r.object_id, frame_id:f.id, rgb_file:f.id+'_color.jpg', depth_file:f.id+'_depth.png', image_size:[f.width,f.height], time_group:f.group, object_bbox_xyxy:[...r.object_bbox_xyxy], contact_bbox_xyxy:r.contact_bbox_xyxy == null ? null : [...r.contact_bbox_xyxy], coordinate_convention:'original_image_pixels_xyxy_half_open', material:r.material, sku_id:typeof r.sku_id === 'string' ? r.sku_id.slice(0,200) : '', annotation_status:extra.material_review_status==='confirmed'?'confirmed':'needs_review',...extra};
  });
  return {objects:clean, annotator:typeof data.annotator === 'string' ? data.annotator.slice(0,80) : ''};
}
if (typeof module !== 'undefined' && module.exports) module.exports = {validateAnnotationImport};
if (typeof document !== 'undefined') (() => {
  const bar = document.createElement('div');
  bar.className = 'bar';
  bar.innerHTML = '<label>标注人 <input id="annotatorName" type="text" maxlength="80" placeholder="填写姓名或组内昵称" style="width:210px"></label><button id="importDraft">导入历史标注，继续工作</button><input id="importFile" type="file" accept=".json,application/json" hidden><a href="使用说明.html" target="_blank">使用说明</a><span class="muted">离线标注版 · 每次结束前导出 JSON，发回项目负责人</span>';
  document.querySelector('main').insertBefore(bar, document.querySelector('main > .bar'));
  const nameInput = $('annotatorName');
  const nameKey = key + ':annotator';
  try { nameInput.value = localStorage.getItem(nameKey) || ''; } catch (_) {}
  let lastExportSignature = '';
  const signature = () => JSON.stringify({annotator:nameInput.value.trim(), records});
  nameInput.oninput = () => {try { localStorage.setItem(nameKey, nameInput.value.trim()); } catch (_) {}};
  const basePersist = persist;
  persist = function() {basePersist(); if(storageOkay) $('storage').textContent = '浏览器草稿仅供临时恢复；移动文件夹、换浏览器或清理缓存后可能失效。请导出 JSON 备份。';};
  persist();
  $('export').onclick = () => {
    const annotator = nameInput.value.trim();
    if (!annotator) {message('请先填写上方的标注人姓名或昵称，再导出。'); nameInput.focus(); return;}
    const exportedAt = new Date().toISOString();
    const data = {schema_version:2, dataset_id:'yingjie_rgbd_20260826_245', annotator, exported_at:exportedAt, depth_unit:'unconfirmed', alignment:'unconfirmed', frame_reviews:window.getAnnotationFrameReviews?.()||[], objects:records};
    const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)], {type:'application/json'}));
    const a = document.createElement('a'); a.href=url;
    a.download = 'yingjie_'+annotator.replace(/[\\/:*?"<>|\s]/g,'_')+'_'+exportedAt.replace(/[:.]/g,'-')+'.json';
    a.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
    lastExportSignature = signature();
    const pending = records.filter(r => !r.material).length;
    message('已发起下载，共 '+records.length+' 个商品'+(pending?'，其中 '+pending+' 个还没选材质':'')+'。请在浏览器下载列表确认文件已保存。');
  };
  $('importDraft').onclick = () => $('importFile').click();
  window.loadAnnotationData = raw => {
      if(raw.dataset_id && raw.dataset_id !== 'yingjie_rgbd_20260826_245') throw new Error('该文件属于其他数据集，未导入。');
      const data=validateAnnotationImport(raw,frames);
      if(records.length && !confirm('导入会替换当前全部 '+records.length+' 条草稿。请先导出当前工作。确定继续吗？')) return;
      records=data.objects; nextId=Math.max(1,...records.map(r => Number(r.object_id.split('_').pop())+1));
      history=[]; selectedId=null; mode='new';
      if(data.annotator) {nameInput.value=data.annotator; nameInput.oninput();}
      if(records.length) idx=frames.findIndex(f => f.id===records[records.length-1].frame_id);
      window.setAnnotationFrameReviews?.(raw.frame_reviews||[]);
      load(); lastExportSignature=signature();
      message('已导入 '+records.length+' 个商品。模型轮廓仍需人工复核；请确认标注人。');
  };
  $('importFile').onchange = async e => {
    const file=e.target.files[0]; if(!file) return;
    try {
      if (file.size > 20*1024*1024) throw new Error('文件过大，请选择本工具导出的 JSON 标注文件。');
      const raw=JSON.parse(await file.text());
      window.loadAnnotationData(raw);
    } catch(error) {message('导入失败：'+error.message);}
    finally {e.target.value='';}
  };
  window.onbeforeunload = e => {if(records.length && signature() !== lastExportSignature) {e.preventDefault(); e.returnValue='请先导出标注文件。';}};
})();
