const {test}=require('node:test');
const assert=require('node:assert/strict');
const {validateAnnotationImport}=require('../scripts/portable_annotation.js');
const frames=[{id:'frame',width:1280,height:960,group:'time_01'}];
const fixture=()=>({schema_version:2,annotator:'test',objects:[{object_id:'object_0001',frame_id:'frame',image_size:[1280,960],object_bbox_xyxy:[20,30,100,200],contact_bbox_xyxy:null,coordinate_convention:'original_image_pixels_xyxy_half_open',material:'paper',sku_id:'',annotation_status:'needs_review'}]});
test('round trip preserves labels, original coordinates and optional contact',()=>{const d=fixture();const r=validateAnnotationImport(d,frames);assert.deepEqual(r.objects[0].object_bbox_xyxy,[20,30,100,200]);assert.equal(r.objects[0].contact_bbox_xyxy,null);assert.equal(r.objects[0].material,'paper');assert.equal(r.annotator,'test');});
test('multiple objects, including unfinished material, remain separate',()=>{const d=fixture();d.objects.push({...d.objects[0],object_id:'object_0002',material:''});assert.equal(validateAnnotationImport(d,frames).objects.length,2);});
test('wrong dataset frame is rejected',()=>{const d=fixture();d.objects[0].frame_id='other';assert.throws(()=>validateAnnotationImport(d,frames));});
test('duplicate IDs are rejected',()=>{const d=fixture();d.objects.push({...d.objects[0]});assert.throws(()=>validateAnnotationImport(d,frames));});
test('out of bounds, reversed and noninteger coordinates are rejected',()=>{for(const b of [[0,0,1281,10],[100,30,20,200],[0.5,0,20,20]]){const d=fixture();d.objects[0].object_bbox_xyxy=b;assert.throws(()=>validateAnnotationImport(d,frames));}});
test('contact must be inside the object',()=>{const d=fixture();d.objects[0].contact_bbox_xyxy=[0,0,10,10];assert.throws(()=>validateAnnotationImport(d,frames));d.objects[0].contact_bbox_xyxy=[30,40,50,60];assert.deepEqual(validateAnnotationImport(d,frames).objects[0].contact_bbox_xyxy,[30,40,50,60]);});
test('unsupported schemas and labels are rejected',()=>{const d=fixture();d.schema_version=1;assert.throws(()=>validateAnnotationImport(d,frames));d.schema_version=2;d.objects[0].material='invented';assert.throws(()=>validateAnnotationImport(d,frames));});
test('untrusted reviewed status cannot bypass review',()=>{const d=fixture();d.objects[0].annotation_status='approved';assert.equal(validateAnnotationImport(d,frames).objects[0].annotation_status,'needs_review');});

// Exercise the real page handlers with synthetic data; no browser automation.
const vm=require('node:vm'), fs=require('node:fs'), path=require('node:path');
function pageHarness(){
  const nodes=new Map();
  const context2d={clearRect(){},drawImage(){},strokeRect(){},fillRect(){},fillText(){},measureText(){return {width:20};},createImageData(w,h){return {data:new Uint8ClampedArray(w*h*4)};},putImageData(){}};
  const element=()=>({value:'',disabled:false,textContent:'',style:{},dataset:{},files:[],classList:{toggle(){}},append(){},replaceChildren(){},insertBefore(){},focus(){},click(){},getContext(){return context2d;},getBoundingClientRect(){return {left:0,top:0,width:1280,height:960};},setPointerCapture(){}});
  const get=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
  const buttons=['paper','plastic','metal','wood','fabric','unknown'].map(material=>Object.assign(element(),{dataset:{material}}));
  get('materials').querySelectorAll=()=>buttons;
  let exportedBlob, downloadedName;
  const storage=new Map();
  const sandbox={document:{getElementById:get,querySelector:()=>element(),createElement:tag=>{const el=element();if(tag==='a')el.click=()=>{downloadedName=el.download;};return el;}},location:{pathname:'/annotate.html'},localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},Image:class {set src(value){this.onload();}},URL:{createObjectURL(blob){exportedBlob=blob;return 'blob:test';},revokeObjectURL(){}},Blob,setTimeout:fn=>fn(),console,confirm:()=>true};
  sandbox.window=sandbox;
  vm.createContext(sandbox);
  const template=fs.readFileSync(path.join(__dirname,'../scripts/rgbd_viewer.html'),'utf8');
  const script=template.match(/<script>([\s\S]*?)<\/script>/)[1].replace('__FRAMES__',JSON.stringify(frames));
  vm.runInContext(script,sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../scripts/portable_annotation.js'),'utf8'),sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../scripts/annotation_masks.js'),'utf8'),sandbox);
  return {get,buttons,sandbox,state:()=>JSON.parse(vm.runInContext('JSON.stringify(records)',sandbox)),blob:()=>exportedBlob,download:()=>downloadedName};
}
test('page draws two objects and exports separate materials with annotator',async()=>{
  const h=pageHarness(),canvas=h.get('rgb');
  for(const [n,material] of [[0,'paper'],[1,'plastic']]){
    canvas.onpointerdown({button:0,clientX:20+n*200,clientY:30,pointerId:1});
    canvas.onpointerup({clientX:100+n*200,clientY:200});
    h.buttons.find(b=>b.dataset.material===material).onclick();
  }
  h.get('export').onclick();assert.equal(h.blob(),undefined,'name is required');
  h.get('annotatorName').value='测试同学';h.get('export').onclick();
  const d=JSON.parse(await h.blob().text());
  assert.equal(d.annotator,'测试同学');assert.deepEqual(d.objects.map(r=>r.material),['paper','plastic']);
  assert.match(h.download(),/^yingjie_测试同学_/);assert.equal(d.objects[1].contact_bbox_xyxy,null);
});

const codec=require('../scripts/annotation_masks.js');
test('mask codec preserves holes, validates sizes and interpolates strokes',()=>{
  const pixels=new Uint8Array(400);codec.stampMask(pixels,20,20,[2,10],[18,10],2,1);codec.stampMask(pixels,20,20,[10,10],[10,10],1,0);
  assert.deepEqual(codec.decodeMask(codec.encodeMask(pixels,20,20),20,20),pixels);
  assert.equal(pixels[10*20+10],0);assert.equal(pixels[10*20+6],1);
  assert.throws(()=>codec.decodeMask({encoding:'rle-row-major-v1',size:[20,20],counts:[399]},20,20));
});
test('automatic masks survive export/import; edits reset reviews and undo restores',async()=>{
  const h=pageHarness();const d=fixture(),r=d.objects[0],pixels=new Uint8Array(1280*960);
  for(let y=30;y<200;y++)pixels.fill(1,y*1280+20,y*1280+100);
  r.mask_rle=codec.encodeMask(pixels,1280,960);r.material='';r.auto_source_id='model:1';h.sandbox.AUTO_ANNOTATIONS=d;
  h.get('loadAuto').onclick();h.get('loadAuto').onclick();assert.equal(h.state().length,1);
  h.get('annotatorName').value='reviewer';h.buttons[0].onclick();h.get('confirmMaterial').onclick();h.get('confirmMask').onclick();
  assert.equal(h.state()[0].mask_review_status,'confirmed');h.get('export').onclick();
  const exported=JSON.parse(await h.blob().text()),restored=validateAnnotationImport(exported,frames).objects[0];
  assert.deepEqual(restored.mask_rle,r.mask_rle);assert.equal(restored.material_review_status,'confirmed');
  h.get('maskRadius').value=5;h.get('maskErase').onclick();
  h.get('rgb').onpointerdown({button:0,clientX:60,clientY:70,pointerId:1});h.get('rgb').onpointerup({clientX:60,clientY:70});
  assert.equal(h.state()[0].mask_review_status,'needs_review');assert.equal(codec.decodeMask(h.state()[0].mask_rle,1280,960)[70*1280+60],0);
  h.get('undo').onclick();assert.equal(h.state()[0].mask_review_status,'confirmed');
  h.get('redraw').onclick();h.get('rgb').onpointerdown({button:0,clientX:10,clientY:10,pointerId:1});h.get('rgb').onpointerup({clientX:200,clientY:210});
  assert.equal(h.state()[0].mask_rle,null);assert.equal(h.state()[0].material_review_status,'needs_review');
});
test('page restores exported draft and rejects malformed file without replacing work',async()=>{
  const h=pageHarness();
  const importData=async d=>h.get('importFile').onchange({target:{files:[{size:100,text:async()=>JSON.stringify(d)}],value:'test'}});
  await importData(fixture());assert.equal(h.state().length,1);assert.equal(h.get('annotatorName').value,'test');
  const malformed=fixture();malformed.objects[0].object_bbox_xyxy=[0,0,99999,20];
  await importData(malformed);assert.deepEqual(h.state()[0].object_bbox_xyxy,[20,30,100,200]);assert.match(h.get('message').textContent,/导入失败/);
  const wrong=fixture();wrong.dataset_id='other';await importData(wrong);assert.equal(h.state().length,1);
  h.sandbox.confirm=()=>false;await importData({schema_version:2,objects:[]});assert.equal(h.state().length,1,'cancel preserves existing records');
});
