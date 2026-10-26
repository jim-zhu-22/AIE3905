const test=require('node:test');
const assert=require('node:assert/strict');
const {ImageQueue}=require('../api_service/static/image-queue.js');
const file=(name)=>({name,size:100,lastModified:1});

test('three images are admitted sequentially and each result is delivered',async()=>{
  const order=[];const queue=new ImageQueue({submit:async item=>{order.push('send:'+item.file.name);return{id:item.id,status:'queued'};},
    wait:async job=>{order.push('done:'+queue.items.find(item=>item.id===job.id).file.name);return{...job,status:'completed'};}});
  queue.add([file('1.png'),file('2.png'),file('3.png')]);await queue.start({prompt:'box.'});
  assert.deepEqual(order,['send:1.png','done:1.png','send:2.png','done:2.png','send:3.png','done:3.png']);
  assert.ok(queue.items.every(item=>item.status==='completed'));
});
test('one bad image does not prevent subsequent images from being submitted',async()=>{
  const queue=new ImageQueue({submit:async item=>{if(item.file.name==='bad.png')throw Object.assign(new Error('bad image'),{status:400});return{id:item.id,status:'completed'};},wait:async job=>job});
  queue.add([file('bad.png'),file('good.png')]);await queue.start({prompt:'box.'});
  assert.deepEqual(queue.items.map(item=>item.status),['failed','completed']);
});
test('ambiguous submission keeps the original key and pauses until deliberate resume',async()=>{
  const keys=[];let failed=false;
  const queue=new ImageQueue({submit:async item=>{keys.push(item.key);if(!failed){failed=true;throw new Error('connection lost');}return{id:'accepted',status:'completed'};},wait:async job=>job});
  queue.add([file('one.png')]);await queue.start({prompt:'original'});assert.equal(queue.enabled,false);
  await queue.start({prompt:'edited later'});assert.equal(keys[0],keys[1]);assert.equal(queue.items[0].options.prompt,'original');
  assert.equal(queue.items[0].status,'completed');
});
test('pausing does not cancel the active image and resumes from the next image',async()=>{
  let release;const barrier=new Promise(resolve=>release=resolve);const sent=[];
  const queue=new ImageQueue({submit:async item=>{sent.push(item.file.name);return{id:item.id,status:'running'};},wait:async job=>{await barrier;return{...job,status:'completed'};}});
  queue.add([file('one.png'),file('two.png')]);const active=queue.start({prompt:'box.'});await Promise.resolve();
  queue.pause();release();await active;assert.deepEqual(sent,['one.png']);assert.equal(queue.items[1].status,'waiting');
  await queue.start({prompt:'box.'});assert.deepEqual(sent,['one.png','two.png']);
});
test('continuously arriving files are appended while an earlier image is processing',async()=>{
  let release;const barrier=new Promise(resolve=>release=resolve);let first=true;
  const queue=new ImageQueue({submit:async item=>({id:item.id,status:'running'}),wait:async job=>{if(first){first=false;await barrier;}return{...job,status:'completed'};}});
  queue.add([file('one.png')]);const active=queue.start({prompt:'box.'});await Promise.resolve();queue.add([file('two.png')]);release();await active;
  assert.ok(queue.items.every(item=>item.status==='completed'));
});
test('model not ready and backpressure pause instead of silently dropping remaining images',async()=>{
  for(const status of [429,503]){
    const queue=new ImageQueue({submit:async()=>{throw Object.assign(new Error('unavailable'),{status});},wait:async job=>job});
    queue.add([file('one.png'),file('two.png')]);await queue.start({prompt:'box.'});
    assert.equal(queue.enabled,false);assert.deepEqual(queue.items.map(item=>item.status),['waiting','waiting']);
  }
});
test('removing an unsent file does not cancel submitted tasks',async()=>{
  const queue=new ImageQueue({submit:async item=>({id:item.id,status:'completed'}),wait:async job=>job});
  queue.add([file('remove.png'),file('keep.png')]);queue.remove(queue.items[0].id);
  assert.equal(queue.items.length,1);await queue.start({prompt:'box.'});
  queue.remove(queue.items[0].id);assert.equal(queue.items.length,1);assert.equal(queue.items[0].status,'completed');
});
