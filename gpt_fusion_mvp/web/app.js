const $ = id => document.getElementById(id);
let state, selectedCase, uploads = null, currentRun = null, pending = null, compareMode = 'output';
let libraryPage=0;
const pageSize=18;
const statusLabels = {succeeded:'已完成', failed:'失败', running:'生成中', dry_run:'请求检查通过'};
const note = (text, error = false) => { $('feedback').textContent = text; $('feedback').classList.toggle('error', error); };
const src = p => '/library/' + p.split('/').map(encodeURIComponent).join('/');
const runSrc = (r, file) => '/runs/' + encodeURIComponent(r.id) + '/' + encodeURIComponent(file);
function showShare(info){
  if(!info)return;
  state.share=info;$('share-notice').hidden=false;
  $('share-notice').textContent=`同学试用 · 共享生图剩余 ${info.remaining} / ${info.max_generations} 次 · 开放至 ${new Date(info.expires_at*1000).toLocaleString('zh-CN')}。每次只处理一个任务；你新提交的任务只在当前浏览器会话中显示。`;
  $('connection').textContent='曦源 · 同学试用';
}
function selectCase(c) {
  selectedCase=c; uploads=null;
  $('case-title').textContent=c.title; $('case-kind').textContent=c.kind==='controlled_crop'?'受控裁剪 · 有参考图':'真实拍摄 · 无完整真值';
  $('left-image').src=src(c.preview_left||c.left); $('right-image').src=src(c.preview_right||c.right);
  $('prompt').value=c.prompt||state.prompt;
  $('case-note').textContent=(c.kind==='controlled_crop'?'两张图共享约 40% 的各自视野；完整参考图仅留在本地。':'输入顺序不固定左右，模型根据重叠内容判断位置。') + (c.limitations||'');
  const source=state.manifest.sources.find(s=>s.id===c.source_id);
  $('case-provenance').textContent=[source?.title,c.scene,c.tags?.join(' / '),c.review_required?'配对待目视检查':'',c.recommended?'推荐先试':''].filter(Boolean).join(' · ');
  document.querySelectorAll('.case').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.case===c.id)));
  const previous=state.runs.find(r=>r.case_id===c.id&&r.status==='succeeded');
  if(previous)showRun(previous);else{currentRun=null;$('comparison').hidden=true;$('download').hidden=true;$('empty-result').hidden=false;$('empty-result').textContent='这组素材还没有生成结果。';$('result-meta').textContent=c.title;$('metrics').replaceChildren();$('output-warnings').textContent=''}
}
function renderCases(reset=false) {
  if(reset)libraryPage=0;
  const query=$('case-search').value.trim().toLowerCase(),source=$('source-filter').value,suite=$('suite-filter').value;
  const cases=state.manifest.cases.filter(c=>(!source||c.source_id===source)&&(!query||[c.title,c.id,c.scene,...(c.tags||[])].join(' ').toLowerCase().includes(query))&&(suite==='all'||suite==='recommended'&&(c.recommended||c.id==='coffee-clean')||suite==='real'&&c.kind!=='controlled_crop'||suite==='controlled'&&Boolean(c.reference)||suite==='review'&&c.review_required));
  const pages=Math.max(1,Math.ceil(cases.length/pageSize));libraryPage=Math.min(libraryPage,pages-1);
  $('filter-count').textContent=`符合筛选：${cases.length} 对`;
  $('page-number').textContent=`${libraryPage+1} / ${pages}`;$('page-prev').disabled=libraryPage===0;$('page-next').disabled=libraryPage===pages-1;
  $('cases').replaceChildren();
  cases.slice(libraryPage*pageSize,(libraryPage+1)*pageSize).forEach(c=>{
    const b=document.createElement('button');b.className='case';b.dataset.case=c.id;
    b.textContent=c.title;b.setAttribute('aria-pressed',String(selectedCase?.id===c.id));b.onclick=()=>selectCase(c);$('cases').append(b);
  });
}
function renderLibrary() {
  $('library-count').textContent=`${state.manifest.sources.length} 个来源 · ${state.manifest.cases.length} 对样例`;
  state.manifest.sources.forEach(s=>{const option=document.createElement('option');option.value=s.id;option.textContent=s.title;$('source-filter').append(option)});
  $('evaluation-scope').textContent=state.manifest.evaluation_scope;
  $('sources').replaceChildren();
  state.manifest.sources.forEach(s=>{
    const p=document.createElement('p');p.textContent=`${s.title} · ${s.author} · ${s.license} `;
    if(s.source_url){const a=document.createElement('a');a.href=s.source_url;a.target='_blank';a.rel='noopener';a.textContent='来源';p.append(a)}
    $('sources').append(p);
  });
  selectCase(state.manifest.cases.find(c=>c.id==='coffee-clean') || state.manifest.cases[0]);
  renderCases();
}
function renderRuns() {
  $('runs').replaceChildren();
  if(!state.runs.length){const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=6;td.textContent='还没有实验记录。';tr.append(td);$('runs').append(tr);return}
  state.runs.forEach(r=>{
    const tr=document.createElement('tr');
    const title=state.manifest.cases.find(c=>c.id===r.case_id)?.title || '自选图片';
    const name=document.createElement('td');name.textContent=title;tr.append(name);
    const status=document.createElement('td'),tag=document.createElement('span');tag.className='status '+r.status;tag.textContent=statusLabels[r.status]||r.status;status.append(tag);tr.append(status);
    const model=document.createElement('td');model.textContent=r.model_requested||'—';tr.append(model);
    const elapsed=document.createElement('td');elapsed.textContent=r.elapsed_seconds==null?'—':r.elapsed_seconds.toFixed(1)+' 秒';tr.append(elapsed);
    const ssim=document.createElement('td');ssim.textContent=r.evaluation?.ssim_resized?.toFixed(3)||'—';tr.append(ssim);
    const action=document.createElement('td'),b=document.createElement('button');b.className='text-button';b.textContent=r.status==='succeeded'?'查看对照':'查看记录';b.onclick=()=>showRun(r);action.append(b);tr.append(action);$('runs').append(tr);
  });
}
function compare() {
  const ref=$('reference-image'); const hasRef=Boolean(currentRun?.reference_file);
  ref.hidden=!hasRef||compareMode==='output';
  ref.style.clipPath=compareMode==='split'?`inset(0 ${100-Number($('slider').value)}% 0 0)`:'none';
  $('slider-label').hidden=compareMode!=='split'||!hasRef;
  const referenceSize=currentRun?.evaluation?.reference_size;
  const dims=compareMode==='output'?currentRun?.size_actual:(referenceSize||currentRun?.size_actual);
  document.querySelector('.compare-frame').style.aspectRatio=dims?`${dims[0]}/${dims[1]}`:'3/2';
  $('result-image').style.objectFit=compareMode==='split'?'fill':'contain';
  ref.style.objectFit=compareMode==='split'?'fill':'contain';
  $('compare-note').textContent=compareMode==='split'?'左侧为完整参考图，右侧为生成结果；显示时统一缩放到参考图画布，尺寸比例不同时会拉伸。下载的原始结果不变。':compareMode==='reference'?'这是本地完整参考图，未发送给模型。':'这是服务商返回的原始生成结果，未进行修缝或颜色后处理。';
}
function showRun(r) {
  currentRun=r;$('metrics').replaceChildren();$('output-warnings').textContent='';
  const resultTitle=state.manifest.cases.find(c=>c.id===r.case_id)?.title||'自选图片';
  $('result-meta').textContent=`${resultTitle} · ${r.id} · ${statusLabels[r.status]||r.status} · ${r.provider?.name||''}`;
  const ok=r.status==='succeeded';$('comparison').hidden=!ok;$('empty-result').hidden=ok;$('download').hidden=!ok;
  if(!ok){$('empty-result').textContent=r.error || (r.status==='dry_run'?'已确认两张输入图片；没有调用图像生成服务。':'任务正在运行，请稍候。');return}
  $('result-meta').textContent+=` · ${r.size_actual.join(' × ')} · ${r.elapsed_seconds.toFixed(1)} 秒`;
  const actualQualities=(r.quality_returned||r.image_tool_metadata?.map(x=>x.quality)||[]).filter(Boolean).join(', ');
  $('output-warnings').textContent=`请求质量：${r.quality}；工具回报：${actualQualities||'未提供'}。${r.size_matches_request?'':'实际尺寸与请求不同。'} ${r.model_returned?'工具声明的图像模型：'+r.model_returned:'图像工具未声明模型名，无法核实上游是否使用 GPT Image 2.5。'}`;
  $('result-image').src=runSrc(r,'result.png');$('download').href=runSrc(r,'result.png');
  if(r.reference_file)$('reference-image').src=runSrc(r,r.reference_file);
  $('show-reference').hidden=!r.reference_file;$('show-split').hidden=!r.reference_file;
  compareMode='output';compare();
  if(r.evaluation?.reference_available){
    [['整体相似度 SSIM',r.evaluation.ssim_resized.toFixed(3)],['平均像素差 / 255',r.evaluation.mae_rgb_0_255.toFixed(1)],['重叠区外像素差 / 255',r.evaluation.outside_overlap_mae_rgb_0_255.toFixed(1)]].forEach(([label,value])=>{
      const d=document.createElement('div');d.className='metric';d.textContent=label;const b=document.createElement('b');b.textContent=value;d.append(b);$('metrics').append(d);
    });
  }
}
async function refresh() {
  const response=await fetch('/api/runs');if(!response.ok)throw Error('无法读取运行记录。');
  const data=await response.json();state.runs=data.runs;state.busy=data.busy;renderRuns();
  showShare(data.share);$('generate').disabled=data.busy||state.share?.remaining===0;$('dry-run').disabled=data.busy;
  if(pending){const r=data.runs.find(r=>r.id===pending);if(r&&r.status!=='running'){pending=null;showRun(r);note(r.status==='failed'?r.error:r.status==='dry_run'?'请求检查通过，没有产生生图用量。':'融合完成，可以查看结果与原图。',r.status==='failed')}}
}
async function submit(dryRun) {
  $('generate').disabled=true;$('dry-run').disabled=true;note(dryRun?'正在检查请求…':'正在发送两张原图，等待服务商返回结果…');
  try{
    const body={case_id:selectedCase?.id,uploads,prompt:$('prompt').value,model:$('model').value,quality:$('quality').value,size:$('size').value,dry_run:dryRun};
    const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json','X-Fusion-Token':state.token},body:JSON.stringify(body)});
    const data=await response.json();if(!response.ok)throw Error(data.error||'提交失败。');pending=data.id;await refresh();
  }catch(e){note(e.message,true);$('generate').disabled=false;$('dry-run').disabled=false}
}
async function fileData(file) {return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=()=>reject(Error('读取图片失败。'));reader.readAsDataURL(file)})}
$('use-uploads').onclick=async()=>{
  try{const l=$('left-upload').files[0],r=$('right-upload').files[0];if(!l||!r)throw Error('请先选择左图和右图。');if(l.size>30*1024*1024||r.size>30*1024*1024)throw Error('单张图片不能超过 30 MB。');uploads=await Promise.all([fileData(l),fileData(r)]);selectedCase=null;$('left-image').src=URL.createObjectURL(l);$('right-image').src=URL.createObjectURL(r);$('case-title').textContent='自选图片';$('case-kind').textContent='无完整参考图';$('case-note').textContent='图片会先在本机转换为 PNG 并校正 EXIF 方向，再直接提交；不进行预对齐或修缝。';document.querySelectorAll('.case').forEach(b=>b.setAttribute('aria-pressed','false'));note('自选图片已就绪，尚未发送。')}catch(e){note(e.message,true)}
};
$('generate').onclick=()=>submit(false);$('dry-run').onclick=()=>submit(true);$('refresh').onclick=()=>refresh().catch(e=>note(e.message,true));
$('case-search').oninput=()=>renderCases(true);$('source-filter').onchange=()=>renderCases(true);$('suite-filter').onchange=()=>renderCases(true);
$('page-prev').onclick=()=>{libraryPage--;renderCases()};$('page-next').onclick=()=>{libraryPage++;renderCases()};
$('slider').oninput=compare;
[['show-output','output'],['show-reference','reference'],['show-split','split']].forEach(([id,mode])=>$(id).onclick=()=>{compareMode=mode;compare()});
(async()=>{try{const response=await fetch('/api/state');state=await response.json();if(!response.ok)throw Error(state.error);if(location.hash.startsWith('#invite='))history.replaceState(null,'',location.pathname);$('connection').textContent=state.provider.configured?`CC Switch · ${state.provider.name} · ${state.mainline_model}`:'CC Switch 配置不可用';$('prompt').value=state.prompt;renderLibrary();renderRuns();showShare(state.share);$('generate').disabled=state.busy||state.share?.remaining===0;setInterval(()=>refresh().catch(e=>note(e.message,true)),3000)}catch(e){note(e.message,true)}})();
