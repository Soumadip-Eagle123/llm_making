const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg',W=124,H=46,CW=176,RH=58;
const el=(t,a={},p)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.append(e);return e};
const f=(x,d=4)=>x==null?'—':(+x).toFixed(d),sg=x=>Math.abs(x)<1e-12?'':x>0?'+':'−';
const st={E:[],cur:-1,g:null,losses:[],playing:false,done:0,epochs:1,updFetched:false,sel:null,init:{},params:{},pos:[],tf:{x:0,y:0,s:1},dim:new Set(),ex:[],dead:{},moved:0};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function api(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
 const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j}
const msg=(t,e)=>{$('msg').textContent=t;$('msg').className=e?'err':''};
const cfg=()=>({nin:$('nin').value,nout:$('nout').value,hidden:$('hidden').value,act:$('act').value,out_act:$('out_act').value,
 loss:$('loss').value,lr:$('lr').value,seed:$('seed').value,data:$('data').value});

/* ---------- training flow ---------- */
async function build(){st.playing=false;
 try{const ep=parseInt($('epochs').value);if(!(ep>=1))throw new Error('Epochs must be at least 1.');
  const r=await api('/api/init',cfg());
  Object.assign(st,{epochs:ep,losses:[],done:0,E:[],cur:-1,sel:null,init:r.params,params:{...r.params}});
  $('focus').value='all';
  $('np').textContent=r.nparams;$('kLr').textContent=$('lr').value;$('kAct').textContent=$('act').value;
  await loadIter();fitView();render();paramTable();msg('Model built. Press Step or Play.')}
 catch(e){msg(e.message,1)}ui()}
async function loadIter(){const r=await api('/api/iteration');
 Object.assign(st,{g:r.graph,E:r.events,cur:-1,updFetched:false,dead:r.dead,ex:r.examples,params:r.params});
 st.losses.push(r.loss);st.done++;
 if($('focus').options.length!==st.ex.length+1)
  $('focus').innerHTML='<option value="all">all examples</option>'+st.ex.map((_,i)=>`<option value="${i}">example ${i}</option>`).join('');
 layout();draw();focus();exTable();deadPanel()}
async function advance(){
 if(st.cur<st.E.length-1){st.cur++;render();return true}
 if(!st.updFetched){const r=await api('/api/update');st.E.push(...r.events);st.params=r.params;st.updFetched=true;st.cur++;render();paramTable();return true}
 if(st.done>=st.epochs)return false;
 await loadIter();render();return true}
async function play(){if(!st.g||st.playing)return;st.playing=true;ui();
 try{while(st.playing){const d=+$('delay').value;let ok=true;
  if(d===0&&st.cur<st.E.length-1){st.cur=st.E.length-1;render()}else ok=await advance();
  if(!ok){st.playing=false;msg('Training finished.');break}await sleep(d)}}
 catch(e){st.playing=false;msg(e.message,1)}ui()}
async function stepFwd(){if(!st.g)return;st.playing=false;try{if(!(await advance()))msg('Training finished.')}catch(e){msg(e.message,1)}ui()}
function stepBack(){st.playing=false;if(st.cur>-1){st.cur--;render()}else msg('Start of this iteration (cannot step into the previous one).');ui()}

/* ---------- layout: columns by depth, rows ordered to reduce crossings ---------- */
function layout(){const N=st.g.nodes,n=N.length,d=Array(n).fill(0),ch=N.map(()=>[]);
 N.forEach((v,i)=>v.parents.forEach(p=>ch[p].push(i)));
 N.forEach((v,i)=>{if(v.parents.length)d[i]=1+Math.max(...v.parents.map(p=>d[p]))});
 for(let i=n-1;i>=0;i--)if(!N[i].parents.length&&ch[i].length)d[i]=Math.min(...ch[i].map(c=>d[c]))-1;
 const m0=Math.min(...d);for(let i=0;i<n;i++)d[i]-=m0;
 const cols=[];N.forEach((v,i)=>(cols[d[i]]=cols[d[i]]||[]).push(i));
 const y=Array(n);cols.forEach(c=>c.forEach((i,k)=>y[i]=k));
 const mean=(a,def)=>a.length?a.reduce((s,i)=>s+y[i],0)/a.length:def;
 for(let it=0;it<8;it++){const back=it%2===1,order=cols.map((_,k)=>k);if(back)order.reverse();
  for(const k of order){const c=cols[k],key=i=>mean(back?ch[i]:N[i].parents,y[i]);
   const ks=new Map(c.map(i=>[i,key(i)]));c.sort((a,b)=>ks.get(a)-ks.get(b));c.forEach((i,r)=>y[i]=r)}}
 const tall=Math.max(...cols.map(c=>c.length));
 st.pos=Array(n);cols.forEach((c,k)=>c.forEach((i,r)=>st.pos[i]={x:k*CW,y:(r+(tall-c.length)/2)*RH}))}

function draw(){const g=$('vp');g.innerHTML='';const N=st.g.nodes;st.edges=[];st.nodes=[];
 const eg=el('g',{},g),ng=el('g',{},g);
 N.forEach((v,i)=>v.parents.forEach(p=>{const a=st.pos[p],b=st.pos[i],x1=a.x+W,y1=a.y+H/2,x2=b.x,y2=b.y+H/2,h=(x2-x1)/2;
  st.edges.push({l:el('path',{class:'edge',d:`M${x1} ${y1}C${x1+h} ${y1},${x2-h} ${y2},${x2} ${y2}`},eg),p,c:i})}));
 N.forEach((v,i)=>{const p=st.pos[i],n=el('g',{class:'node '+v.kind,tabindex:0,role:'button','aria-label':v.label,transform:`translate(${p.x},${p.y})`},ng);
  el('rect',{width:W,height:H},n);
  const t=[0,1,2].map(k=>el('text',{x:6,y:14+k*14,class:k===0?'t0':k===2?'g':''},n));
  t[0].textContent=v.label.length>18?v.label.slice(0,17)+'…':v.label;
  const pick=()=>{if(st.moved>5){st.moved=0;return}st.sel=i;render()};
  n.onclick=pick;n.onkeydown=e=>{if(e.key==='Enter')pick()};st.nodes.push({n,t})});
 applyTf()}

/* ---------- pan / zoom ---------- */
function applyTf(){const t=st.tf;$('vp').style.transform=`translate(${t.x}px,${t.y}px) scale(${t.s})`}
function zoomAt(k,cx,cy){const t=st.tf,s=Math.min(3,Math.max(.05,t.s*k));k=s/t.s;t.x=cx-(cx-t.x)*k;t.y=cy-(cy-t.y)*k;t.s=s;applyTf()}
function zoomC(k){const r=$('svg').getBoundingClientRect();zoomAt(k,r.width/2,r.height/2)}
function fitView(){if(!st.pos.length)return;const r=$('svg').getBoundingClientRect(),xs=st.pos.map(p=>p.x),ys=st.pos.map(p=>p.y);
 const x0=Math.min(...xs),y0=Math.min(...ys),bw=Math.max(...xs)+W-x0,bh=Math.max(...ys)+H-y0;
 const s=Math.min(1.1,.94*Math.min(r.width/bw,r.height/bh));
 st.tf={s,x:(r.width-bw*s)/2-x0*s,y:(r.height-bh*s)/2-y0*s};applyTf()}
function centerOn(i){const p=st.pos[i],r=$('svg').getBoundingClientRect(),s=Math.max(st.tf.s,.8),vp=$('vp');
 st.tf={s,x:r.width/2-(p.x+W/2)*s,y:r.height/2-(p.y+H/2)*s};vp.classList.add('smooth');applyTf();setTimeout(()=>vp.classList.remove('smooth'),300)}
(()=>{const s=$('svg');let drag=null;
 s.addEventListener('wheel',e=>{e.preventDefault();const r=s.getBoundingClientRect();zoomAt(e.deltaY>0?.89:1.12,e.clientX-r.left,e.clientY-r.top)},{passive:false});
 s.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,m:0};st.moved=0});
 addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.m+=Math.abs(dx)+Math.abs(dy);
  drag.x=e.clientX;drag.y=e.clientY;st.tf.x+=dx;st.tf.y+=dy;applyTf()});
 addEventListener('pointerup',()=>{if(drag)st.moved=drag.m;drag=null});
 addEventListener('resize',()=>{if(st.g)fitView()})})();

/* ---------- focus one example (others fade) ---------- */
function focus(){const k=$('focus').value;st.dim=new Set();if(k==='all'||!st.g)return;
 const N=st.g.nodes,keep=new Set(),re=new RegExp('^sq\\d+\\['+k+'\\]$'),stack=[];
 N.forEach((v,i)=>{if(re.test(v.label))stack.push(i)});
 while(stack.length){const i=stack.pop();if(keep.has(i))continue;keep.add(i);stack.push(...N[i].parents)}
 N.forEach((v,i)=>{if(!keep.has(i))st.dim.add(i)})}

/* ---------- replay state, re-derived from the recorded events ---------- */
function view(){const N=st.g.nodes,shown=N.map(v=>!v.op),grad=N.map(()=>0),cnt=N.map(()=>0);let started=false;
 for(let i=0;i<=st.cur;i++){const e=st.E[i];
  if(e.phase==='forward')shown[e.node]=true;
  else if(e.phase==='seed'){grad[e.node]=1;cnt[e.node]=1;started=true}
  else if(e.phase==='backward')e.contrib.forEach(c=>{grad[c.to]+=c.delta;cnt[c.to]++})}
 return{shown,grad,cnt,started,ev:st.E[st.cur]}}
function describe(e){const N=st.g.nodes;
 if(!e)return 'READY — the forward pass begins at the inputs. Each step reveals one real operation.';
 if(e.phase==='forward')return 'FORWARD — '+e.text;
 if(e.phase==='loss')return 'LOSS — '+e.text;
 if(e.phase==='zero')return 'GRADIENT RESET — every .grad = 0 before backpropagation.';
 if(e.phase==='seed')return 'BACKWARD — dL/dL = 1 seeds the loss node.';
 if(e.phase==='backward')return `BACKWARD — ${N[e.node].label}: incoming grad ${f(e.grad_in)}. `+e.contrib.map(c=>
  `${N[c.to].label}.grad += ${f(e.grad_in)} × local ${f(c.local)} = ${sg(c.delta)}${f(Math.abs(c.delta))}`).join(' ; ');
 return `UPDATE — ${e.label}: new = old − lr × grad = ${f(e.old)} − ${e.lr} × ${f(e.grad)} = ${f(e.new)} (Δ ${sg(e.delta)}${f(Math.abs(e.delta))})`}
function render(){if(!st.g)return;const V=view(),ev=V.ev,N=st.g.nodes;
 N.forEach((v,i)=>{const o=st.nodes[i];o.t[1].textContent=V.shown[i]?'= '+f(v.data):'= …';
  const g=V.grad[i];o.t[2].textContent=V.started&&V.cnt[i]?`∂ ${sg(g)}${f(Math.abs(g))}`+(V.cnt[i]>1?` ×${V.cnt[i]}`:''):'';
  o.t[2].setAttribute('class','g '+(Math.abs(g)<1e-12?'zero':g>0?'pos':'neg'));
  o.n.classList.toggle('active',!!ev&&ev.node===i);o.n.classList.toggle('sel',st.sel===i);o.n.classList.toggle('dim',st.dim.has(i))});
 st.edges.forEach(e=>{const L=e.l.classList;
  L.toggle('hot',!!ev&&ev.phase==='backward'&&ev.node===e.c&&ev.contrib.some(c=>c.to===e.p));
  L.toggle('fwd',!!ev&&ev.phase==='forward'&&ev.node===e.c);L.toggle('dim',st.dim.has(e.c)||st.dim.has(e.p))});
 if($('follow').checked&&ev&&ev.node!=null&&st.pos[ev.node])centerOn(ev.node);
 $('step').textContent=describe(ev);
 $('kPhase').textContent=!ev?'start':{forward:'forward pass',loss:'loss',zero:'gradient reset',seed:'backprop',backward:'backprop',update:'parameter update'}[ev.phase];
 const L=st.losses,n=L.length;$('kEp').textContent=`${st.done} / ${st.epochs}`;$('kLoss').textContent=f(L[n-1]);
 $('kPrev').textContent=n>1?f(L[n-2]):'—';$('kChg').textContent=n>1?(L[n-1]-L[n-2]>=0?'+':'−')+f(Math.abs(L[n-1]-L[n-2])):'—';
 $('prog').value=(st.done-1+(st.cur+1)/Math.max(1,st.E.length))/st.epochs;chart();inspect(V)}

/* ---------- panels ---------- */
function chart(){const c=$('chart');c.innerHTML='';const L=st.losses;if(!L.length)return;
 const mx=Math.max(...L),mn=Math.min(...L),sp=(mx-mn)||1,X=i=>34+i*(236/Math.max(1,L.length-1)),Y=v=>88-(v-mn)/sp*72;
 el('polyline',{class:'ln',points:L.map((v,i)=>`${X(i)},${Y(v)}`).join(' ')},c);
 [[mx,12],[mn,92]].forEach(([v,y])=>{el('text',{x:0,y},c).textContent=f(v,2)})}
function inspect(V){const i=st.sel;if(i==null){$('insp').textContent='Click a node.';return}
 const N=st.g.nodes,v=N[i],bw=st.E.slice(0,st.cur+1).find(e=>e.phase==='backward'&&e.node===i);
 $('insp').innerHTML=`<b>${v.label}</b> <span class="tag">${v.kind}${v.op?' · '+v.op:''}</span><dl>
 <dt>data</dt><dd>${V.shown[i]?f(v.data,6):'not computed yet'}</dd><dt>grad</dt><dd>${V.started&&V.cnt[i]?f(V.grad[i],6):'—'}</dd>
 <dt>parents</dt><dd>${v.parents.map(p=>N[p].label+'='+f(N[p].data)).join(', ')||'none (leaf)'}</dd>
 ${bw?`<dt>incoming</dt><dd>${f(bw.grad_in,6)}</dd><dt>local d</dt><dd>${bw.contrib.map(c=>N[c.to].label+': '+f(c.local,6)).join('<br>')}</dd>`:''}</dl>`}
function exTable(){$('ex').innerHTML='<tr><th>#</th><th>target</th><th>pred</th><th>sq err</th></tr>'+st.ex.map((x,i)=>
 `<tr class="click" data-n="${x.node}"><td>${i}</td><td>${x.target.map(v=>f(v,2))}</td><td>${x.pred.map(v=>f(v,3))}</td><td>${f(x.sq)}</td></tr>`).join('');
 $('ex').querySelectorAll('tr.click').forEach(r=>r.onclick=()=>{st.sel=+r.dataset.n;render();centerOn(st.sel)})}
function deadPanel(){const d=Object.entries(st.dead||{});$('dead').innerHTML=d.length?
 '<p>Share of examples with z ≤ 0 (ReLU output 0, gradient 0). 100% means the neuron is dead on this data.</p><table>'+
 d.map(([k,v])=>`<tr><td>${k}</td><td>${(v*100).toFixed(0)}%</td><td>${v===1?'DEAD':v>0?'partly inactive':'active'}</td></tr>`).join('')+'</table>':'Tanh selected: no ReLU neurons.'}
function paramTable(){$('pt').innerHTML='<tr><th>param</th><th>initial</th><th>current</th><th>Δ</th></tr>'+Object.keys(st.init).map(k=>
 `<tr><td>${k}</td><td>${f(st.init[k],3)}</td><td>${f(st.params[k],3)}</td><td>${f(st.params[k]-st.init[k],3)}</td></tr>`).join('')}

/* ---------- export, theme, wiring ---------- */
async function exp(){try{const r=await fetch('/api/export.jpg');if(!r.ok)throw new Error((await r.json()).error);
 const a=document.createElement('a');a.href=URL.createObjectURL(await r.blob());a.download='computational_graph.jpg';a.click();msg('JPG downloaded (snapshot taken after the last backprop).')}
 catch(e){msg(e.message,1)}}
function ui(){const has=!!st.g;$('bPlay').textContent=st.playing?'Pause':(st.cur>-1||st.done>1?'Resume':'Play');
 ['bPlay','bFwd','bBack','bJpg','bFit','bZin','bZout'].forEach(b=>$(b).disabled=!has)}
function theme(t,save){document.documentElement.dataset.theme=t;
 $('bTheme').textContent=t==='dark'?'Light mode':'Dark mode';$('bTheme').setAttribute('aria-pressed',t==='dark');
 if(save)try{localStorage.setItem('theme',t)}catch(e){}}
$('bBuild').onclick=build;$('bReset').onclick=build;$('bPlay').onclick=()=>st.playing?(st.playing=false,ui()):play();
$('bFwd').onclick=stepFwd;$('bBack').onclick=stepBack;$('bFit').onclick=fitView;$('bZin').onclick=()=>zoomC(1.25);$('bZout').onclick=()=>zoomC(.8);$('bJpg').onclick=exp;
$('delay').oninput=()=>$('dv').textContent=$('delay').value+' ms';
$('focus').onchange=()=>{focus();render()};
$('bTheme').onclick=()=>theme(document.documentElement.dataset.theme==='dark'?'light':'dark',true);
theme(document.documentElement.dataset.theme||'light');ui();