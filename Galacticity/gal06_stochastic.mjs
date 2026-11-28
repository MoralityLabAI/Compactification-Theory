import { pathToFileURL } from "node:url";

function rng(seed){let x=seed>>>0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return x>>>0}}

function mass(e,B){let n=0;for(let w=0;w<e.weights.length;w++)if(B&(1<<w))n+=e.weights[w];return n}

function payload(e,M){return e.projects.reduce((s,p,i)=>s+((M&(1<<i))?p.size:0),0)}

function legal(e,t,K,M,P){let opts=[-1],pending=P.reduce((s,[i])=>s|(1<<i),0),done=M&~pending;if(M&(1<<e.goal))return opts;for(let i=0;i<e.projects.length;i++){const p=e.projects[i];if(!(M&(1<<i))&&(done&p.pre)===p.pre&&p.cost<=K&&t+p.delay<=e.H&&payload(e,M)+p.size<=e.cap)opts.push(i)}return opts}

function acquire(e,t,K,M,P,i){if(i<0)return {K,M,P};const p=e.projects[i];return {K:K-p.cost,M:M|(1<<i),P:[...P,[i,t+p.delay]].sort((a,b)=>a[0]-b[0])}}

function reserve(e,B,P){let worst=0,ts=[...new Set(P.map(x=>x[1]))].sort((a,b)=>a-b);for(let w=0;w<e.weights.length;w++)if(B&(1<<w)){let cum=0;for(const t of ts){for(const [i,d] of P)if(d===t)cum+=e.projects[i].returns[w];worst=Math.max(worst,-cum)}}return worst}

function marginalReserve(e,B,P){return P.reduce((sum,[i])=>{let lo=Infinity;for(let w=0;w<e.weights.length;w++)if(B&(1<<w))lo=Math.min(lo,e.projects[i].returns[w]);return sum+Math.max(0,-lo)},0)}

function prune(points){let out=[],best=-1;points.sort((a,b)=>a.r-b.r||b.s-a.s);for(const p of points)if(p.s>best){out.push(p);best=p.s}return out}

function oracle(e,capital){let memo=new Map(),nodes=0,goal=1<<e.goal,full=(1<<e.weights.length)-1;
 function pre(t,K,M,P,B){
  const id=[t,K,M,B,P.map(x=>x.join(":")).join(",")].join("|");if(memo.has(id))return memo.get(id);nodes++;
  const due=P.filter(x=>x[1]===t),left=P.filter(x=>x[1]!==t),groups=new Map();
  for(let w=0;w<e.weights.length;w++)if(B&(1<<w)){const vals=due.map(([i])=>e.projects[i].returns[w]),key=vals.join(",");let g=groups.get(key);if(!g){g={B:0,delta:vals.reduce((s,x)=>s+x,0)};groups.set(key,g)}g.B|=1<<w}
  let acc=[{s:0,r:0,w:{kind:"chance",by:{}}}];
  for(const [key,g] of groups){let p;const k=K+g.delta;
   if(k<0)p=[{s:0,r:mass(e,g.B),w:{kind:"end"}}];else if(t===e.H){if(left.length)throw Error("unsettled terminal");p=[{s:M&goal?mass(e,g.B):0,r:0,w:{kind:"end"}}]}else p=post(t,k,M,left,g.B);
   let next=[];for(const a of acc)for(const b of p)next.push({s:a.s+b.s,r:a.r+b.r,w:{kind:"chance",by:{...a.w.by,[key]:b.w}}});acc=prune(next);
  }memo.set(id,acc);return acc
 }
 function post(t,K,M,P,B){let a=[];for(const i of legal(e,t,K,M,P)){let x=acquire(e,t,K,M,P,i);for(const p of pre(t+1,x.K,x.M,x.P,B))a.push({s:p.s,r:p.r,w:{kind:"act",i,next:p.w}})}return prune(a)}
 return {frontier:pre(0,capital,0,[],full),nodes}
}

function replayWorld(e,capital,w,tree,mode){let K=capital,M=0,B=(1<<e.weights.length)-1,due=Array(e.projects.length).fill(-1),node=tree,actions=[],peakReserve=0;for(let t=0;t<=e.H;t++){
 let obs=[],ids=[];for(let i=0;i<due.length;i++)if(due[i]===t){ids.push(i);obs.push(e.projects[i].returns[w]);K+=obs[obs.length-1];due[i]=-1}let nb=0;for(let j=0;j<e.weights.length;j++)if(B&(1<<j)&&ids.every((i,k)=>e.projects[i].returns[j]===obs[k]))nb|=1<<j;B=nb;if(!(B&(1<<w)))throw Error("belief excludes world");
 if(tree){if(node?.kind!=="chance")throw Error("bad chance witness");node=node.by[obs.join(",")];if(!node)throw Error("missing observation branch")}
 if(K<0||t===e.H){if(tree&&node.kind!=="end")throw Error("nonterminal witness");return {success:K>=0&&!!(M&(1<<e.goal)),ruin:K<0,actions,peakReserve}}
 let P=[];for(let i=0;i<due.length;i++)if(due[i]>=0)P.push([i,due[i]]);let i=tree?node.i:choose(e,t,K,M,P,B,mode);
 if(tree){if(node.kind!=="act")throw Error("bad act witness");node=node.next}
 if(!Number.isInteger(i)||i< -1||i>=e.projects.length)throw Error("bad action");
 if(i>=0){let p=e.projects[i],complete=M;for(let j=0;j<due.length;j++)if(due[j]>=0)complete&=~(1<<j);if(M&(1<<e.goal)||M&(1<<i)||(complete&p.pre)!==p.pre||p.cost>K||t+p.delay>e.H||payload(e,M)+p.size>e.cap)throw Error("illegal replay action");K-=p.cost;M|=1<<i;due[i]=t+p.delay}
 let pp=[];for(let j=0;j<due.length;j++)if(due[j]>=0)pp.push([j,due[j]]);peakReserve=Math.max(peakReserve,reserve(e,B,pp));if(mode==="joint"||mode==="marginal"){if(K<reserve(e,B,pp))throw Error("reserve violation")}
 actions.push(i)
 }throw Error("unreachable replay")}

function replay(e,K,tree=null,mode=null){let s=0,r=0,paths=[];for(let w=0;w<e.weights.length;w++){let v=replayWorld(e,K,w,tree,mode);s+=v.success?e.weights[w]:0;r+=v.ruin?e.weights[w]:0;paths.push(v.actions)}return {s,r,paths}}

function choose(e,t,K,M,P,B,mode){let opts=legal(e,t,K,M,P),mandatory=1<<e.goal;for(let z=0;z<e.projects.length;z++)for(let i=0;i<e.projects.length;i++)if(mandatory&(1<<i))mandatory|=e.projects[i].pre;
 let rs=mode==="joint"?reserve:mode==="marginal"?marginalReserve:()=>0;
 opts=opts.filter(i=>{let x=acquire(e,t,K,M,P,i);return i<0||payload(e,x.M|mandatory)<=e.cap&&x.K>=rs(e,B,x.P)});
 if(opts.includes(e.goal))return e.goal;if(mode==="wait")return -1;let best=-1,score=0;
 for(const i of opts)if(i>=0){let p=e.projects[i],value=-p.cost;for(let w=0;w<e.weights.length;w++)if(B&(1<<w))value+=p.returns[w]*e.weights[w]/mass(e,B);if(mandatory&(1<<i))value+=100;
 value/=p.delay;if(value>score){best=i;score=value}}
 return best}

function brute(e,K0){const full=e.weights.map((_,i)=>i);
 function f(t,K,M,due,worlds){let ids=due.map((x,i)=>x===t?i:-1).filter(i=>i>=0),groups=new Map();for(const w of worlds){let key=JSON.stringify(ids.map(i=>e.projects[i].returns[w]));if(!groups.has(key))groups.set(key,[]);groups.get(key).push(w)}let combinations=[[0,0]];
 for(const [key,ws] of groups){let cash=K+JSON.parse(key).reduce((s,x)=>s+x,0),weight=ws.reduce((s,w)=>s+e.weights[w],0),points=[],d=due.map(x=>x===t?-1:x);
 if(cash<0)points=[[0,weight]];else if(t===e.H)points=[[M&(1<<e.goal)?weight:0,0]];else{for(let i=-1;i<e.projects.length;i++){let nm=M,nd=d.slice(),nc=cash;if(i>=0){let p=e.projects[i];if(M&(1<<e.goal)||M&(1<<i)||p.cost>cash||t+p.delay>e.H)continue;let ok=true,sz=p.size;for(let j=0;j<e.projects.length;j++){if(M&(1<<j))sz+=e.projects[j].size;if(p.pre&(1<<j)&&(!(M&(1<<j))||d[j]>=0))ok=false}if(!ok||sz>e.cap)continue;nm|=1<<i;nd[i]=t+p.delay;nc-=p.cost}points.push(...f(t+1,nc,nm,nd,ws))}}
 let temp=new Set();for(const a of combinations)for(const b of points)temp.add(`${a[0]+b[0]},${a[1]+b[1]}`);combinations=[...temp].map(x=>x.split(",").map(Number))}
 return combinations}
 let pts=f(0,K0,0,e.projects.map(()=>-1),full);return pts.filter(a=>!pts.some(b=>b[0]>=a[0]&&b[1]<=a[1]&&(b[0]>a[0]||b[1]<a[1]))).sort((a,b)=>a[1]-b[1]||b[0]-a[0])}

function gen(seed,family){let q=rng(seed),a=2+q()%3,b=2+q()%3,goal=8+q()%7,H=5+q()%2,cap=3+q()%2;let p=[{cost:a,delay:1,size:1,pre:0,returns:[a+2,a+2,a+2,a+2]},{cost:b,delay:2,size:1,pre:0,returns:[b+9,b+5,-b-2,-b-1]},{cost:2,delay:2,size:1,pre:0,returns:[8,6,-3,-2]},{cost:goal,delay:1,size:1,pre:0,returns:[0,0,0,0]}];
 if(family==="hedged")p[2].returns=[-3,-2,8,6];
 else if(family==="delayed"){p[0].delay=3;p[1].delay=3;p[2].delay=1;p[3].pre=1}
 else if(family==="correlated")p[3].pre=2;
 else if(family==="shift"){p[0].returns=[a+4,a+1,-1,0];p[1].returns=[-4,12,-3,10];p[2].returns=[8,-5,9,-4];p[2].delay=1;p[3].pre=4}
 else if(family!=="stable")throw Error("family");
 return {projects:p,weights:[1,1,1,1],goal:3,H,cap}}

function run(){let checks=0;const check=(v,msg)=>{checks++;if(!v)throw Error("check "+checks+" "+msg)};let tiny=0,witness=0,nodes=0;
 for(let seed=1;seed<=32;seed++){let q=rng(seed*173),e={weights:[1,2],goal:1,H:2+q()%2,cap:1+q()%2,projects:[{cost:1+q()%3,delay:1+q()%2,size:1,pre:0,returns:[Number(q()%7)-3,Number(q()%7)-3]},{cost:2+q()%4,delay:1,size:1,pre:q()%2,returns:[0,0]}]};for(let K=0;K<=6;K++){let o=oracle(e,K);check(JSON.stringify(o.frontier.map(p=>[p.s,p.r]))===JSON.stringify(brute(e,K)),"tiny brute");for(const p of o.frontier){let z=replay(e,K,p.w);check(z.s===p.s&&z.r===p.r,"tiny witness");witness++}tiny++}}
 let rows=[],families=["stable","correlated","hedged","delayed","shift"],modes=["wait","expected","joint","marginal"];let minRows=[];
 for(let fi=0;fi<families.length;fi++)for(let j=0;j<8;j++){let seed=60001+fi*100+j,family=families[fi],e=gen(seed,family),minima={"4/0":null,"3/0":null,"3/1":null,"2/1":null};let prior=[0,0,0,0,0];
 for(let K=0;K<=24;K++){let o=oracle(e,K);nodes+=o.nodes;let best=[0,1,2,3,4].map(r=>Math.max(...o.frontier.filter(p=>p.r<=r).map(p=>p.s)));check(best.every((v,i)=>v>=prior[i]),"monotone "+family+" "+seed+" "+K);prior=best;
 for(let k of Object.keys(minima)){let [s,r]=k.split("/").map(Number);if(minima[k]===null&&best[r]>=s)minima[k]=K}
 for(const p of o.frontier){let v=replay(e,K,p.w);check(v.s===p.s&&v.r===p.r,"main witness");witness++}
 if([2,4,6,8,10].includes(K)){let policies={};for(const mode of modes){let p=replay(e,K,null,mode);check(mode!=="joint"&&mode!=="marginal"||p.r===0,"guard "+mode);check(o.frontier.some(q=>q.s>=p.s&&q.r<=p.r),"dominance");policies[mode]={s:p.s,r:p.r}}rows.push({family,seed,K,oracle:best,frontier:o.frontier.map(p=>[p.s,p.r]),policies})}
 }minRows.push({family,seed,minima})}
 let summary={};for(const f of [...families,"all"]){let rr=rows.filter(r=>f==="all"||r.family===f);let ss={cases:rr.length,worldMass:rr.length*4,oracleSafe:rr.reduce((a,r)=>a+r.oracle[0],0),oracleQuarter:rr.reduce((a,r)=>a+r.oracle[1],0),policies:{}};for(const mode of modes)ss.policies[mode]={successMass:rr.reduce((a,r)=>a+r.policies[mode].s,0),ruinMass:rr.reduce((a,r)=>a+r.policies[mode].r,0)};summary[f]=ss}
 let hedge={weights:[1,1],goal:2,H:4,cap:3,projects:[{cost:1,delay:3,size:1,pre:0,returns:[-5,8]},{cost:1,delay:2,size:1,pre:0,returns:[8,-5]},{cost:4,delay:1,size:1,pre:0,returns:[0,0]}]};check(reserve(hedge,3,[[0,3],[1,3]])===0,"hedge");check(marginalReserve(hedge,3,[[0,3],[1,3]])===10,"marginal");check(reserve(hedge,3,[[0,2],[1,3]])===5,"staggered");
 let gamble={weights:[1,1],goal:1,H:3,cap:2,projects:[{cost:1,delay:1,size:1,pre:0,returns:[10,-4]},{cost:5,delay:1,size:1,pre:0,returns:[0,0]}]};let g=oracle(gamble,1);check(JSON.stringify(g.frontier.map(p=>[p.s,p.r]))===JSON.stringify([[0,0],[1,1]]),"gamble");
 return {schema:"gal06.stochastic.v1",checks,tinyBruteCases:tiny,witnessReplays:witness,oracleNodes:nodes,environmentCount:minRows.length,gridCases:minRows.length*25,policyCases:rows.length,capitalSweep:[0,24],summary,minima:minRows,examples:{hedge:{jointSameDue:0,marginalSameDue:10,jointStaggered:5},gamble:{seed:1,frontier:g.frontier.map(p=>[p.s,p.r])}},rows,scope:"exact finite weighted hidden-world model; deterministic policies, atomic settlements, known scenario table, no controller cost or wall timing; robust reserves assume all liabilities modeled; null minima right censored above capital 24"}
}

export {gen, oracle, replay, reserve, marginalReserve, mass};
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) console.log(JSON.stringify(run(),null,2));
