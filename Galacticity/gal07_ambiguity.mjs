function rng(seed){let x=seed>>>0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return x>>>0}}

function mass(weights,mask){return weights.reduce((s,w,i)=>s+((mask&(1<<i))?w:0),0)}

function gen(seed,family){let rnd=rng(seed),c=4+rnd()%3,d=4+rnd()%3,e=6+rnd()%3,L=4+rnd()%4;
 let projects=[{cost:c,returns:[c+5,c+4,c+3,c+2,-L,-L-1]},{cost:d,returns:[d+4,-L,d+3,-L-1,d+2,-L]},{cost:e,returns:[0,e+3,0,e+4,0,e+3]}];
 let tests=[{cost:1+rnd()%2,signals:[0,0,0,0,1,1]},{cost:1+rnd()%2,signals:[0,1,0,1,0,1]},{cost:2+rnd()%2,signals:[0,0,1,1,1,0]},{cost:4+rnd()%2,signals:[0,1,2,3,4,5]}],limit=2;
 if(family==="aliased"){tests[0].signals=[0,0,1,1,0,1];tests[1].signals=[0,0,1,1,0,1];tests[2].signals=[0,1,0,1,1,0];tests[3].cost+=2}
 else if(family==="alternative"){projects[1].returns=[-L,-L,-L,-L,d+6,d+5];projects[2].cost+=3;projects[2].returns=projects[2].returns.map((v,i)=>i%2?v+3:v)}
 else if(family==="hazard"){projects[0].returns=[c+5,c+5,c+5,c+5,c+5,-L-3];tests[0].signals=[0,0,0,0,0,0];tests[1].signals=[0,1,0,1,0,1];tests[3].cost+=2}
 else if(family==="expensive"){tests.forEach(t=>t.cost+=3);limit=1}
 else if(family!=="screenable")throw Error("family");
 return {projects,tests,limit,gain:2,priors:[[8,6,4,2,2,2],[2,2,4,4,6,6],[4,4,2,6,4,4]],midpoint:[5,4,4,3,4,4],outside:[1,1,1,1,10,10]}
}

function masks(e,K,B,i){let p=e.projects[i],s=0,r=0;if(p.cost>K)throw Error("unaffordable");for(let w=0;w<p.returns.length;w++)if(B&(1<<w)){if(K-p.cost+p.returns[w]<0)r|=1<<w;else if(p.returns[w]>=p.cost+e.gain)s|=1<<w}return {s,r}}

function partition(e,B,i){let map=new Map(),t=e.tests[i];for(let w=0;w<t.signals.length;w++)if(B&(1<<w)){let a=t.signals[w];map.set(a,(map.get(a)||0)|(1<<w))}return [...map].sort((a,b)=>a[0]-b[0])}

function prune(points){let unique=new Map();for(let p of points){let k=p.s+","+p.r;if(!unique.has(k))unique.set(k,p)}let rows=[...unique.values()];return rows.filter(a=>!rows.some(b=>a!==b&&(a.s&b.s)===a.s&&(a.r&b.r)===b.r))}

function solve(e,K0,allowTests=true){let memo=new Map(),nodes=0,n=e.priors[0].length;
 function go(K,B,U,left){let key=[K,B,U,left].join(",");if(memo.has(key))return memo.get(key);nodes++;let all=[{s:0,r:0,tree:{kind:"abstain"}}];
 for(let i=0;i<e.projects.length;i++)if(e.projects[i].cost<=K){let a=masks(e,K,B,i);all.push({...a,tree:{kind:"buy",i}})}
 if(allowTests&&left>0)for(let i=0;i<e.tests.length;i++)if(!(U&(1<<i))&&e.tests[i].cost<=K){let groups=partition(e,B,i);if(groups.length<2)continue;let agg=[{s:0,r:0,by:{}}];for(let [obs,mask] of groups){let next=[];for(let a of agg)for(let b of go(K-e.tests[i].cost,mask,U|(1<<i),left-1))next.push({s:a.s|b.s,r:a.r|b.r,by:{...a.by,[obs]:b.tree}});agg=prune(next)}for(let a of agg)all.push({s:a.s,r:a.r,tree:{kind:"test",i,by:a.by}})}
 let out=prune(all);memo.set(key,out);return out
 }return {points:go(K0,(1<<n)-1,0,e.limit),nodes}}

function brute(e,K0){function go(K,worlds,used,left){let points=[[0,0]];for(let i=0;i<e.projects.length;i++){const p=e.projects[i];if(p.cost>K)continue;let s=0,r=0;for(const w of worlds){if(K-p.cost+p.returns[w]<0)r+=2**w;else if(p.returns[w]-p.cost>=e.gain)s+=2**w}points.push([s,r])}
 if(left)for(let i=0;i<e.tests.length;i++){let t=e.tests[i];if(used.includes(i)||t.cost>K)continue;let groups={};for(let w of worlds)(groups[t.signals[w]]??=[]).push(w);let acc=[[0,0]];for(let ws of Object.values(groups)){let next=[];for(let a of acc)for(let b of go(K-t.cost,ws,[...used,i],left-1))next.push([a[0]+b[0],a[1]+b[1]]);acc=next}points.push(...acc)}
 return [...new Set(points.map(x=>x.join(",")))].map(x=>x.split(",").map(Number))}
 let a=go(K0,e.priors[0].map((_,i)=>i),[],e.limit);return a.filter(x=>!a.some(y=>(x[0]&y[0])===x[0]&&(x[1]&y[1])===y[1]&&(x[0]!==y[0]||x[1]!==y[1]))).sort((a,b)=>a[0]-b[0]||a[1]-b[1])}

function replay(e,K0,tree){let s=0,r=0,spent=[],actions=[];for(let w=0;w<e.priors[0].length;w++){let K=K0,U=new Set(),node=tree,used=0,history=[];while(node&&node.kind==="test"){let i=node.i,t=e.tests[i];if(!t||U.has(i)||U.size>=e.limit||K<t.cost)throw Error("illegal test");K-=t.cost;used+=t.cost;U.add(i);let signal=t.signals[w];history.push([i,signal]);node=node.by[signal]}if(!node)throw Error("missing branch");if(node.kind==="buy"){let p=e.projects[node.i];if(!p||p.cost>K)throw Error("illegal buy");K+=p.returns[w]-p.cost;if(K<0)r|=1<<w;else if(p.returns[w]>=p.cost+e.gain)s|=1<<w}else if(node.kind!=="abstain")throw Error("illegal leaf");spent.push(used);actions.push(history)}return {s,r,spent,actions}}

function score(e,p,priors){return {S:Math.min(...priors.map(w=>mass(w,p.s))),R:Math.max(...priors.map(w=>mass(w,p.r)))}}

function select(e,points,priors,budget){let best=null;for(const p of points){let {S,R}=score(e,p,priors);if(R>budget)continue;if(!best||S>best.S||S===best.S&&R<best.R){best={...p,S,R}}}if(!best)throw Error("no abstain");return best}

function metrics(e,p,K){let v=replay(e,K,p.tree);if(v.s!==p.s||v.r!==p.r)throw Error("replay mismatch");return {s:p.s,r:p.r,nominalS:mass(e.priors[0],p.s),nominalR:mass(e.priors[0],p.r),worstS:score(e,p,e.priors).S,worstR:score(e,p,e.priors).R,midS:mass(e.midpoint,p.s),midR:mass(e.midpoint,p.r),outsideS:mass(e.outside,p.s),outsideR:mass(e.outside,p.r),testCostNominal:v.spent.reduce((a,c,w)=>a+c*e.priors[0][w],0),tree:p.tree}}

function run(){let checks=0,bruteCases=0,witnesses=0,nodes=0;const check=(v,msg)=>{checks++;if(!v)throw Error("check "+checks+" "+msg)};
 for(let seed=1;seed<=40;seed++){let rand=rng(seed*12371),e={gain:1,limit:2,priors:[[1,2,1]],projects:Array.from({length:2},()=>({cost:1+rand()%4,returns:Array.from({length:3},()=>Number(rand()%9)-3)})),tests:Array.from({length:2},()=>({cost:rand()%3,signals:Array.from({length:3},()=>rand()%2)}))};for(let K=0;K<=6;K++){let o=solve(e,K);let p=o.points.map(x=>[x.s,x.r]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);check(JSON.stringify(p)===JSON.stringify(brute(e,K)),"brute "+seed+" "+K);for(let x of o.points){let r=replay(e,K,x.tree);check(r.s===x.s&&r.r===x.r,"witness");witnesses++}bruteCases++}}
 let rows=[],thresholds=[],families=["screenable","aliased","alternative","hazard","expensive"];for(let fi=0;fi<families.length;fi++)for(let j=0;j<8;j++){let family=families[fi],seed=73001+fi*100+j,e=gen(seed,family),previous=[0,0,0],min={withInfo:null,withoutInfo:null};for(let K=0;K<=20;K++){let o=solve(e,K),direct=solve(e,K,false);nodes+=o.nodes;for(let x of o.points){let r=replay(e,K,x.tree);check(r.s===x.s&&r.r===x.r,"main witness");witnesses++}
 let r0=select(e,o.points,e.priors,0),r3=select(e,o.points,e.priors,3),r6=select(e,o.points,e.priors,6),d0=select(e,direct.points,e.priors,0);
 let vals=[r0.S,r3.S,r6.S];check(vals.every((x,i)=>x>=previous[i]),"capital monotone");previous=vals;
 check(r0.S>=d0.S&&r6.S>=r3.S&&r3.S>=r0.S,"relaxation");
 if(min.withInfo===null&&r0.S>=12)min.withInfo=K;if(min.withoutInfo===null&&d0.S>=12)min.withoutInfo=K;
 if([4,6,8,10,12].includes(K)){let nominal=select(e,o.points,[e.priors[0]],3),robust=r3,direct3=select(e,direct.points,e.priors,3),policies={};for(const [name,p] of [["nominal",nominal],["robust",robust],["direct",direct3],["safe",r0]]){let v=metrics(e,p,K);if(name!=="nominal"){check(v.worstR<=(name==="safe"?0:3),"risk");check(v.midR<=v.worstR&&v.midS>=v.worstS,"convex midpoint")}policies[name]=v}
 rows.push({family,seed,K,frontierSize:o.points.length,policies})}}
 thresholds.push({family,seed,...min})}
 const summary={};for(const family of [...families,"all"]){const rs=rows.filter(r=>family==="all"||r.family===family),ss={cases:rs.length,policies:{}};for(const name of ["nominal","robust","direct","safe"]){const ps=rs.map(x=>x.policies[name]);ss.policies[name]={worstS:ps.reduce((s,p)=>s+p.worstS,0),nominalS:ps.reduce((s,p)=>s+p.nominalS,0),midS:ps.reduce((s,p)=>s+p.midS,0),midR:ps.reduce((s,p)=>s+p.midR,0),violations:ps.filter(p=>p.worstR>3).length,midViolations:ps.filter(p=>p.midR>3).length,outsideViolations:ps.filter(p=>p.outsideR>3).length,testCostMass:ps.reduce((s,p)=>s+p.testCostNominal,0),tested:ps.filter(p=>p.tree.kind==="test").length}}ss.infoStrict=rs.filter(x=>x.policies.robust.worstS>x.policies.direct.worstS).length;summary[family]=ss}
 return {schema:"gal07.ambiguity.v1",checks,bruteCases,witnesses,nodes,grid:thresholds.length*21,cases:rows.length,thresholds,summary,rows,scope:"synthetic deterministic policies; known finite scenarios, fixed prior chosen once, upfront paid tests, terminal project, no controller work or native timing"}
}

console.log(JSON.stringify(run(),null,2));
