function rng(seed){let x=seed>>>0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return x>>>0}}

function gen(seed,family,cap,fee){let rnd=rng(seed),p=Array.from({length:7},(_,i)=>({cost:3+rnd()%22,yield:rnd()%8,size:1+rnd()%3,pre:0}));let goal=6,H=7+rnd()%5;p[goal]={cost:42+rnd()%40,yield:0,size:2,pre:0};
if(family==="ladder"){for(let i=0;i<4;i++)p[i]={cost:4*2**i+rnd()%3,yield:3*2**i,size:1,pre:i?1<<(i-1):0};p[goal].pre=8;p[goal].cost=65+rnd()%30;H=9+rnd()%4}
else if(family==="dependencies"){p[1].pre=1;p[3].pre=1<<2;p[goal].pre=(1<<1)|(1<<3);for(let i=0;i<4;i++)p[i].size=1;H=9+rnd()%4}
else if(family==="distractors"){p[0]={cost:5,yield:6,size:4,pre:0};p[1]={cost:12,yield:1,size:2,pre:0};p[goal].pre=2;p[goal].cost=28+rnd()%15;H=6+rnd()%4}
else if(family==="short"){p[0]={cost:3,yield:2,size:1,pre:0};p[1]={cost:8,yield:9,size:1,pre:1};p[goal].pre=2;p[goal].cost=23+rnd()%12;H=4+rnd()%3}
else if(family!=="independent")throw Error("family");
return {projects:p,goal,H,cap,fee}}

function stats(e){let N=1<<e.projects.length,sizes=Array(N).fill(0),income=Array(N).fill(0);for(let m=1;m<N;m++){let bit=m&-m,i=31-Math.clz32(bit);sizes[m]=sizes[m^bit]+e.projects[i].size;income[m]=income[m^bit]+e.projects[i].yield}return {N,sizes,income}}

function legal(e,s,m){let a=[-1];for(let i=0;i<e.projects.length;i++){let p=e.projects[i];if(!(m&(1<<i))&&(m&p.pre)===p.pre&&s.sizes[m]+p.size<=e.cap)a.push(i)}return a}

function transition(e,s,m,i){let z=i<0?m:m|(1<<i);return {mask:z,cost:e.fee+(i<0?0:e.projects[i].cost),income:s.income[z]}}

function solve(e){const s=stats(e),goal=1<<e.goal;let next=Array.from({length:s.N},(_,m)=>m&goal?0:Infinity),layers=[next],actions=[];
for(let t=e.H-1;t>=0;t--){let row=Array(s.N).fill(Infinity),a=Array(s.N).fill(-2);for(let m=0;m<s.N;m++){if(s.sizes[m]>e.cap)continue;if(m&goal){row[m]=0;continue}for(const i of legal(e,s,m)){let z=transition(e,s,m,i),need=z.cost+Math.max(0,next[z.mask]-z.income);if(need<row[m]){row[m]=need;a[m]=i}}}next=row;layers.push(row);actions.push(a)}
if(!Number.isFinite(next[0]))return {seed:null,plan:null};
let m=0,k=next[0],plan=[];for(let t=0;t<e.H&&!(m&goal);t++){let i=actions[e.H-1-t][m],z=transition(e,s,m,i);plan.push(i);if(k<z.cost)throw Error("bad certificate");k+=z.income-z.cost;m=z.mask}if(!(m&goal))throw Error("no goal");return {seed:next[0],plan}}

function replay(e,seed,plan){let cash=seed,m=0,rows=[],goal=1<<e.goal;const s=stats(e);for(let t=0;t<plan.length;t++){if(t>=e.H||m&goal||!legal(e,s,m).includes(plan[t]))return {ok:false,reason:"illegal"};let z=transition(e,s,m,plan[t]);if(cash<z.cost)return {ok:false,reason:"cash"};cash+=z.income-z.cost;m=z.mask;rows.push({t,action:plan[t],cost:z.cost,income:z.income,cash,mask:m})}return {ok:!!(m&goal),cash,rows}}

function forward(e,seed){if(seed<0)return false;const s=stats(e);let dp=new Map([[0,seed]]);for(let t=0;t<e.H;t++){let q=new Map();for(const [m,cash] of dp){for(const i of legal(e,s,m)){let z=transition(e,s,m,i);if(cash<z.cost)continue;let k=cash-z.cost+z.income;if(z.mask&(1<<e.goal))return true;if(!q.has(z.mask)||q.get(z.mask)<k)q.set(z.mask,k)}}dp=q}return false}

function brute(e,cash,t=0,m=0){if(m&(1<<e.goal))return true;if(t===e.H)return false;let sz=0;for(let j=0;j<e.projects.length;j++)if(m&(1<<j))sz+=e.projects[j].size;for(let i=-1;i<e.projects.length;i++){if(i>=0&&((m&(1<<i))||(m&e.projects[i].pre)!==e.projects[i].pre||sz+e.projects[i].size>e.cap))continue;let cost=e.fee+(i<0?0:e.projects[i].cost);if(cost>cash)continue;let mask=i<0?m:m|(1<<i),income=0;for(let j=0;j<e.projects.length;j++)if(mask&(1<<j))income+=e.projects[j].yield;if(brute(e,cash-cost+income,t+1,mask))return true}return false}

function mandatory(e){let m=1<<e.goal;for(let k=0;k<e.projects.length;k++)for(let j=0;j<e.projects.length;j++)if(m&(1<<j))m|=e.projects[j].pre;return m}

function policy(e,seed,mode){let s=stats(e),m=0,cash=seed,plan=[],work=0,need=mandatory(e),g=1<<e.goal;
for(let t=0;t<e.H&&!(m&g);t++){let choices=legal(e,s,m).filter(i=>transition(e,s,m,i).cost<=cash);work+=e.projects.length;if(!choices.length)break;let i=-1;if(choices.includes(e.goal))i=e.goal;else if(mode!=="wait"){let candidates=choices.filter(i=>i>=0);if(mode==="ancestors")candidates=candidates.filter(i=>need&(1<<i));if(mode==="reserve"||mode==="reserve_value")candidates=candidates.filter(i=>s.sizes[m|(1<<i)|need]<=e.cap);let rem=e.H-t;function value(i){const p=e.projects[i];if(mode==="income")return p.yield;if(mode==="ratio")return p.yield/(p.cost+e.fee);if(mode==="ancestors")return -p.cost;if(mode==="reserve")return (need&(1<<i)?10000:0)+p.yield/(p.cost+e.fee);if(mode==="reserve_value")return (need&(1<<i)?10000:0)+(rem*p.yield-p.cost)/(p.size+1);throw Error("mode")}
candidates=candidates.filter(i=>(need&(1<<i))||e.projects[i].yield*rem>e.projects[i].cost);work+=candidates.length;candidates.sort((a,b)=>value(b)-value(a)||a-b);if(candidates.length)i=candidates[0]}
if(!choices.includes(i))break;let z=transition(e,s,m,i);cash+=z.income-z.cost;m=z.mask;plan.push(i)}
let r=replay(e,seed,plan);return {success:r.ok,ticks:plan.length,plan,work,cash:r.ok?r.cash:null}}

function run(){let checks=0;const check=x=>{checks++;if(!x)throw Error("check "+checks)};let tiny=0;
for(let seed=1;seed<=96;seed++){let rnd=rng(seed*0x731),p=Array.from({length:3},(_,i)=>({cost:1+rnd()%5,yield:rnd()%4,size:1+rnd()%2,pre:i===2&&rnd()%2?1:0})),e={projects:p,goal:2,H:1+rnd()%4,cap:1+rnd()%5,fee:rnd()%2},o=solve(e);for(let K=0;K<=16;K++){let v=brute(e,K);check(v===(o.seed!==null&&K>=o.seed));check(v===forward(e,K));tiny++}if(o.seed!==null)check(replay(e,o.seed,o.plan).ok)}
const modes=["wait","income","ratio","ancestors","reserve","reserve_value"],families=["independent","ladder","dependencies","distractors","short"],cal=Object.fromEntries(modes.map(x=>[x,{success:0,ticks:0,work:0}]));
function episodes(split){let rows=[],fs=split==="cal"?families.slice(0,4):families;for(let fi=0;fi<fs.length;fi++)for(let j=0;j<(split==="cal"?5:8);j++)for(const cap of [6,9])for(const fee of [0,1]){let seed=(split==="cal"?10001:20001)+100*fi+j,e=gen(seed,fs[fi],cap,fee),o=solve(e);if(o.seed===null){check(!forward(e,100000));rows.push({family:fs[fi],seed,cap,fee,minimum:null});continue}check(replay(e,o.seed,o.plan).ok);check(forward(e,o.seed));check(!forward(e,o.seed-1));for(const extra of [0,10]){let K=o.seed+extra,policies={};for(const mode of modes){let v=policy(e,K,mode);check(!v.success||replay(e,K,v.plan).ok);policies[mode]=v;if(split==="cal"){cal[mode].success+=+v.success;cal[mode].ticks+=v.success?v.ticks:0;cal[mode].work+=v.work}}rows.push({family:fs[fi],seed,cap,fee,minimum:o.seed,budget:K,extra,oraclePlan:o.plan,policies})}}return rows}
let calibration=episodes("cal"),selected=modes.slice().sort((a,b)=>cal[b].success-cal[a].success||cal[a].ticks-cal[b].ticks||cal[a].work-cal[b].work)[0],rows=episodes("test");
const summary={};for(const f of [...families,"all"]){let rs=rows.filter(r=>r.minimum!==null&&(f==="all"||r.family===f)),ss={cases:rs.length,impossible:rows.filter(r=>r.minimum===null&&(f==="all"||r.family===f)).length,atMinimum:rs.filter(r=>r.extra===0).length,policies:{}};for(const mode of modes){let successes=rs.filter(r=>r.policies[mode].success);ss.policies[mode]={success:successes.length,atMinimum:successes.filter(r=>r.extra===0).length,meanTicks:successes.length?successes.reduce((a,r)=>a+r.policies[mode].ticks,0)/successes.length:null,work:rs.reduce((a,r)=>a+r.policies[mode].work,0)}}summary[f]=ss}
let trap={projects:[{cost:2,yield:8,size:4,pre:0},{cost:2,yield:2,size:1,pre:0},{cost:9,yield:0,size:2,pre:2}],goal:2,H:6,cap:4,fee:0},o=solve(trap),trapResults=Object.fromEntries(modes.map(m=>[m,policy(trap,o.seed,m)]));check(o.seed===2);check(!trapResults.income.success);check(trapResults.reserve.success);
return {schema:"gal05.portfolio.v1",checks,tinyBruteCases:tiny,calibrationCases:calibration.filter(r=>r.minimum!==null).length,testCases:summary.all.cases,selected,calibration:cal,summary,trap:{env:trap,oracle:o,policies:trapResults},rows,scope:"synthetic stationary additive independent savings, known costs/horizon, persistent investments, one build per tick, no borrowing, capacity and fixed action fee; planning scan counts separate; oracle computation uncharged"}}

console.log(JSON.stringify(run(),null,2));
