import {readFileSync} from "node:fs";

function gcd(a,b){while(b){[a,b]=[b,a%b]}return a}

function frac(n,d){const g=gcd(n,d);return [n/g,d/g]}

function cmp(a,b){return a[0]*b[1]-b[0]*a[1]}

function mixture(frontier,cap){let best=[0,1];for(const [s,r] of frontier)if(r<=cap&&cmp([s,1],best)>0)best=[s,1];for(const [s1,r1] of frontier)for(const [s2,r2] of frontier)if(r1<cap&&cap<r2){let v=frac(s1*(r2-cap)+s2*(cap-r1),r2-r1);if(cmp(v,best)>0)best=v}return best}

function bruteMix(frontier,cap,W){let best=[0,1];for(const [s1,r1] of frontier)for(const [s2,r2] of frontier)for(let d=1;d<=W;d++)for(let a=0;a<=d;a++)if(a*r1+(d-a)*r2<=cap*d){let v=frac(a*s1+(d-a)*s2,d);if(cmp(v,best)>0)best=v}return best}

function deficits(payoffs,due){return payoffs[0].map((_,w)=>{let cum=0,min=0;for(const t of [...new Set(due)].sort((a,b)=>a-b)){for(let i=0;i<due.length;i++)if(due[i]===t)cum+=payoffs[i][w];min=Math.min(min,cum)}return -min})}

function chanceReserve(ds,weights,budget){for(const k of [...new Set([0,...ds])].sort((a,b)=>a-b))if(ds.reduce((s,d,i)=>s+(d>k?weights[i]:0),0)<=budget)return k;throw Error("no reserve")}

function realizedRuin(payoffs,due,weights,K){let loss=0;for(let w=0;w<weights.length;w++){let cash=K;for(let t=1;t<=Math.max(...due);t++){let v=0;for(let i=0;i<due.length;i++)if(due[i]===t)v+=payoffs[i][w];cash+=v;if(cash<0){loss+=weights[w];break}}}return loss}

function run(stochastic){let checks=0,reserveCases=0,mixCases=0;const check=v=>{checks++;if(!v)throw Error("check "+checks)};
for(let a=-2;a<=2;a++)for(let b=-2;b<=2;b++)for(let c=-2;c<=2;c++)for(let d=-2;d<=2;d++)for(let t1=1;t1<=2;t1++)for(let t2=1;t2<=2;t2++)for(const weights of [[1,1],[1,2]]){let p=[[a,b],[c,d]],due=[t1,t2],ds=deficits(p,due),W=weights.reduce((x,y)=>x+y,0);for(let r=0;r<=W;r++){let k=chanceReserve(ds,weights,r);check(realizedRuin(p,due,weights,k)<=r);if(k>0)check(realizedRuin(p,due,weights,k-1)>r);reserveCases++}}
let W=4,points=[];for(let s=0;s<=W;s++)for(let r=0;r+s<=W;r++)points.push([s,r]);for(let i=0;i<points.length;i++)for(let j=i;j<points.length;j++)for(let k=j;k<points.length;k++)for(let cap=0;cap<=W;cap++){let f=[points[i],points[j],points[k],[0,0]],v=mixture(f,cap);check(cmp(v,bruteMix(f,cap,W))===0);mixCases++}
let improvement=0,total=0,records=[];for(const row of stochastic.rows){let best=mixture(row.frontier,1);let d=row.oracle[1];check(cmp(best,[d,1])>=0);if(cmp(best,[d,1])>0)improvement++;total+=best[0]/best[1];records.push({family:row.family,seed:row.seed,K:row.K,deterministic:d,randomizedMass:best})}
let hedge={sameMarginals:[[[-5,8],[-5,8]],[[-5,8],[8,-5]]].map(p=>({payoffs:p,deficits:deficits(p,[2,2]),safe:chanceReserve(deficits(p,[2,2]),[1,1],0)})),staggered:chanceReserve(deficits([[-5,8],[8,-5]],[1,2]),[1,1],0)};check(hedge.sameMarginals[0].safe===10&&hedge.sameMarginals[1].safe===0&&hedge.staggered===5);check(cmp(mixture([[0,0],[1,1]],0),[0,1])===0);
return {schema:"gal06.risk.v1",checks,reserveCases,mixCases,randomized:{cases:records.length,improved:improvement,totalSuccessMass:total,deterministicMass:stochastic.summary.all.oracleQuarter,records},hedge,scope:"exact weighted reserve and two-policy initial random mixtures; only the aggregate display sum uses floating point"}
}

console.log(JSON.stringify(run(JSON.parse(readFileSync(new URL("./gal06_stochastic_receipt.json",import.meta.url),"utf8"))),null,2));
