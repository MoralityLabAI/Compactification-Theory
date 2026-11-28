import {gen,mass,solve,select,metrics} from "./gal07_ambiguity.mjs";

function distributions(W,n){if(n===1)return [[W]];let out=[];for(let i=0;i<=W;i++)for(const rest of distributions(W-i,n-1))out.push([i,...rest]);return out}

function tvMass(p,q){return p.reduce((s,x,i)=>s+Math.abs(x-q[i]),0)/2}

function eventBounds(weights,mask,rho){let W=weights.reduce((s,x)=>s+x,0),full=(1<<weights.length)-1,p=mass(weights,mask);return {lo:mask===full?W:Math.max(0,p-rho),hi:mask===0?0:Math.min(W,p+rho)}}

function tvScore(e,p,rho){return {S:Math.min(...e.priors.map(w=>eventBounds(w,p.s,rho).lo)),R:Math.max(...e.priors.map(w=>eventBounds(w,p.r,rho).hi))}}

function selectTV(e,points,budget,rho){let best=null;for(let p of points){let {S,R}=tvScore(e,p,rho);if(R<=budget&&(!best||S>best.S||S===best.S&&R<best.R))best={...p,S,R}}if(!best)throw Error("no abstain");return best}

function ceilDiv(a,b){return (a+b-1n)/b}

function auditMin(N,p,e){if(N<1n||p<=0n||e<0n)throw Error("domain");return e>=p?0n:ceilDiv(N*(p-e),p)}

function auditBrute(N,p,e){for(let m=0n;m<=N;m++)if(p*(N-m)<=e*N)return m;throw Error("unreachable")}

function runRisk(){let checks=0,tvCases=0,auditCases=0;const check=v=>{checks++;if(!v)throw Error("check "+checks)};
 const D=distributions(6,3);for(const p of D)for(let rho=0;rho<=6;rho++){let Q=D.filter(q=>tvMass(p,q)<=rho);for(let mask=0;mask<8;mask++){let z=eventBounds(p,mask,rho),m=Q.map(q=>mass(q,mask));check(z.lo===Math.min(...m)&&z.hi===Math.max(...m));tvCases++}}
 for(let N=1n;N<=64n;N++)for(let p=1n;p<=8n;p++)for(let e=0n;e<=8n;e++){let m=auditMin(N,p,e);check(m===auditBrute(N,p,e));check(p*(N-m)<=e*N);if(m>0n)check(p*(N-m+1n)>e*N);auditCases++}
 let auditProfiles=[];for(let k of [4,8,16,24,32,40]){let N=2n**BigInt(k),m=auditMin(N,9n,1n),safe=auditMin(N,9n,0n);auditProfiles.push({addressBits:k,cases:String(N),probes90Accept10False:String(m),probesZeroFalse:String(safe),seedWithUnitProbeAndBuild10:String(m+10n)})}
 const priors=[[9,1,1,9],[1,9,9,1]],event=10;check(priors.every(p=>mass(p,event)===10));const rect={priorWeights:priors,branchMass:10,ruinMassByPrior:[10,10],fixedPriorRisk:"1/2",branchwiseAdversaryRisk:"9/10"};check(Math.max(...priors.map(p=>p[1]))+Math.max(...priors.map(p=>p[3]))===18);
 let rows=[],families=["screenable","aliased","alternative","hazard","expensive"];for(let fi=0;fi<families.length;fi++)for(let j=0;j<8;j++){let seed=73001+fi*100+j,e=gen(seed,families[fi]);for(let K of [4,6,8,10,12]){let o=solve(e,K),nominal=select(e,o.points,[e.priors[0]],6),robust=select(e,o.points,e.priors,6),tv=selectTV(e,o.points,6,2);let policies={};for(const [name,p] of [["nominal",nominal],["robust",robust],["tv",tv]]){let z=metrics(e,p,K);let vs=tvScore(e,p,2);if(name==="robust")check(z.worstR<=6);if(name==="tv")check(vs.R<=6);policies[name]={worstS:z.worstS,worstR:z.worstR,midS:z.midS,midR:z.midR,outsideS:z.outsideS,outsideR:z.outsideR,tvS:vs.S,tvR:vs.R}}check(policies.robust.worstS>=policies.tv.worstS);rows.push({family:families[fi],seed,K,policies})}}
 const summary={};for(const name of ["nominal","robust","tv"]){let ps=rows.map(r=>r.policies[name]);summary[name]={worstS:ps.reduce((s,p)=>s+p.worstS,0),tvS:ps.reduce((s,p)=>s+p.tvS,0),nonzeroRisk:ps.filter(p=>p.worstR>0).length,hullViolations:ps.filter(p=>p.worstR>6).length,tvViolations:ps.filter(p=>p.tvR>6).length,outsideViolations:ps.filter(p=>p.outsideR>6).length}}
 return {schema:"gal07.risk.v1",checks,tvCases,auditCases,auditProfiles,rect,summary,cases:rows.length,rows,scope:"exact finite TV event envelopes, hard audit-cap formula, same-grid 25% risk sensitivity; model ambiguity only, no controller cost"}
}

console.log(JSON.stringify(runRisk(),null,2));
