import {readFileSync} from "node:fs";
import {gen,mass,solve,replay} from "./gal07_ambiguity.mjs";

function validate(p){const int=(x,a,b)=>Number.isSafeInteger(x)&&x>=a&&x<=b;if(!p||p.schema!=="gal07.packet.v1"||!p.env)throw Error("schema");let e=p.env;if(!Array.isArray(e.priors)||e.priors.length<1||e.priors.length>5||!Array.isArray(e.priors[0]))throw Error("priors");let n=e.priors[0].length;if(!int(n,1,6))throw Error("world count");let W=e.priors[0].reduce((s,x)=>s+x,0);if(!int(W,1,120)||!e.priors.every(a=>Array.isArray(a)&&a.length===n&&a.every(x=>int(x,0,120))&&a.reduce((s,x)=>s+x,0)===W))throw Error("weights");if(!int(p.capital,0,100)||!int(p.riskMass,0,W)||!int(p.radiusMass,0,W)||!int(e.gain,0,100)||!int(e.limit,0,3)||!Array.isArray(e.projects)||e.projects.length<1||e.projects.length>4||!Array.isArray(e.tests)||e.tests.length>5)throw Error("resources");for(let x of e.projects)if(!x||!int(x.cost,0,100)||!Array.isArray(x.returns)||x.returns.length!==n||!x.returns.every(y=>int(y,-100,100)))throw Error("project");for(let t of e.tests)if(!t||!int(t.cost,0,100)||!Array.isArray(t.signals)||t.signals.length!==n||!t.signals.every(y=>int(y,0,100)))throw Error("test");return {e,W,n}}

function packet(seed,family,capital,riskMass,radiusMass){const {midpoint,outside,...env}=gen(seed,family);let p={schema:"gal07.packet.v1",capital,riskMass,radiusMass,env};validate(p);return p}

function obj(p,s,r){const {e,W,n}=validate(p),full=(1<<n)-1;let S=Math.min(...e.priors.map(w=>s===full?W:Math.max(0,mass(w,s)-p.radiusMass))),R=Math.max(...e.priors.map(w=>r===0?0:Math.min(W,mass(w,r)+p.radiusMass)));return {S,R}}

function shape(e,t){let count=0;function go(x,d){if(!x||typeof x!=="object"||Array.isArray(x)||++count>512||d>e.limit)return false;if(x.kind==="abstain")return true;if(x.kind==="buy")return Number.isInteger(x.i)&&x.i>=0&&x.i<e.projects.length;if(x.kind!=="test"||d===e.limit||!Number.isInteger(x.i)||x.i<0||x.i>=e.tests.length||!x.by||typeof x.by!=="object"||Array.isArray(x.by))return false;let keys=Object.keys(x.by);return keys.length>=1&&keys.length<=e.priors[0].length&&keys.every(k=>/^(0|[1-9]\d*)$/.test(k)&&go(x.by[k],d+1))}return go(t,0)}

function optimum(p){let {e}=validate(p),o=solve(e,p.capital),best=null;for(let a of o.points){let z=obj(p,a.s,a.r);if(z.R<=p.riskMass&&(!best||z.S>best.S||z.S===best.S&&z.R<best.R))best={...a,...z}}return best}

function grade(p,r){let {e,W}=validate(p);if(!r||r.schema!=="gal07.response.v1"||!Number.isInteger(r.worstSuccessMass)||r.worstSuccessMass<0||r.worstSuccessMass>W||!Number.isInteger(r.worstRuinMass)||r.worstRuinMass<0||r.worstRuinMass>W||!shape(e,r.policy))return {schema:"gal07.grade.v1",format:false,pass:false};let v;try{v=replay(e,p.capital,r.policy)}catch{return {schema:"gal07.grade.v1",format:true,witness:false,pass:false}}let z=obj(p,v.s,v.r),o=optimum(p),witness=z.S===r.worstSuccessMass&&z.R===r.worstRuinMass,risk=z.R<=p.riskMass,optimal=z.S===o.S;return {schema:"gal07.grade.v1",format:true,witness,risk,optimal,pass:witness&&risk&&optimal}}

function answer(p){let a=optimum(p);return {schema:"gal07.response.v1",worstSuccessMass:a.S,worstRuinMass:a.R,policy:a.tree}}

function selftest(){let checks=0;const check=x=>{checks++;if(!x)throw Error("check "+checks)};for(const f of ["screenable","aliased","alternative","hazard","expensive"])for(let i=0;i<4;i++)for(let K of [4,8,12])for(let radius of [0,2]){let p=packet(74001+i,f,K,6,radius),r=answer(p);check(grade(p,r).pass);check(!grade(p,{...r,worstSuccessMass:(r.worstSuccessMass+1)%25}).pass);check(!grade(p,{...r,policy:{kind:"test",i:99,by:{}}}).pass)}
let p={schema:"gal07.packet.v1",capital:2,riskMass:1,radiusMass:0,env:{priors:[[9,1],[1,9]],gain:1,limit:0,projects:[{cost:2,returns:[5,-4]}],tests:[]}};let r={schema:"gal07.response.v1",worstSuccessMass:1,worstRuinMass:9,policy:{kind:"buy",i:0}},g=grade(p,r);check(g.witness&&!g.risk&&!g.pass);check(grade(p,answer(p)).pass);let q={...p,env:{...p.env,priors:[[10,0]]},radiusMass:1,riskMass:0};check(!grade(q,{...r,worstSuccessMass:9,worstRuinMass:1}).pass);return {schema:"gal07.gym_selftest.v1",checks,scope:"public finite ambiguity packet and exact private recomputed grader; no model calls"}}

const [command,...args]=process.argv.slice(2);let result;
if(command==="emit"){if(args.length!==5)throw Error("emit SEED FAMILY CAPITAL RISK_MASS RADIUS_MASS");result=packet(Number(args[0]),args[1],Number(args[2]),Number(args[3]),Number(args[4]))}
else if(command==="grade"){if(args.length!==2)throw Error("grade PACKET.json RESPONSE.json");result=grade(JSON.parse(readFileSync(args[0],"utf8")),JSON.parse(readFileSync(args[1],"utf8")))}
else if(command==="selftest")result=selftest();
else throw Error("emit | grade | selftest");
console.log(JSON.stringify(result,null,2));
