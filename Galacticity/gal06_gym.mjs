import {readFileSync} from "node:fs";
import {gen,oracle,replay} from "./gal06_stochastic.mjs";

function validate(p){const int=(x,a,b)=>Number.isSafeInteger(x)&&x>=a&&x<=b;if(!p||p.schema!=="gal06.packet.v1"||!p.env)throw Error("packet schema");let e=p.env;if(!Array.isArray(e.weights)||e.weights.length<1||e.weights.length>5||!e.weights.every(w=>int(w,1,20)))throw Error("weights");let W=e.weights.reduce((s,x)=>s+x,0);if(!Array.isArray(e.projects)||e.projects.length<1||e.projects.length>5||!int(e.goal,0,e.projects.length-1)||!int(e.H,1,7)||!int(e.cap,0,32)||!int(p.capital,0,500)||!int(p.riskMass,0,W))throw Error("resource contract");for(let x of e.projects)if(!x||!int(x.cost,0,100)||!int(x.delay,1,e.H)||!int(x.size,1,10)||!int(x.pre,0,(1<<e.projects.length)-1)||!Array.isArray(x.returns)||x.returns.length!==e.weights.length||!x.returns.every(v=>int(v,-100,100)))throw Error("project contract");return {e,W}}

function packet(seed,family,capital,riskMass){let p={schema:"gal06.packet.v1",capital,riskMass,env:gen(seed,family)};validate(p);return p}

function treeShape(root,e){let n=0;function visit(x,k,d){if(!x||typeof x!=="object"||Array.isArray(x)||++n>4096||d>2*e.H+2||x.kind!==k&&!(k==="act"&&x.kind==="end"))return false;if(x.kind==="end")return true;if(x.kind==="act")return Number.isInteger(x.i)&&x.i>=-1&&x.i<e.projects.length&&visit(x.next,"chance",d+1);if(!x.by||typeof x.by!=="object"||Array.isArray(x.by))return false;let keys=Object.keys(x.by);return keys.length>=1&&keys.length<=e.weights.length&&keys.every(key=>/^(?:-?\d+(?:,-?\d+)*)?$/.test(key)&&visit(x.by[key],"act",d+1))}return visit(root,"chance",0)}

function grade(p,r){const {e,W}=validate(p);if(!r||r.schema!=="gal06.response.v1"||!Number.isInteger(r.successMass)||!Number.isInteger(r.ruinMass)||r.successMass<0||r.ruinMass<0||r.successMass+r.ruinMass>W||!treeShape(r.policy,e))return {schema:"gal06.grade.v1",format:false,pass:false};let v;try{v=replay(e,p.capital,r.policy)}catch{return {schema:"gal06.grade.v1",format:true,witness:false,pass:false}}let witness=v.s===r.successMass&&v.r===r.ruinMass,risk=v.r<=p.riskMass;let o=oracle(e,p.capital),opt=Math.max(...o.frontier.filter(x=>x.r<=p.riskMass).map(x=>x.s));return {schema:"gal06.grade.v1",format:true,witness,risk,optimal:v.s===opt,pass:witness&&risk&&v.s===opt}}

function answer(p){let o=oracle(p.env,p.capital).frontier.filter(x=>x.r<=p.riskMass).sort((a,b)=>b.s-a.s||a.r-b.r)[0];return {schema:"gal06.response.v1",successMass:o.s,ruinMass:o.r,policy:o.w}}

function selftest(){let checks=0;const check=x=>{checks++;if(!x)throw Error("check "+checks)};for(const f of ["stable","correlated","hedged","delayed","shift"])for(let i=0;i<4;i++)for(const K of [2,6,10])for(const r of [0,1]){let p=packet(71001+i,f,K,r),a=answer(p);check(grade(p,a).pass);check(!grade(p,{...a,successMass:(a.successMass+1)%5}).pass);check(!grade(p,{...a,policy:{kind:"chance",by:{}}}).pass)}let p={schema:"gal06.packet.v1",capital:1,riskMass:0,env:{weights:[1,1],goal:1,H:3,cap:2,projects:[{cost:1,delay:1,size:1,pre:0,returns:[10,-4]},{cost:5,delay:1,size:1,pre:0,returns:[0,0]}]}};check(grade(p,answer(p)).pass);let unsafe=answer({...p,riskMass:1});let g=grade(p,unsafe);check(g.witness&&!g.risk&&!g.pass);check(!grade(p,{...unsafe,ruinMass:0}).pass);return {schema:"gal06.gym_selftest.v1",checks,scope:"standalone public scenarios and private recomputed deterministic risk-constrained grader; no model calls"}}

const [command,...args]=process.argv.slice(2);let result;
if(command==="emit"){if(args.length!==4)throw Error("emit SEED FAMILY CAPITAL RISK_MASS");result=packet(Number(args[0]),args[1],Number(args[2]),Number(args[3]))}
else if(command==="grade"){if(args.length!==2)throw Error("grade PACKET.json RESPONSE.json");result=grade(JSON.parse(readFileSync(args[0],"utf8")),JSON.parse(readFileSync(args[1],"utf8")))}
else if(command==="selftest")result=selftest();
else throw Error("emit | grade | selftest");
console.log(JSON.stringify(result,null,2));
