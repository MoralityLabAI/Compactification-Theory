import {readFileSync} from "node:fs";
import {gen,solve,replay,forward} from "./gal05_portfolio.mjs";

function validate(packet){if(packet.schema!=="gal05.packet.v1")throw Error("packet schema");const {env:e,capital}=packet;if(!e||!Array.isArray(e.projects)||e.projects.length<1||e.projects.length>12)throw Error("projects");const integer=(x,l,h)=>Number.isSafeInteger(x)&&x>=l&&x<=h;let n=e.projects.length;if(!integer(e.goal,0,n-1)||!integer(e.H,1,32)||!integer(e.cap,0,128)||!integer(e.fee,0,1000)||!integer(capital,0,1000000))throw Error("resource contract");for(const p of e.projects)if(!p||!integer(p.cost,0,100000)||!integer(p.yield,0,100000)||!integer(p.size,1,128)||!integer(p.pre,0,(1<<n)-1))throw Error("project contract");return e}

function packet(seed,family,cap,fee,capital){const p={schema:"gal05.packet.v1",capital,env:gen(seed,family,cap,fee)};validate(p);return p}

function validPlan(e,p){return Array.isArray(p)&&p.length<=e.H&&p.every(x=>Number.isInteger(x)&&x>=-1&&x<e.projects.length)}

function grade(p,r){const e=validate(p),o=solve(e);if(!r||r.schema!=="gal05.response.v1"||typeof r.reachable!=="boolean"||!(r.minimumSeed===null||Number.isSafeInteger(r.minimumSeed)&&r.minimumSeed>=0)||!validPlan(e,r.plan)||!validPlan(e,r.minimumPlan))return {schema:"gal05.grade.v1",pass:false,format:false};const expected=o.seed!==null&&p.capital>=o.seed;const reachableCorrect=r.reachable===expected;const witnessCorrect=r.reachable?replay(e,p.capital,r.plan).ok:r.plan.length===0&&!forward(e,p.capital);const seedCorrect=r.minimumSeed===o.seed;const minimumWitnessCorrect=r.minimumSeed===null?r.minimumPlan.length===0&&o.seed===null:replay(e,r.minimumSeed,r.minimumPlan).ok;const oracleAudit=o.seed===null?!forward(e,e.projects.reduce((s,p)=>s+p.cost,0)+e.H*e.fee):forward(e,o.seed)&&!forward(e,o.seed-1)&&replay(e,o.seed,o.plan).ok;if(!oracleAudit)throw Error("oracle audit");return {schema:"gal05.grade.v1",format:true,reachableCorrect,witnessCorrect,seedCorrect,minimumWitnessCorrect,pass:reachableCorrect&&witnessCorrect&&seedCorrect&&minimumWitnessCorrect}}

function answer(p){let o=solve(p.env),reachable=o.seed!==null&&p.capital>=o.seed;return {schema:"gal05.response.v1",reachable,plan:reachable?o.plan:[],minimumSeed:o.seed,minimumPlan:o.plan??[]}}

function eInvalid(p){return p.env.projects.length}

function testGym(){let checks=0;const check=v=>{checks++;if(!v)throw Error("check "+checks)};for(const family of ["independent","ladder","dependencies","distractors","short"])for(let seed=71;seed<=80;seed++)for(const capital of [0,12,100]){let p=packet(seed,family,6,1,capital),r=answer(p);check(grade(p,r).pass);check(!grade(p,{...r,minimumSeed:r.minimumSeed===null?0:r.minimumSeed+1}).pass);check(!grade(p,{...r,reachable:!r.reachable}).pass);check(!grade(p,{...r,plan:[eInvalid(p)]}).pass)}let impossible={schema:"gal05.packet.v1",capital:99,env:{projects:[{cost:1,yield:0,size:3,pre:0}],goal:0,H:2,cap:2,fee:0}};check(grade(impossible,answer(impossible)).pass);check(!grade(impossible,{...answer(impossible),minimumSeed:0}).pass);return {schema:"gal05.gym_selftest.v1",checks,scope:"public packet and private recomputed exact grader, no live model calls"}}

const [command,...args]=process.argv.slice(2);
let result;
if(command==="emit"){
 if(args.length!==5)throw Error("emit SEED FAMILY CAP FEE CAPITAL");
 const [seed,family,cap,fee,capital]=args;
 result=packet(Number(seed),family,Number(cap),Number(fee),Number(capital));
}else if(command==="grade"){
 if(args.length!==2)throw Error("grade PACKET.json RESPONSE.json");
 result=grade(JSON.parse(readFileSync(args[0],"utf8")),JSON.parse(readFileSync(args[1],"utf8")));
}else if(command==="selftest"){
 result=testGym();
}else throw Error("emit | grade | selftest");
console.log(JSON.stringify(result,null,2));
