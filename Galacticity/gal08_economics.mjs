import {readFileSync} from "node:fs";

function first(C,u,v,q){C=BigInt(C);u=BigInt(u);v=BigInt(v);q=BigInt(q);const d=u-q*v;if(d<0n||d===0n&&C>0n)return null;if(d===0n)return "1";let m=(q*C+d-1n)/d;return String(m>0n?m:1n)}

function run(receipt){let checks=0;const check=x=>{checks++;if(!x)throw Error("check "+checks)};let cases=0;for(let C=0;C<=20;C++)for(let u=1;u<=16;u++)for(let v=0;v<=16;v++)for(let q=1;q<=4;q++){let m=first(C,u,v,q);if(m!==null){let k=Number(m);check(q*(C+k*v)<=k*u);if(k>1)check(q*(C+(k-1)*v)>(k-1)*u)}else check(u<q*v||u===q*v&&C>0);cases++}let rows=[];for(const p of receipt.profiles)for(const w of [1,2,4]){let C=p.constructionUnits+p.verificationUnits,u=w*p.gates+1,v=2*p.n+1,m=first(C,u,v,2);if(m!==null){let k=BigInt(m);check(2n*(BigInt(C)+k*BigInt(v))<=k*BigInt(u));if(k>1n)check(2n*(BigInt(C)+(k-1n)*BigInt(v))>(k-1n)*BigInt(u))}rows.push({n:p.n,gates:p.gates,gatePrice:w,setup:C,directPerUse:u,affinePerUse:v,first2x:m})}return {schema:"gal08.economics.v1",checks,cases,rows,scope:"declared scalar price only; fixed authenticated circuit, one-time checked factor form, dense affine evaluation; no timings or optimized algorithm lower bound"}}

console.log(JSON.stringify(run(JSON.parse(readFileSync(new URL("./gal08_certificates_receipt.json",import.meta.url),"utf8"))),null,2));
