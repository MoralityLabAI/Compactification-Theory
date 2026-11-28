function parity(x){let p=0;while(x){x&=x-1;p^=1}return p}

function xorRank(rows){let piv=new Map();for(let v of rows){while(v){let k=31-Math.clz32(v);if(piv.has(k))v^=piv.get(k);else{piv.set(k,v);break}}}return piv.size}

function discover(rows,n){let piv=new Map(),tests=0,xors=0;for(const row of rows){let v=row;while(v){tests++;let k=31-Math.clz32(v);if(piv.has(k)){v^=piv.get(k);xors+=n}else{piv.set(k,v);break}}}return {basis:[...piv.values()],tests,xors}}

function annihilator(rows,n){for(let z=1;z<(1<<n);z++)if(rows.every(a=>parity(a&z)===0))return z;return null}

function requiredTestCount(n,model){return model==="explicit"?2**n:model==="linear"?n:model==="affine"?n+1:null}

function evalAffine(code,x){return parity(code&x)^((code>>>16)&1)}

function cost(n,r,q,C){let u=2*n*r+n,v=2*r+n;let delta=u-q*v;if(delta<=0)return C===0&&delta===0?1:null;return Math.max(1,Math.ceil(q*C/delta))}

function campaign(){let checks=0;let check=x=>{checks++;if(!x)throw Error("check "+checks)};let rowCases=0,affineCases=0,codeCases=0;
for(let n=1;n<=4;n++){let N=1<<n;for(let q=0;q<(1<<N);q++){let rows=[];for(let a=0;a<N;a++)if(q&(1<<a))rows.push(a);let rank=xorRank(rows),z=annihilator(rows,n);check((rank===n)===(z===null));if(z!==null){check(rows.every(a=>parity(a&z)===0));check(Array.from({length:N},(_,a)=>parity(a&z)).some(Boolean))}rowCases++}}
for(let n=1;n<=8;n++){let N=1<<n;for(let a=0;a<N;a++)for(let b=0;b<2;b++){let f=x=>parity(a&x)^b;let bias=f(0),coeff=0;for(let k=0;k<n;k++)coeff|=(f(1<<k)^bias)<<k;check(coeff===a&&bias===b);check(Array.from({length:N},(_,x)=>f(x)===(parity(coeff&x)^bias)).every(Boolean));affineCases++}
for(let a=1;a<N;a++){let f=x=>parity(a&x);let tests=Array.from({length:n},(_,i)=>1<<i);check(tests.some(x=>f(x)));codeCases++}}
let profiles=[];for(let n of [4,8,16,24,32,40]){let explicit=2**n,linear=n,affine=n+1,certBits=n+1;profiles.push({n,explicit,linear,affine,certBits})}
return {schema:"gal08.structural.v1",checks,rowCases,affineCases,codeCases,profiles,scope:"exact finite-field observations and trusted affine/linear promise; no claim that arbitrary programs satisfy the promise"}}

console.log(JSON.stringify(campaign(),null,2));
