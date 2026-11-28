function parity(x){let z=0;while(x){z^=1;x&=x-1}return z}

function dot(a,x){return parity(a&x)}

function rank(vs){const piv=new Map();for(let v of vs){while(v){const i=31-Math.clz32(v);if(piv.has(i))v^=piv.get(i);else{piv.set(i,v);break}}}return piv.size}

function allSpans(r){let key=a=>a.join(",");const seen=new Map([["0",[0]]]),queue=[[0]];for(let i=0;i<queue.length;i++){const a=queue[i],s=new Set(a);for(let v=1;v<(1<<r);v++){if(s.has(v))continue;const b=[...a,...a.map(x=>x^v)].sort((a,b)=>a-b),k=key(b);if(!seen.has(k)){seen.set(k,b);queue.push(b)}}}return queue}

function tensorFeatures(K,L){return Array.from({length:K*L},(_,i)=>1<<i)}

function separableFeatures(K,L){return Array.from({length:K*L},(_,i)=>(1<<Math.floor(i/L))|(1<<(K+i%L)))}

function answer(a,world){return world.reduce((p,x)=>p^dot(a,x),0)}

function online(features,world){const piv=new Map();let probes=0,answers=[];for(const a of features){let v=a,z=0;while(v){let i=31-Math.clz32(v);if(piv.has(i)){const [w,y]=piv.get(i);v^=w;z^=y}else{const y=answer(a,world);probes+=world.length;piv.set(i,[v,y^z]);z=y;break}}answers.push(z)}return {probes,answers}}

function threshold(r,b,q){r=BigInt(r);b=BigInt(b);q=BigInt(q);if(r<1n||b<1n||q<=1n||q>=b+1n)throw Error("invalid domain");let d=b+1n-q;return (q*b*r+d-1n)/d}

function run(){
 let checks=0;const check=(v)=>{checks++;if(!v)throw Error("failed "+checks)};
 let spans=0,witnesses=0;for(let r=1;r<=4;r++)for(const s of allSpans(r)){spans++;const basisRank=rank(s);check(s.length===2**basisRank);for(let a=0;a<(1<<r);a++){const z=Array.from({length:1<<r},(_,i)=>i).find(z=>s.every(v=>dot(z,v)===0)&&dot(z,a)===1);check((z===undefined)===s.includes(a));if(z!==undefined){witnesses++;for(let x=0;x<(1<<r);x++){check(s.every(v=>dot(v,x)===dot(v,x^z)));check(dot(a,x)!==dot(a,x^z))}}}}
 let cases=0;for(const [name,fs] of [["tensor",tensorFeatures(2,2)],["separable",separableFeatures(2,2)]])for(let x=0;x<16;x++)for(let y=0;y<16;y++)for(let i=0;i<64;i++){const req=[fs[i&3],fs[(i>>2)&3],fs[(i>>4)&3]];const out=online(req,[x,y]);check(out.probes===2*rank(req));check(out.answers.every((v,j)=>v===answer(req[j],[x,y])));cases++}
 let rankCases=0;for(let K=1;K<=4;K++)for(let L=1;L<=4;L++){check(rank(tensorFeatures(K,L))===K*L);check(rank(separableFeatures(K,L))===K+L-1);rankCases++}
 let additiveCases=0;for(let c1=0;c1<=20;c1++)for(let c2=0;c2<=20;c2++)for(let d1=1;d1<=10;d1++)for(let d2=1;d2<=10;d2++){let a=Math.ceil(c1/d1),b=Math.ceil(c2/d2),c=Math.ceil((c1+c2)/(d1+d2));check(c>=Math.min(a,b)&&c<=Math.max(a,b));additiveCases++}
 let profiles=[];for(let d=1;d<=8;d++){const r=8n**BigInt(d),s=1n+7n*BigInt(d);let t=threshold(r,8,2),u=threshold(s,8,2);check(7n*t>=16n*r&&7n*(t-1n)<16n*r);check(7n*u>=16n*s&&7n*(u-1n)<16n*s);profiles.push({depth:d,tensorRank:String(r),separableRank:String(s),tensor2x:String(t),separable2x:String(u)})}
 return {schema:"gal02.receipt.v1",checks,subspaces:spans,orthogonalWitnesses:witnesses,adaptiveBasisReplayCases:cases,rankCases,additiveCases,profiles,scope:"exact GF(2), slice-local oracle with free arithmetic and charged probes/output; no wall time or native model training"}
}

console.log(JSON.stringify(run(),null,2));
