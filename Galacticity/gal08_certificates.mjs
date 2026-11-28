function parity(x){let p=0;while(x){x&=x-1;p^=1}return p}

function evalGate(g,values,x){switch(g.op){case "in":return (x>>>g.i)&1;case "const":return g.v;case "xor":return values[g.a]^values[g.b];case "and":return values[g.a]&values[g.b];default:throw Error("gate")}}

function evaluate(c,x){let v=[];for(let g of c.gates)v.push(evalGate(g,v,x));return v[c.out]}

function wellFormed(c){let n=c.n;return Number.isInteger(n)&&n>=1&&n<=40&&Array.isArray(c.gates)&&c.gates.length>0&&Number.isInteger(c.out)&&c.out>=0&&c.out<c.gates.length&&c.gates.every((g,i)=>g&&(["in","const","xor","and"].includes(g.op))&&(g.op==="in"?Number.isInteger(g.i)&&g.i>=0&&g.i<n:g.op==="const"?g.v===0||g.v===1:Number.isInteger(g.a)&&Number.isInteger(g.b)&&g.a>=0&&g.b>=0&&g.a<i&&g.b<i))}

function affineStep(g,prev){if(g.op==="in")return [1n<<BigInt(g.i),0];if(g.op==="const")return [0n,g.v];let [a,b]=prev[g.a],[c,d]=prev[g.b];if(g.op==="xor")return [a^c,b^d];if(a===0n)return b?[c,d]:[0n,0];if(c===0n)return d?[a,b]:[0n,0];if(a===c)return b===d?[a,b]:[0n,0];return null}

function build(c){if(!wellFormed(c))throw Error("malformed circuit");let v=[],work=0;for(const g of c.gates){let row=affineStep(g,v);work+=g.op==="xor"||g.op==="and"?c.n+1:1;if(row===null)return {status:"unsupported",work};v.push(row)}return {status:"certified",work,certificate:v.map(([a,b])=>[String(a),b]),root:v[c.out].map(String)}}

function verify(c,cert,target){if(!wellFormed(c)||!Array.isArray(cert)||cert.length!==c.gates.length||!Array.isArray(target)||target.length!==2)return {ok:false,work:0};let work=0,prev=[];for(let i=0;i<cert.length;i++){let row=cert[i];if(!Array.isArray(row)||row.length!==2||typeof row[0]!=="string"||!/^(0|[1-9]\d*)$/.test(row[0])||row[0].length>13||(row[1]!==0&&row[1]!==1))return {ok:false,work};let a=BigInt(row[0]),b=row[1];if(a>=(1n<<BigInt(c.n)))return {ok:false,work};let g=c.gates[i];work+=c.n+1;let expected=affineStep(g,prev);work+=g.op==="xor"||g.op==="and"?c.n+1:1;if(expected===null||expected[0]!==a||expected[1]!==b)return {ok:false,work};prev.push([a,b])}let [a,b]=prev[c.out];return {ok:String(a)===target[0]&&b===target[1],work:work+c.n+1}}

function affineTruth(a,b,n){return Array.from({length:1<<n},(_,x)=>parity(a&x)^b)}

function isAffine(truth,n){let b=truth[0],a=0;for(let i=0;i<n;i++)a|=(truth[1<<i]^b)<<i;return truth.every((v,x)=>v===(parity(a&x)^b))}

function rng(seed){let x=seed>>>0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return x>>>0}}

function make(n,steps,seed,allowNonlinear=false){let r=rng(seed),gates=Array.from({length:n},(_,i)=>({op:"in",i}));gates.push({op:"const",v:0},{op:"const",v:1});for(let j=0;j<steps;j++){let a=r()%gates.length,b=r()%gates.length,mode=r()%5;if(mode===0){gates.push({op:"and",a,b:allowNonlinear?b:n+1})}else if(mode===1)gates.push({op:"and",a,b:a});else gates.push({op:"xor",a,b})}return {n,gates,out:gates.length-1}}

function anf(c,cap){let rows=[],peak=0,work=0;for(const g of c.gates){let set;if(g.op==="in")set=new Set([1n<<BigInt(g.i)]);else if(g.op==="const")set=new Set(g.v?[0n]:[]);else if(g.op==="xor"){set=new Set(rows[g.a]);for(const v of rows[g.b]){work++;if(set.has(v))set.delete(v);else set.add(v)}}else{set=new Set();for(const a of rows[g.a])for(const b of rows[g.b]){work++;let v=a|b;if(set.has(v))set.delete(v);else set.add(v);if(set.size>cap)return {status:"cap",peak:Math.max(peak,set.size),work}}}peak=Math.max(peak,set.size);if(set.size>cap)return {status:"cap",peak,work};rows.push(set)}return {status:"exact",peak,work,outTerms:rows[c.out].size}}

function productCancel(n){let gates=Array.from({length:n},(_,i)=>({op:"in",i}));let one=gates.push({op:"const",v:1})-1,terms=[];for(let i=0;i<n;i++)terms.push(gates.push({op:"xor",a:i,b:one})-1);let prod=terms[0];for(let i=1;i<n;i++)prod=gates.push({op:"and",a:prod,b:terms[i]})-1;let out=gates.push({op:"xor",a:prod,b:prod})-1;return {n,gates,out}}

function localZero(c){let g=c.gates[c.out];return g.op==="xor"&&g.a===g.b}

function run(){let checks=0,pairs=0,certCases=0,nonlinearCases=0,negative=0;const check=x=>{checks++;if(!x)throw Error("check "+checks)};
 for(let n=1;n<=5;n++)for(let a=0;a<(1<<n);a++)for(let c=0;c<(1<<n);c++)for(let b=0;b<2;b++)for(let d=0;d<2;d++){let x=affineStep({op:"and",a:0,b:1},[[BigInt(a),b],[BigInt(c),d]]),t=affineTruth(a,b,n).map((v,i)=>v&(parity(c&i)^d));check((x!==null)===isAffine(t,n));if(x)check(t.every((v,i)=>v===(parity(Number(x[0])&i)^x[1])));pairs++}
 for(let n=1;n<=8;n++)for(let seed=1;seed<=32;seed++){let c=make(n,30,seed*17+n),z=build(c);check(z.status==="certified");let target=[z.root[0],Number(z.root[1])];check(verify(c,z.certificate,target).ok);for(let x=0;x<(1<<n);x++)check(evaluate(c,x)===(parity(Number(target[0])&x)^target[1]));let corrupt=z.certificate.map(x=>x.slice());corrupt[c.out][1]^=1;check(!verify(c,corrupt,target).ok);check(!verify(c,z.certificate,[target[0],target[1]^1]).ok);let extra=z.certificate.slice(0,-1);check(!verify(c,extra,target).ok);negative+=3;certCases++;
 let c2=make(n,30,seed*17+n,true),z2=build(c2);if(z2.status==="certified"){check(verify(c2,z2.certificate,[z2.root[0],Number(z2.root[1])]).ok);for(let x=0;x<(1<<n);x++)check(evaluate(c2,x)===(parity(Number(z2.root[0])&x)^Number(z2.root[1])))}else nonlinearCases++}
 let profiles=[];for(let n of [4,8,16,24,32,40]){let c=make(n,4*n,123+n),z=build(c),v=verify(c,z.certificate,[z.root[0],Number(z.root[1])]);check(v.ok);profiles.push({n,gates:c.gates.length,exhaustiveGateEvaluations:String(BigInt(c.gates.length)*(1n<<BigInt(n))),constructionUnits:z.work,verificationUnits:v.work,certificatePayloadBits:c.gates.length*(n+1)})}
 let cancellation=[];for(let n of [2,4,8,12,16,20,24]){let c=productCancel(n),z=build(c),a=anf(c,65536);check(z.status==="unsupported");check(localZero(c));if(n<=16){check(a.status==="exact"&&a.peak===2**n&&a.outTerms===0)}else check(a.status==="cap");if(n<=12)for(let x=0;x<(1<<n);x++)check(evaluate(c,x)===0);cancellation.push({n,gates:c.gates.length,anf:a,localIdentity:"same-wire XOR"})}
 return {schema:"gal08.certificates.v1",checks,affineProductPairs:pairs,certCases,negative,unsupportedRandomCircuits:nonlinearCases,profiles,cancellation,scope:"authenticated circuit DAG as input; affine-at-every-gate proof; logical coefficient work, not hardware or cryptographic proof cost; finite ANF cap"}
}

console.log(JSON.stringify(run(),null,2));
