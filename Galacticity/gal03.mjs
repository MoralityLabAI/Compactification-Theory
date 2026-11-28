function ceilDiv(a,b){return (a+b-1n)/b}

function threshold(C,u,v,q=2n){C=BigInt(C);u=BigInt(u);v=BigInt(v);q=BigInt(q);let d=u-q*v;if(d<0n||d===0n&&C>0n)return null;if(d===0n)return 1n;let k=ceilDiv(q*C,d);return k>0n?k:1n}

function rng(seed){let x=seed>>>0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return x>>>0}}

function mat(n,m){return Array.from({length:n},()=>new Uint8Array(m))}

function multiply(U,V){let n=U.length,r=V.length,m=V[0].length,A=mat(n,m);for(let i=0;i<n;i++)for(let j=0;j<m;j++){let y=0;for(let k=0;k<r;k++)y^=U[i][k]&V[k][j];A[i][j]=y}return A}

function make(n,m,r,seed){let rand=rng(seed),U=mat(n,r),V=mat(r,m);for(let i=0;i<n;i++)for(let k=0;k<r;k++)U[i][k]=i<r?+(i===k):rand()&1;for(let k=0;k<r;k++)for(let j=0;j<m;j++)V[k][j]=j<r?+(k===j):rand()&1;return multiply(U,V)}

function discover(A){const n=A.length,m=A[0].length,basis=[],pivots=[],coeffs=[];let reads=0,tests=0,xors=0;for(let j=0;j<m;j++){let v=new Uint8Array(n),c=[];for(let i=0;i<n;i++){v[i]=A[i][j];reads++}for(let k=0;k<basis.length;k++){let hit=v[pivots[k]];tests++;c.push(hit);if(hit)for(let i=0;i<n;i++){v[i]^=basis[k][i];xors++}}let p=-1;for(let i=0;i<n;i++){tests++;if(v[i]){p=i;break}}if(p>=0){basis.push(v);pivots.push(p);c.push(1)}coeffs.push(c)}const r=basis.length,U=mat(n,r),V=mat(r,m);for(let k=0;k<r;k++){for(let i=0;i<n;i++)U[i][k]=basis[k][i];for(let j=0;j<m;j++)V[k][j]=coeffs[j][k]??0}return {U,V,r,build:{reads,tests,xors,total:reads+tests+xors},factorBits:r*(n+m),basisWorkingBits:r*n+r*m+n}}

function verify(A,U,V){const n=A.length,m=A[0].length,r=V.length;let equal=true,checks=0;for(let i=0;i<n;i++)for(let j=0;j<m;j++){let z=0;for(let k=0;k<r;k++)z^=U[i][k]&V[k][j];if(A[i][j]!==z)equal=false;checks++}return {equal,checks,cost:2*n*m*r+2*n*m}}

function mv(A,x){return A.map(row=>{let z=0;for(let j=0;j<x.length;j++)z^=row[j]&x[j];return z})}

function onlineCost(n,m,r){return {u:2*n*m+n,v:2*r*(n+m)+n}}

function campaign(){let checks=0;const check=x=>{checks++;if(!x)throw Error("check "+checks)};let exhaustive=0;for(let word=0;word<512;word++){const A=mat(3,3);for(let i=0;i<3;i++)for(let j=0;j<3;j++)A[i][j]=(word>>(3*i+j))&1;const d=discover(A);check(verify(A,d.U,d.V).equal);for(let a=0;a<8;a++){const x=[a&1,(a>>1)&1,(a>>2)&1];check(JSON.stringify(mv(A,x))===JSON.stringify(mv(d.U,mv(d.V,x))))}exhaustive++}let spikeCases=0;for(let n=1;n<=8;n++){const A=mat(n,n);for(let i=0;i<n;i++)for(let j=0;j<n;j++){A[i][j]=1;const d=discover(A);check(d.r===1&&verify(A,d.U,d.V).equal);check(d.build.reads===n*n);A[i][j]=0;spikeCases++}}
const rows=[];for(const n of [32,64,128,256])for(const r of [1,4,8,16,32].filter(x=>x<=n)){const m=n,A=make(n,m,r,0x8128+n*17+r),d=discover(A),vfy=verify(A,d.U,d.V);check(d.r===r&&vfy.equal);const rand=rng(n*31+r);for(let s=0;s<8;s++){let x=Array.from({length:m},()=>rand()&1);check(JSON.stringify(mv(A,x))===JSON.stringify(mv(d.U,mv(d.V,x))))}const {u,v}=onlineCost(n,m,r),C=d.build.total+vfy.cost;let t=threshold(C,u,v),tBuild=threshold(d.build.total,u,v),tTrusted=threshold(0,u,v);for(const [cc,tt] of [[C,t],[d.build.total,tBuild],[0,tTrusted]])if(tt!==null){check(2n*(BigInt(cc)+tt*BigInt(v))<=tt*BigInt(u));if(tt>1n)check(2n*(BigInt(cc)+(tt-1n)*BigInt(v))>(tt-1n)*BigInt(u))}else check(u-2*v<0||u-2*v===0&&cc>0);rows.push({n,r,denseBits:n*m,factorBits:d.factorBits,build:d.build,verify:vfy.cost,u,v,first2x:{trusted:tTrusted===null?null:String(tTrusted),build:tBuild===null?null:String(tBuild),buildAndVerify:t===null?null:String(t)}})}let conditioning=[];for(let k=0;k<=40;k+=4){let p=k+11;check(2n**BigInt(p)>=2n**BigInt(k+11));check(2n**BigInt(p-1)<2n**BigInt(k+11));conditioning.push({k,precisionBits:p})}return {schema:"gal03.receipt.v1",checks,exhaustiveMatrices:exhaustive,spikeCases,rows,conditioning,scope:"GF(2) operation-count model; source entry reads, XOR, AND, tests, output units; resident accesses/writes and allocation free; storage payload separately reported; no native timing"}}

console.log(JSON.stringify(campaign(),null,2));
