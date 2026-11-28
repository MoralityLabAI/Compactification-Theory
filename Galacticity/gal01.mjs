function ceilDiv(a,b){if(a<0n||b<=0n)throw Error("domain");return (a+b-1n)/b}

function threshold(K,b,q){if(K<1n||b<1n||q<1n)throw Error("domain");if(q===1n)return 1n;if(q>=b+1n)return null;return ceilDiv(q*K*b,b+1n-q)}

function direct(b,m){return m*(b+1n)}

function optimal(K,b,m){return (m<K?m:K)*b+m}

function parity(block){return block.reduce((x,y)=>x^y,0)}

function replay(blocks,queries,cached){let count=0,cache=new Map(),answers=[];for(const i of queries){let y;if(cached&&cache.has(i)){y=cache.get(i)}else{count+=blocks[i].length;y=parity(blocks[i]);if(cached)cache.set(i,y)}answers.push(y);count++}return {count,answers}}

function strassen(k,w){let n=BigInt(k);return (w+6n)*7n**n-6n*4n**n}

function classical(k,w){let n=BigInt(k);return (w+1n)*8n**n-4n**n}

function amortize(C,d){if(C<0n)throw Error("domain");return d>0n?ceilDiv(C,d):C===0n&&d===0n?0n:null}

function inferCost(rows){if(rows.length!==2)throw Error("need two distinct horizons");let [[m1,y1],[m2,y2]]=rows;if(m2<=m1||(y2-y1)%(m2-m1)!==0n)return null;let d=(y2-y1)/(m2-m1),C=y1-m1*d;return C>=0n&&d>=0n?{C,d}:null}

function campaign(){
 let checks=0;const check=(v)=>{checks++;if(!v)throw Error("check "+checks)};
 let bounds=[];for(let k=0;k<=10;k++)for(let b=1n;b<=16n;b++)for(let q=1n;q<=b+2n;q++){let K=2n**BigInt(k),a=threshold(K,b,q);if(a!==null){check(direct(b,a)>=q*optimal(K,b,a));if(a>1n)check(direct(b,a-1n)<q*optimal(K,b,a-1n));check(direct(b,a+K+1n)>=q*optimal(K,b,a+K+1n))}else{check(direct(b,K*100000n)<q*optimal(K,b,K*100000n))}if(b===8n&&q===2n)bounds.push({addressBits:k,K:String(K),threshold:String(a)})}
 let exhaustive=0;for(let x=0;x<256;x++){let blocks=Array.from({length:4},(_,j)=>[(x>>(2*j))&1,(x>>(2*j+1))&1]);for(let s=0;s<64;s++){let qs=[s&3,(s>>2)&3,(s>>4)&3];let a=replay(blocks,qs,false),c=replay(blocks,qs,true);check(JSON.stringify(a.answers)===JSON.stringify(c.answers));check(c.count===2*new Set(qs).size+qs.length);exhaustive++}}
 let tensor=[];for(const w of [1n,2n,4n,8n,16n,32n]){let s=w,first=null;for(let k=0;k<=32;k++){if(k>0)s=7n*s+18n*4n**BigInt(k-1);check(s===strassen(k,w));if(first===null&&k>0&&2n*s<=classical(k,w))first=k;}tensor.push({multiplyWeight:String(w),first2xPower:first,dimension:first===null?null:String(2n**BigInt(first))})}
 // Public calibration packets: entire construction-cost ranges held out.
 let fitPass=0,meanBaselinePass=0,packets=0,train=[];
 for(let C=1n;C<=16n;C++)for(let d=1n;d<=8n;d++)train.push(amortize(C,d));
 train.sort((a,b)=>a<b?-1:a>b?1:0);const median=train[Math.floor(train.length/2)];
 for(let C=17n;C<=32n;C++)for(let d=1n;d<=8n;d++){let v=3n;let o=inferCost([[1n,C+v],[4n,C+4n*v]]);let target=amortize(C,d);if(o&&amortize(o.C,d)===target)fitPass++;if(median===target)meanBaselinePass++;packets++}
 check(inferCost([[1n,4n],[4n,8n]])===null);
 return {schema:"gal01.receipt.v1",checks,exhaustiveParityCases:exhaustive,bounds,tensor,prediction:{trainCases:train.length,heldoutCases:packets,fitPass,medianBaselinePass:meanBaselinePass,median:String(median)},scope:"exact logical operation counts; no timing, live gym or neural training"}
}

console.log(JSON.stringify(campaign(),null,2));
