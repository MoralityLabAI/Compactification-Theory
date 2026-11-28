function parity(v){let p=0;while(v){v&=v-1;p^=1}return p}

function dot(a,x){return parity(a&x)}

function rng(seed){let x=seed>>>0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return x>>>0}}

function makeRequests(r,n,mode,seed){let random=rng(seed),mask=(1<<r)-1;let a=[];for(let i=0;i<n;i++){if(mode==="repeat")a.push(1<<(i%Math.min(r,4)));else if(mode==="subspace")a.push((random()&((1<<Math.min(r,4))-1))||1);else if(mode==="dense")a.push((random()&mask)||1);else throw Error("unknown mode")}return a}

function evaluate(r,b,requests,world,mode,probePrice,memCap=Infinity){
 const basis=[],exact=new Map(),answers=[];let probes=0,arith=0,key=0,peak=0,bypasses=0;
 function direct(a){let y=0;for(const x of world){y^=dot(a,x);probes++;arith++}return y}
 for(const a of requests){let result;
  if(mode==="direct")result=direct(a);
  else if(mode==="repeat"){
   key+=r+1;
   if(exact.has(a))result=exact.get(a);
   else{result=direct(a);if((exact.size+1)*(r+1)<=memCap)exact.set(a,result);else bypasses++}
  }else if(mode==="basis"){
   let v=a,z=0;
   for(const [p,w,y] of basis){arith++;if((v>>>p)&1){v^=w;arith+=r;z^=y;arith++}}
   if(v===0){result=z}
   else{result=direct(a);if((basis.length+1)*(r+1+Math.ceil(Math.log2(r)))<=memCap){let p=31-Math.clz32(v);arith+=r;basis.push([p,v,z^result]);arith++}else bypasses++}
  }else throw Error("mode");
  peak=Math.max(peak,mode==="basis"?basis.length*(r+1+Math.ceil(Math.log2(r))):mode==="repeat"?exact.size*(r+1):0);answers.push(result)
 }
 return {answers,probes,arith,key,output:requests.length,peakPayloadBits:peak,bypasses,cost:probePrice*probes+arith+key+requests.length}
}

function run(){
 let checks=0;const check=v=>{checks++;if(!v)throw Error("check "+checks)};let fixtures=0;
 for(let x=0;x<8;x++)for(let y=0;y<8;y++)for(let i=0;i<343;i++){let req=[1+i%7,1+Math.floor(i/7)%7,1+Math.floor(i/49)%7],base=evaluate(3,2,req,[x,y],"direct",1);for(const mode of ["repeat","basis"])for(const cap of [0,4,8,100]){let out=evaluate(3,2,req,[x,y],mode,1,cap);check(JSON.stringify(out.answers)===JSON.stringify(base.answers));check(out.peakPayloadBits<=cap);fixtures++}}
 const rows=[];for(const r of [8,16,24])for(const b of [2,8,32])for(const mode of ["repeat","subspace","dense"]){let random=rng(r*991+b);let req=makeRequests(r,256,mode,r*123+b);let world=Array.from({length:b},()=>random()&((1<<r)-1));for(const price of [1,16,128]){let a=evaluate(r,b,req,world,"direct",price),c=evaluate(r,b,req,world,"repeat",price),d=evaluate(r,b,req,world,"basis",price);for(const z of [c,d])check(JSON.stringify(z.answers)===JSON.stringify(a.answers));let strip=({answers,...o})=>o;rows.push({r,b,workload:mode,price,direct:strip(a),repeat:strip(c),basis:strip(d),winner:[["direct",a.cost],["repeat",c.cost],["basis",d.cost]].sort((a,b)=>a[1]-b[1])[0][0]})}}
 const winners={};for(const z of rows)winners[z.winner]=(winners[z.winner]||0)+1;
 return {schema:"gal03.priced_basis.v1",checks,fixtures,profiles:rows.length,winners,rows,scope:"synthetic slice-local oracle; explicit probe price, XOR and pivot-test units, repeat-key r+1 ideal dictionary units, output units; resident payload cap excludes runtime overhead"}
}

console.log(JSON.stringify(run(),null,2));
