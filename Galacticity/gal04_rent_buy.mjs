function rentBuy(B,ds,num=1,den=1){let rent=0,bought=false,cost=0,buyAt=null;for(let t=0;t<ds.length;t++){if(bought)continue;const d=ds[t];if(den*(rent+d)>=num*B){bought=true;buyAt=t;cost+=B}else{rent+=d;cost+=d}}return {cost,buyAt}}

function oracle(B,ds){return Math.min(B,ds.reduce((s,x)=>s+x,0))}

function run(){let checks=0,strict=0,max={n:0,d:1};const check=x=>{checks++;if(!x)throw Error("check "+checks)};for(let B=1;B<=16;B++)for(let length=0;length<=6;length++)for(let i=0;i<4**length;i++){let ds=Array.from({length},(_,j)=>(i>>(2*j))&3),D=ds.reduce((s,x)=>s+x,0),o=oracle(B,ds),a=rentBuy(B,ds);check(a.cost<=2*o);if(o>0&&a.cost*max.d>max.n*o)max={n:a.cost,d:o};if(a.buyAt!==null){check(a.cost<2*B);check(D>=B);strict++}else check(a.cost===o);for(const [num,den] of [[1,2],[1,4],[2,1],[4,1]]){let p=rentBuy(B,ds,num,den);let L=Math.max(num,den),s=Math.min(num,den);check(s*p.cost<=(s+L)*o);if(num<den&&D>=B)check(den*p.cost<=(den+num)*o);if(num>den&&D<B)check(p.cost===o)}}
let unit=[];for(let B=1;B<=64;B++){let worst={n:0,d:1};for(let H=1;H<=2*B+2;H++){let a=rentBuy(B,Array(H).fill(1)).cost,o=Math.min(B,H);check(B*a<=(2*B-1)*o);if(a*worst.d>worst.n*o)worst={n:a,d:o}}check(B*worst.n===(2*B-1)*worst.d);unit.push({B,worstNumerator:2*B-1,worstDenominator:B})}
return {schema:"gal04.rent_buy.v1",checks,buyCases:strict,maxObservedRatio:max,unit,scope:"finite exact arithmetic; fixed build price, no holding cost, no cap conflict, no controller overhead"}}

console.log(JSON.stringify(run(),null,2));
