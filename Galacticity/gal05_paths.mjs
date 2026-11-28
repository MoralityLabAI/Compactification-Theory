function leaf(c,r){return {need:c,gain:r-c}}

function compose(a,b){return {need:Math.max(a.need,b.need-a.gain),gain:a.gain+b.gain}}

function summary(xs){return xs.reduce((a,[c,r])=>compose(a,leaf(c,r)),{need:0,gain:0})}

function simulate(xs,k){for(const [c,r] of xs){if(k<c)return null;k+=r-c}return k}

function ordered(xs){return xs.slice().sort((a,b)=>{const ap=a[1]>=a[0],bp=b[1]>=b[0];if(ap!==bp)return ap?-1:1;return ap?a[0]-b[0]:b[1]-a[1]})}

function permMin(xs){if(!xs.length)return 0;let best=Infinity;function go(rest,seq){if(!rest.length){best=Math.min(best,summary(seq).need);return}for(let i=0;i<rest.length;i++)go(rest.filter((_,j)=>i!==j),[...seq,rest[i]])}go(xs,[]);return best}

function rng(seed){let x=seed>>>0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return x>>>0}}

function run(){let checks=0;const check=v=>{checks++;if(!v)throw Error("check "+checks)};let pairs=0,triples=0,permutationCases=0;
const xs=[];for(let c=0;c<=8;c++)for(let r=0;r<=8;r++)xs.push([c,r]);
for(const a of xs)for(const b of xs){const s=summary([a,b]),o=summary(ordered([a,b]));check(o.need<=s.need);check(simulate([a,b],s.need)===s.need+s.gain);if(s.need>0)check(simulate([a,b],s.need-1)===null);pairs++}
const ys=xs.filter(x=>x[0]<=3&&x[1]<=3);for(const a of ys)for(const b of ys)for(const c of ys){const x=compose(compose(leaf(...a),leaf(...b)),leaf(...c)),y=compose(leaf(...a),compose(leaf(...b),leaf(...c)));check(x.need===y.need&&x.gain===y.gain);check(summary(ordered([a,b,c])).need===permMin([a,b,c]));triples++}
for(let seed=1;seed<=96;seed++){const rand=rng(seed*0x12345);const seq=Array.from({length:6},()=>[rand()%30,rand()%30]);const s=summary(ordered(seq));check(s.need===permMin(seq));check(simulate(ordered(seq),s.need)===s.need+s.gain);permutationCases++}
const ladders=[];for(let d=1;d<=40;d++){const seq=Array.from({length:d},(_,i)=>[2**i,2**(i+1)]);seq.push([2**d,2**d]);const s=summary(seq);check(s.need===1);check(simulate(seq,1)===2**d);check(simulate(seq,0)===null);ladders.push({depth:d,targetCost:2**d,minimumSeed:s.need,finalCapital:2**d})}
return {schema:"gal05.capital_paths.v1",checks,pairs,triples,permutationCases,ladders,scope:"exact integer one-shot investments with immediate realized returns before next acquisition; no planning overhead, dependencies or shared returns"}}

console.log(JSON.stringify(run(),null,2));
