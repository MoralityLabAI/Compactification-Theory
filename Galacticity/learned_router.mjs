function ceilRatio(c,d){return Math.ceil(c/d)}

function pub(C,d,m){let u=10+d,v=10;return {observations:[{m:1,baseline:u,candidate:C+v},{m:4,baseline:4*u,candidate:C+4*v}],horizon:m,contract:"affine_exact",id:`${C}:${d}:${m}`}}

function features(p){const [a,b]=p.observations;let v=(b.candidate-a.candidate)/(b.m-a.m),C=a.candidate-a.m*v,u=a.baseline/a.m;return [1,Math.log(p.horizon),Math.log(C),Math.log(u-v)]}

function label(C,d,m){return m*d>=C?1:0}

function corpus(lo,hi){let rows=[];for(let C=lo;C<=hi;C++)for(let d=1;d<=8;d++)for(const m of [1,2,4,8,16,32])rows.push({packet:pub(C,d,m),target:label(C,d,m)});return rows}

function score(w,p){return features(p).reduce((s,x,i)=>s+x*w[i],0)}

function train(rows){let w=[0,0,0,0];for(let epoch=0;epoch<1000;epoch++){let g=[0,0,0,0];for(const {packet,target} of rows){let x=features(packet),z=x.reduce((s,a,i)=>s+a*w[i],0),p=1/(1+Math.exp(-z));for(let i=0;i<4;i++)g[i]+=(p-target)*x[i]}for(let i=0;i<4;i++)w[i]-=0.1*g[i]/rows.length}return w}

function evaluate(w,rows){let correct=0,unsafe=0,falseAbstain=0;for(const {packet,target} of rows){let prediction=+(score(w,packet)>=0);correct+=+(prediction===target);unsafe+=+(prediction===1&&target===0);falseAbstain+=+(prediction===0&&target===1)}return {n:rows.length,correct,unsafe,falseAbstain}}

function gate(p){if(p.contract!=="affine_exact")return "abstain";const [a,b]=p.observations;if(b.m<=a.m)return "abstain";let u=a.baseline/a.m,v=(b.candidate-a.candidate)/(b.m-a.m),C=a.candidate-a.m*v;if(![u,v,C,p.horizon].every(Number.isSafeInteger)||C<0||v<0||u<0||p.horizon<1||b.baseline!==b.m*u)return "abstain";return BigInt(p.horizon)*(BigInt(u)-BigInt(v))>=BigInt(C)?"compactify":"direct"}

function pilot(){const training=corpus(1,16),test=corpus(17,32),w=train(training);let gatedCorrect=0;for(const {packet,target} of test)gatedCorrect+=+(gate(packet)===(target?"compactify":"direct"));let example={...pub(16,2,16),contract:"unknown_tail"};let adversary={commonEarlyObservations:example.observations,horizon:16,baseline:192,affineCost:176,adversarialCost:240,gate:gate(example)};return {schema:"gal01.learned_router.v1",training: evaluate(w,training),heldout:evaluate(w,test),gatedCorrect,weights:w,adversary,training_admission:false,scope:"synthetic offline logistic baseline; no DataGoblin or LLM weights updated"}}

console.log(JSON.stringify(pilot(),null,2));
