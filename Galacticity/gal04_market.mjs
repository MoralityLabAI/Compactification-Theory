import { pathToFileURL } from "node:url";

function rng(seed){let x=seed>>>0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return x>>>0}}

function itemsFor(seed){const rnd=rng(seed);return Array.from({length:5},(_,i)=>({size:2+rnd()%4,read:12+rnd()%25,hit:1+rnd()%4,extra:20+rnd()%70}))}

function eventsFor(seed,family,H=72){const rnd=rng(seed),events=[];for(let t=0;t<H;t++){let item,price,hold,invalid=0;
 if(family==="stable"){item=rnd()%10<8?0:1+rnd()%4;price=2;hold=0;if(t%29===0)invalid=1<<item}
 else if(family==="bursts"){item=rnd()%5===0?rnd()%5:Math.floor(t/9)%5;price=1+rnd()%3;hold=rnd()%2;if(t%9===0)invalid=1<<item}
 else if(family==="churn"){item=t%5;price=1;hold=2;if(rnd()%4===0)invalid=1<<(rnd()%5)}
 else if(family==="prices"){item=rnd()%3;price=t%16<8?1:8;hold=t%16<8?2:0;if(t%18===0)invalid=1<<item}
 else if(family==="shock"){item=t%19<10?0:rnd()%5;price=t%19<10?8:1;hold=t%19<10?0:8;if(t%19===10)invalid=31;else if(rnd()%6===0)invalid=1<<(rnd()%5)}
 else throw Error("family");
 events.push({item,price,hold,invalid})}return events}

function units(items,mask){let s=0;for(let i=0;i<items.length;i++)if(mask&(1<<i))s+=items[i].size;return s}

function directCost(item,ev){return item.read*ev.price}

function buildCost(item,ev){return directCost(item,ev)+item.extra}

function marketStep(items,prev,ev,next,cap){const j=ev.item,bit=1<<j,live=prev&~ev.invalid;if(next&~(live|bit)||units(items,next)>cap)throw Error("illegal transition");const added=next&~live;return {build:added?buildCost(items[j],ev):0,service:1+(next&bit?Math.min(items[j].hit,directCost(items[j],ev)):directCost(items[j],ev)),holding:ev.hold*units(items,next)}}

function total(x){return x.build+x.service+x.holding}

function offline(items,events,cap){let dp=new Map([[0,0]]),parents=[];for(const ev of events){let nxt=new Map(),par=new Map();for(const [prev,c] of dp){let avail=(prev&~ev.invalid)|(1<<ev.item);for(let m=avail;;m=(m-1)&avail){if(units(items,m)<=cap){let z=c+total(marketStep(items,prev,ev,m,cap));if(!nxt.has(m)||z<nxt.get(m)){nxt.set(m,z);par.set(m,prev)}}if(m===0)break}}parents.push(par);dp=nxt}
 let [mask,cost]=[...dp].sort((a,b)=>a[1]-b[1]||a[0]-b[0])[0],actions=[];for(let i=parents.length-1;i>=0;i--){actions.push(mask);mask=parents[i].get(mask)}actions.reverse();return {cost,actions}}

function brute(items,events,cap,t=0,prev=0){if(t===events.length)return 0;const ev=events[t],live=prev&~ev.invalid,bit=1<<ev.item;let best=Infinity;for(let m=0;m<(1<<items.length);m++){if((m&~live&~bit)!==0||units(items,m)>cap)continue;let acq=(m&bit)&&!(live&bit)?items[ev.item].read*ev.price+items[ev.item].extra:0;let serve=1+((m&bit)?Math.min(items[ev.item].hit,items[ev.item].read*ev.price):items[ev.item].read*ev.price);best=Math.min(best,acq+serve+ev.hold*units(items,m)+brute(items,events,cap,t+1,m))}return best}

function createPolicy(name,items,cap,num=1,den=1,ttl=4){let mask=0,credit=items.map(()=>0),last=items.map(()=>-100000),t=-1;return function decide(ev){t++;let work=0;const j=ev.item,bit=1<<j,old=mask;mask&=~ev.invalid;if(name==="direct")return {mask:0,work:0};
 for(let i=0;i<items.length;i++){work++;if(ev.invalid&(1<<i))credit[i]=0;if(name==="threshold"&&i!==j&&(mask&(1<<i))&&ev.hold>0&&t-last[i]>=ttl){mask&=~(1<<i);credit[i]=0}}
 let rent=directCost(items[j],ev),save=Math.max(0,rent-items[j].hit-ev.hold*items[j].size),B=buildCost(items[j],ev);work++;
 if(name==="threshold"&&(mask&bit)&&save===0){mask&=~bit;credit[j]=0}
 let admit=!(mask&bit)&&items[j].size<=cap&&(name==="lru"||save>0&&den*(credit[j]+save)>=num*B);
 if(admit){while(units(items,mask)+items[j].size>cap){let victim=-1;for(let i=0;i<items.length;i++){work++;if((mask&(1<<i))&&(victim<0||last[i]<last[victim]))victim=i}if(victim<0)throw Error("capacity logic");mask&=~(1<<victim);credit[victim]=0}mask|=bit;credit[j]=0}
 else if(!(mask&bit)&&name==="threshold")credit[j]+=save;
 last[j]=t;return {mask,work}}
}

function parity(x){let p=0;while(x){x&=x-1;p^=1}return p}
function referenceParity(x){let p=0;for(let j=0;j<32;j++)p=(p+((x>>>j)&1))%2;return p}

function replay(items,events,cap,seed,decide,offlineMasks=null){let rand=rng(seed),world=items.map(()=>rand()),cache=new Map(),prev=0,tot={build:0,service:0,holding:0,controller:0},peak=0,fills=0,hits=0,invalidations=0,answers=0;
 for(let t=0;t<events.length;t++){let ev=events[t];for(let i=0;i<items.length;i++)if(ev.invalid&(1<<i)){world[i]=rand();if(cache.delete(i))invalidations++}
 const j=ev.item,bit=1<<j,act=offlineMasks===null?decide(ev):{mask:offlineMasks[t],work:0};let m=act.mask,live=prev&~ev.invalid;
 if(m&~(live|bit)||units(items,m)>cap)throw Error("invalid action");
 for(const i of [...cache.keys()])if(!(m&(1<<i)))cache.delete(i);
 for(let i=0;i<items.length;i++)if(m&(1<<i)){if(!cache.has(i)){if(i!==j)throw Error("prefetch");cache.set(i,parity(world[i]));tot.build+=items[i].read*ev.price+items[i].extra;fills++}}
 if(m&bit&&items[j].hit<=items[j].read*ev.price){if(cache.get(j)!==referenceParity(world[j]))throw Error("stale answer");tot.service+=items[j].hit+1;hits++}else{let y=parity(world[j]);if(y!==referenceParity(world[j]))throw Error("wrong direct");tot.service+=items[j].read*ev.price+1}
 let bytes=0;for(const i of cache.keys())bytes+=items[i].size;if(bytes!==units(items,m))throw Error("cache mismatch");tot.holding+=bytes*ev.hold;tot.controller+=act.work;peak=Math.max(peak,bytes);answers++;prev=m}
 let cost=tot.build+tot.service+tot.holding;return {market:cost,total:cost+tot.controller,...tot,peak,fills,hits,invalidations,answers}}

function configList(){let c=[{name:"direct"},{name:"lru"}];for(const [num,den] of [[1,2],[1,1],[2,1]])for(const ttl of [1,4,12])c.push({name:"threshold",num,den,ttl});return c}

function key(c){return c.name==="threshold"?`threshold_${c.num}_${c.den}_ttl${c.ttl}`:c.name}

function run(){
 let checks=0;const check=v=>{checks++;if(!v)throw Error("check "+checks)};
 let bruteCases=0;const tiny=[{size:1,read:2,hit:1,extra:1},{size:2,read:3,hit:1,extra:2}];for(let H=0;H<=4;H++)for(let s=0;s<4**H;s++){let ev=Array.from({length:H},(_,i)=>{let v=(s>>(2*i))&3;return {item:v&1,price:1+(v>>1),hold:v>>1,invalid:v===3?1:0}});for(let cap=0;cap<=3;cap++){const o=offline(tiny,ev,cap);check(o.cost===brute(tiny,ev,cap));check(replay(tiny,ev,cap,23,null,o.actions).market===o.cost);bruteCases++}}
 const configs=configList(),trainFamilies=["stable","bursts","churn","prices"],testFamilies=[...trainFamilies,"shock"],cal=Object.fromEntries(configs.map(c=>[key(c),0]));const build=(seed,f,cap)=>({items:itemsFor(seed),events:eventsFor(seed^0x5c151f31,f),cap,seed:seed^0x97cab123});
 for(let fi=0;fi<trainFamilies.length;fi++)for(let i=0;i<6;i++)for(let cap of [5,8,12]){let e=build(10001+fi*100+i,trainFamilies[fi],cap);for(let c of configs){let z=replay(e.items,e.events,cap,e.seed,createPolicy(c.name,e.items,cap,c.num,c.den,c.ttl));cal[key(c)]+=z.total;check(z.peak<=cap)}}
 let selected=configs.slice().sort((a,b)=>cal[key(a)]-cal[key(b)])[0], rows=[];
 for(let fi=0;fi<testFamilies.length;fi++)for(let i=0;i<8;i++)for(let cap of [5,8,12]){let f=testFamilies[fi],seed=20001+fi*100+i,e=build(seed,f,cap);let o=offline(e.items,e.events,cap),v=replay(e.items,e.events,cap,e.seed,null,o.actions);check(v.market===o.cost);const p={};for(const c of [configs[0],configs[1],{name:"threshold",num:1,den:1,ttl:4},selected]){const k=key(c);let z=replay(e.items,e.events,cap,e.seed,createPolicy(c.name,e.items,cap,c.num,c.den,c.ttl));check(z.market>=o.cost&&z.peak<=cap&&z.answers===e.events.length);p[k]=z}rows.push({family:f,seed,cap,offline:o.cost,policies:p})}
 let summaries={};let keys=[...new Set(["direct","lru","threshold_1_1_ttl4",key(selected)])];for(const f of [...testFamilies,"all"]){let group=rows.filter(r=>f==="all"||r.family===f);summaries[f]={episodes:group.length,offline:group.reduce((s,r)=>s+r.offline,0),policies:{}};for(const k of keys){let cost=group.reduce((s,r)=>s+r.policies[k].total,0),market=group.reduce((s,r)=>s+r.policies[k].market,0),overhead=cost-market,loss=group.filter(r=>r.policies[k].total>r.policies.direct.total).length;summaries[f].policies[k]={cost,market,overhead,lossesVsDirect:loss,maxRatio:Math.max(...group.map(r=>r.policies[k].total/r.offline))}}}
 return {schema:"gal04.market.v1",checks,bruteCases,trainEpisodes:72,testEpisodes:rows.length,requests:rows.length*72,selected:key(selected),calibration:cal,summaries,rows,scope:"synthetic exact semantic cache; declared work units, payload cap, current public event only; offline oracle pays no planning; no native BPD or DataGoblin training"}
}



export { rng, itemsFor, eventsFor, units, directCost, marketStep, total, offline, createPolicy, replay };
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) console.log(JSON.stringify(run(),null,2));
