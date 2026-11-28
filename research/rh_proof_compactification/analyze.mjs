// RH-DAG-01. Source import graph and a synthetic no-recompute live-frontier proxy.
// No kernel proof checking or RSS estimate. Run with Node >= 20.
// Usage: node analyze.mjs /path/to/pinned/grh-lean | --self-test
import fs from "node:fs";
import path from "node:path";
import {execFileSync} from "node:child_process";
function headerImports(source) {
  let clean="", depth=0;
  for(let i=0;i<source.length;i++){
    const pair=source.slice(i,i+2), ch=source[i];
    if(depth){if(pair==="/-"){depth++;i++;}else if(pair==="-/"){depth--;i++;}else if(ch==="\n")clean+="\n";continue;}
    if(pair==="/-"){depth=1;clean+=" ";i++;continue;}
    if(pair==="--"){while(i+1<source.length&&source[i+1]!=="\n")i++;continue;}
    clean+=ch;
    // Only header parsing is needed; a full scanner is not a Lean parser.
    if(ch==="\n" && /^(?!(?:(?:public |meta )?import |module(?:\s|$)|prelude(?:\s|$)))/.test(clean.slice(0,-1).split("\n").at(-1).trim()) && clean.slice(0,-1).split("\n").at(-1).trim()) break;
  }
  const imports=[];
  for(const l of clean.split("\n")){
    const line=l.trim();
    if(!line||line==="module"||line==="prelude")continue;
    if(/^(?:public\s+|meta\s+)*import\b/.test(line)){
      const m=line.match(/^(?:public\s+|meta\s+)*import\s+(?:all\s+)?([\p{L}_][\p{L}\p{N}_'.]*)(?:\s*)$/u);
      if(!m)throw Error("unsupported import header: "+line);
      imports.push(m[1]);continue;
    }
    break;
  }
  return [...new Set(imports)].sort();
}
function analyze(nodes) {
  const keys=Object.keys(nodes).sort();
  const incoming=new Map(), children=new Map(keys.map(k=>[k,[]]));
  for(const k of keys){const n=nodes[k];if(!Number.isSafeInteger(n.bytes)||n.bytes<0)throw Error("invalid weight");
    const ds=[...new Set(n.imports)].filter(d=>Object.hasOwn(nodes,d)).sort();incoming.set(k,ds);
    for(const d of ds)children.get(d).push(k);}
  function replay(strategy){
    const done=new Set(),live=new Set(), order=[];let peak=0, liveBytes=0;
    const pending=new Map(keys.map(k=>[k,children.get(k).length]));
    while(done.size<keys.length){
      const ready=keys.filter(k=>!done.has(k)&&incoming.get(k).every(d=>done.has(d)));
      if(!ready.length)throw Error("cyclic module graph");
      function after(k){return liveBytes+nodes[k].bytes*(pending.get(k)>0?1:0)-incoming.get(k).filter(d=>pending.get(d)===1).reduce((s,d)=>s+nodes[d].bytes,0);}
      if(strategy==="frontier")ready.sort((a,b)=>Math.max(peak,liveBytes+nodes[a].bytes)-Math.max(peak,liveBytes+nodes[b].bytes)||after(a)-after(b)||(a<b?-1:a>b?1:0));
      const k=ready[0];peak=Math.max(peak,liveBytes+nodes[k].bytes);
      done.add(k);order.push(k);
      for(const d of incoming.get(k)){pending.set(d,pending.get(d)-1);if(pending.get(d)===0&&live.delete(d))liveBytes-=nodes[d].bytes;}
      if(pending.get(k)>0){live.add(k);liveBytes+=nodes[k].bytes;}
    }
    if(liveBytes!==0)throw Error("live weight leak");
    return {peak_source_bytes_proxy:peak,order};
  }
  const lexical=replay("lexical"),frontier=replay("frontier");
  const lengths=new Map();
  for(const k of lexical.order)lengths.set(k,1+Math.max(0,...incoming.get(k).map(d=>lengths.get(d))));
  const boundary={};
  for(const k of keys){const d=[...new Set(nodes[k].imports)].filter(x=>!Object.hasOwn(nodes,x)).sort();if(d.length)boundary[k]=d;}
  return {modules:keys.length,source_bytes:keys.reduce((s,k)=>s+nodes[k].bytes,0),induced_edges:keys.reduce((s,k)=>s+incoming.get(k).length,0),critical_path_modules:Math.max(0,...lengths.values()),unresolved_imports:boundary,lexical,frontier};
}
function selfTest(){
  const assert=(x,m)=>{if(!x)throw Error(m);};
  assert(JSON.stringify(headerImports("/- x /- nested -/ x -/\n-- hi\nimport Foo.Bar\npublic import Baz\n\ndef a := 1\n"))===JSON.stringify(["Baz","Foo.Bar"]),"nested comment");
  assert(headerImports("module\nprelude\nimport all A\n").join()==="A","module syntax");
  assert(headerImports("import X -- comment\nimport X\n").join()==="X","dedup");
  assert(headerImports("def a := 1\nimport X\n").length===0,"stop at body");
  const g={A:{bytes:10,imports:[]},B:{bytes:2,imports:["A","Mathlib"]},C:{bytes:5,imports:["A"]},D:{bytes:1,imports:["B","C"]}};
  const r=analyze(g);assert(r.induced_edges===4&&r.critical_path_modules===3,"DAG counts");
  assert(r.lexical.peak_source_bytes_proxy===17,"allocation before eviction");
  assert(r.frontier.peak_source_bytes_proxy===17,"frontier fixture");
  assert(r.unresolved_imports.B[0]==="Mathlib","external boundary");
  let fail=false;try{analyze({A:{bytes:1,imports:["A"]}});}catch(e){fail=true;}assert(fail,"cycle");
  fail=false;try{analyze({A:{bytes:-1,imports:[]}});}catch(e){fail=true;}assert(fail,"negative");
  return {checks:10,status:"pass"};
}
function scan(root) {
  const nodes={};
  function walk(dir){for(const ent of fs.readdirSync(dir,{withFileTypes:true})){
    const p=path.join(dir,ent.name);
    if(ent.isDirectory())walk(p);
    else if(ent.isFile()&&p.endsWith(".lean")){
      const key=path.relative(root,p).slice(0,-5).split(path.sep).join(".");
      const data=fs.readFileSync(p);
      nodes[key]={bytes:data.length,imports:headerImports(data.toString("utf8"))};
    }
  }}
  walk(root);return nodes;
}
if(process.argv[2]==="--self-test")console.log(JSON.stringify(selfTest(),null,2));
else {
  const root=process.argv[2];if(!root)throw Error("supply pinned upstream checkout");
  const pin=execFileSync("git",["-C",root,"rev-parse","HEAD"],{encoding:"utf8"}).trim();
  if(pin!=="72fead102f663d5c7a407964a13dc0cf9d977207")throw Error("unexpected upstream pin: "+pin);
  if(execFileSync("git",["-C",root,"status","--porcelain","--untracked-files=no"],{encoding:"utf8"}).trim())throw Error("dirty tracked checkout");
  const output={pin,scope:"source module imports only; live frontier in source bytes, not RSS",tests:selfTest(),editions:{}};
  for(const edition of ["grh","grh-compact"])output.editions[edition]=analyze(scan(path.join(root,edition,"src")));
  console.log(JSON.stringify(output,null,2));
}
