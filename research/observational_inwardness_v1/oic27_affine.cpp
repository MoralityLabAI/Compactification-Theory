// OIC-27 independent n=6 robust search under the affine phase-label group.
// A phase relabeling (d,z,m) -> (u*d, u*z+c, u*m+c), u in {1,5},
// preserves the observation game and its unique-shift terminal objective.
// This implementation stores two explicit 36-bit error-layer masks and
// independently computes the pairwise remaining-disagreement obstruction.
#include <bits/stdc++.h>
using namespace std;
constexpr int N=6,W=36,MAXQ=32;
constexpr uint64_t ALL=(1ULL<<W)-1;
struct Node {uint64_t p0,p1;int t;
 bool operator==(Node const&v)const{return p0==v.p0&&p1==v.p1&&t==v.t;}
};
struct NodeHash {size_t operator()(Node const&v) const {
 uint64_t h=v.p0+0x9e3779b97f4a7c15ULL;
 h^=v.p1+0xbf58476d1ce4e5b9ULL+(h<<6)+(h>>2);
 h^=(uint64_t)v.t*0x94d049bb133111ebULL;
 h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;h^=h>>27;h*=0x94d049bb133111ebULL;h^=h>>31;
 return size_t(h);
}};
struct Gauge {Node state;int u,c;};
uint64_t hits[MAXQ][N]{},shiftMasks[N]{};
uint8_t remainingDiff[MAXQ+1][W][W]{};
uint8_t rowLut[2][N][64]{};
vector<int> units={1,5};
int Q=14;size_t LIMIT=2500000,expanded=0,cut=0;
unordered_map<Node,int8_t,NodeHash> cache;
chrono::steady_clock::time_point start;

inline uint64_t permute(uint64_t bits,int uIndex,int c){
 uint64_t result=0;
 for(int d=0;d<N;d++){
  int nextd=(units[uIndex]*d)%N;
  result|=(uint64_t)rowLut[uIndex][c][(bits>>(N*d))&63]<<(nextd*N);
 }
 return result;
}
Gauge canonical(Node state){
 Node best=state;int best_u=1,best_c=0;
 for(int ui=0;ui<2;ui++)for(int c=0;c<N;c++){
  uint64_t a=permute(state.p0,ui,c),b=permute(state.p1,ui,c);
  if(a<best.p0 || (a==best.p0 && b<best.p1)){
   best={a,b,state.t};best_u=units[ui];best_c=c;
  }
 }
 return {best,best_u,best_c};
}
inline int countshifts(uint64_t bits){int k=0;for(int d=0;d<N;d++)k+=((bits&shiftMasks[d])!=0);return k;}
inline bool resolved(Node x){return countshifts(x.p0|x.p1)<=1;}
inline int resolvedshift(Node x){uint64_t p=x.p0|x.p1;for(int d=0;d<N;d++)if(p&shiftMasks[d])return d;return -1;}
inline Node step(Node s,int marker,int received){
 uint64_t mask=hits[s.t][marker];
 uint64_t compatible=received?mask:ALL^mask;
 return {s.p0&compatible,(s.p1&compatible)|(s.p0&(ALL^compatible)),s.t+1};
}

bool impossible_pair(Node s){
 uint64_t unionbits=s.p0|s.p1;
 while(unionbits){
  int a=__builtin_ctzll(unionbits);unionbits&=unionbits-1;
  uint64_t later=unionbits;
  while(later){
   int b=__builtin_ctzll(later);later&=later-1;
   if(a/N==b/N)continue;
   int left=(s.p0>>a)&1,right=(s.p0>>b)&1;
   if(remainingDiff[s.t][a][b]<=left+right)return true;
  }
 }
 return false;
}

bool search(Node in){
 Gauge gauge=canonical(in);
 Node s=gauge.state;
 auto known=cache.find(s);if(known!=cache.end())return known->second>=0;
 if(++expanded>LIMIT)throw runtime_error("resource cap");
 if(resolved(s)){cache.emplace(s,N);return true;}
 if(s.t>=Q){cache.emplace(s,-1);return false;}
 if(impossible_pair(s)){++cut;cache.emplace(s,-1);return false;}
 struct Option{int m,metric;Node child[2];};
 vector<Option> options;
 for(int m=0;m<N;m++){
  Node zero=step(s,m,0),one=step(s,m,1);
  int size0=__builtin_popcountll(zero.p0|zero.p1),size1=__builtin_popcountll(one.p0|one.p1);
  int d0=countshifts(zero.p0|zero.p1),d1=countshifts(one.p0|one.p1);
  // Independent action ordering prioritizes distinct-shift reduction.
  options.push_back({m,max(d0,d1)*256 + (d0+d1)*32 + max(size0,size1),{zero,one}});
 }
 stable_sort(options.begin(),options.end(),[](auto const&a,auto const&b){return a.metric<b.metric;});
 for(auto const&op: options){
  int a=__builtin_popcountll(op.child[0].p0|op.child[0].p1), b=__builtin_popcountll(op.child[1].p0|op.child[1].p1);
  int first=(a>=b?0:1);
  if(search(op.child[first])&&search(op.child[1-first])){
   cache.emplace(s,op.m);return true;
  }
 }
 cache.emplace(s,-1);return false;
}

void emit(ostream &out,Node state){
 if(resolved(state)){out<<"{\"shift\":"<<resolvedshift(state)<<"}";return;}
 auto g=canonical(state);
 auto i=cache.find(g.state);if(i==cache.end()||i->second<0||i->second>=N)throw runtime_error("missing choice");
 // u is a unit modulo 6, and u^{-1}=u; canonical marker = u*m+c.
 int m=(g.u*(i->second-g.c+N))%N;
 out<<"{\"query\":"<<m<<",\"children\":{";
 bool first=true;
 for(int y=0;y<2;y++){
  auto ch=step(state,m,y);
  if(!(ch.p0|ch.p1))continue;
  if(!first)out<<",";first=false;
  out<<"\""<<y<<"\":";emit(out,ch);
 }
 out<<"}}";
}

void setup(){
 for(int d=0;d<N;d++)for(int z=0;z<N;z++)shiftMasks[d]|=1ULL<<(d*N+z);
 for(int t=0;t<Q;t++)for(int m=0;m<N;m++)for(int d=0;d<N;d++)for(int z=0;z<N;z++)
   if((z+t*d)%N==m)hits[t][m]|=1ULL<<(d*N+z);
 for(int t=0;t<=Q;t++)for(int a=0;a<W;a++)for(int b=0;b<W;b++){
   for(int i=t;i<Q;i++)remainingDiff[t][a][b]+=((a%N+i*(a/N))%N)!=((b%N+i*(b/N))%N);
 }
 for(int ui=0;ui<2;ui++)for(int c=0;c<N;c++)for(int mask=0;mask<64;mask++){
  uint8_t dest=0;for(int z=0;z<N;z++)if((mask>>z)&1)dest|=1<<((units[ui]*z+c)%N);
  rowLut[ui][c][mask]=dest;
 }
}
int main(int argc,char**argv){
 if(argc>1)Q=atoi(argv[1]);
 if(argc>2)LIMIT=strtoull(argv[2],nullptr,10);
 if(Q>MAXQ||Q<0)throw runtime_error("bad query horizon");
 setup();cache.reserve(min(LIMIT,(size_t)2000000));start=chrono::steady_clock::now();
 try{
  bool ok=search({ALL,0,0});double sec=chrono::duration<double>(chrono::steady_clock::now()-start).count();
  cout<<"STATUS "<<(ok?"SAT":"UNSAT")<<" Q="<<Q<<" states="<<expanded<<" cache="<<cache.size()<<" pruned="<<cut<<" seconds="<<sec<<"\n";
  if(ok){string file=argc>3?argv[3]:"oic27_affine_winner.json";ofstream out(file);emit(out,{ALL,0,0});out<<"\n";cout<<"TREE "<<file<<"\n";}
 }catch(exception const&e){double sec=chrono::duration<double>(chrono::steady_clock::now()-start).count();
  cerr<<"STATUS CAP Q="<<Q<<" states="<<expanded<<" cache="<<cache.size()<<" error="<<e.what()<<" seconds="<<sec<<"\n";return 2;}
}
