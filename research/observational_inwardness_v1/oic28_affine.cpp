// OIC-28 exact seven-phase one-bit-flip adaptive sensing.
// Finite worst-case decision game, 49-world candidate masks per error layer.
// Supports full affine phase gauge, plus sound cross-shift future-disagreement pruning.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
constexpr int N=7,W=N*N,MAXQ=36,U=N-1;
constexpr uint64_t ALL=(1ULL<<W)-1ULL, ROW=(1ULL<<N)-1;
struct State {uint64_t zero,one;uint8_t t;
 bool operator==(State const&x) const{return zero==x.zero && one==x.one && t==x.t;}
};
struct Hash{size_t operator()(State const&v) const noexcept {
  uint64_t a=v.zero * 0x9e3779b97f4a7c15ULL;
  uint64_t b=v.one * 0xbf58476d1ce4e5b9ULL;
  uint64_t c=(uint64_t)v.t*0x94d049bb133111ebULL;
  uint64_t h=a^(b+(a<<6)+(a>>2))^c;
  h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;h^=h>>27;h*=0x94d049bb133111ebULL;h^=h>>31;
  return size_t(h);
}};
struct Gauge {State state;int ui,c;};
uint64_t HIT[MAXQ][N]{},DM[N]{};
uint8_t FUT[MAXQ+1][W][W]{};
uint8_t rowLut[U][N][1<<N]{};
int units[U]={1,2,3,4,5,6}, invunit[U]={1,4,5,2,3,6};
int Q=18,heuristic=0;size_t LIMIT=1500000, expanded=0, pair_pruned=0;
bool USE_AFFINE=true, USE_PAIR=true;
unordered_map<State,int8_t,Hash> cache;
chrono::steady_clock::time_point started;
inline uint64_t transform(uint64_t bits,int ui,int c){
 uint64_t result=0;
 for(int d=0;d<N;d++){
   int newd=(units[ui]*d)%N;
   result|=(uint64_t)rowLut[ui][c][(bits>>(N*d))&ROW]<<(N*newd);
 }
 return result;
}
inline Gauge canonical(State state){
 State best=state;int best_ui=0,best_c=0;
 for(int ui=0;ui<(USE_AFFINE?U:1);ui++)for(int c=0;c<N;c++){
  uint64_t a=transform(state.zero,ui,c);
  if(a>best.zero)continue;
  uint64_t b=transform(state.one,ui,c);
  if(a<best.zero || (a==best.zero && b<best.one)){
   best={a,b,state.t};best_ui=ui;best_c=c;
  }
 }
 return {best,best_ui,best_c};
}
inline int countshifts(uint64_t bits){int count=0;for(int d=0;d<N;d++) count+=(bits&DM[d])!=0;return count;}
inline bool resolved(State s){return countshifts(s.zero|s.one)<=1;}
inline int resolvedshift(State s){uint64_t p=s.zero|s.one;for(int d=0;d<N;d++)if(p&DM[d])return d;return -1;}
inline State successor(State s,int m,int y){
 uint64_t cmask=y?HIT[s.t][m]:(ALL^HIT[s.t][m]);
 return {s.zero&cmask,(s.one&cmask)|(s.zero&(ALL^cmask)),(uint8_t)(s.t+1)};
}
inline bool impossible_pairs(State s){
 uint64_t bits=s.zero|s.one;
 while(bits){
  int a=__builtin_ctzll(bits);bits&=bits-1;
  uint64_t later=bits & ~DM[a/N];
  while(later){
   int b=__builtin_ctzll(later);later&=later-1;
   int remaining=((s.zero>>a)&1)+((s.zero>>b)&1);
   if(FUT[s.t][a][b]<=remaining) return true;
  }
 }
 return false;
}
struct Candidate {int marker,metric1,metric2;State child[2];};
inline vector<Candidate> options(State s){
 vector<Candidate> acts;acts.reserve(N);
 for(int m=0;m<N;m++){
  State zero=successor(s,m,0),one=successor(s,m,1);
  int b0=__builtin_popcountll(zero.zero|zero.one),b1=__builtin_popcountll(one.zero|one.one);
  int d0=countshifts(zero.zero|zero.one),d1=countshifts(one.zero|one.one);
  int metric=heuristic==0 ? max(d0,d1)*256+(d0+d1)*32+max(b0,b1)
   :heuristic==1 ? max(b0,b1)*100+abs(b0-b1)
   :heuristic==2 ? (d0*d0+d1*d1)*64+b0*b0+b1*b1
   :max(d0,d1)*256+max(b0,b1)*4+abs(b0-b1);
  acts.push_back({m,metric,m,{zero,one}});
 }
 stable_sort(acts.begin(),acts.end(),[](auto const&a,auto const&b){return a.metric1<b.metric1 || (a.metric1==b.metric1 && a.metric2<b.metric2);});
 return acts;
}
bool solve(State input){
 Gauge g=canonical(input);State s=g.state;
 auto known=cache.find(s);if(known!=cache.end())return known->second>=0;
 if(++expanded>LIMIT)throw runtime_error("state cap exceeded");
 if(resolved(s)){cache.emplace(s,N);return true;}
 if(s.t>=Q){cache.emplace(s,-1);return false;}
 if(USE_PAIR && impossible_pairs(s)){++pair_pruned;cache.emplace(s,-1);return false;}
 for(const auto &opt:options(s)){
  int b0=__builtin_popcountll(opt.child[0].zero|opt.child[0].one);
  int b1=__builtin_popcountll(opt.child[1].zero|opt.child[1].one);
  int first=b0>=b1?0:1;
  if(solve(opt.child[first]) && solve(opt.child[1-first])){
    cache.emplace(s,opt.marker);return true;
  }
 }
 cache.emplace(s,-1);return false;
}
size_t tree_nodes=0;
void emit(ostream&out,State s){
 tree_nodes++;
 if(resolved(s)){out<<"{\"shift\":"<<resolvedshift(s)<<"}";return;}
 auto g=canonical(s);auto i=cache.find(g.state);
 if(i==cache.end()||i->second<0 || i->second>=N)throw runtime_error("missing choice while extracting tree");
 int m=(invunit[g.ui]*(i->second-g.c+N))%N;
 out<<"{\"query\":"<<m<<",\"children\":{";
 bool first=true;
 for(int y=0;y<2;y++){
  auto child=successor(s,m,y);
  if(!(child.zero|child.one))continue;
  if(!first)out<<",";first=false;
  out<<'"'<<y<<"\":";emit(out,child);
 }
 out<<"}}";
}
void setup(){
 if(Q>MAXQ || Q<0)throw runtime_error("bad Q");
 for(int d=0;d<N;d++)for(int z=0;z<N;z++)DM[d]|=1ULL<<(d*N+z);
 for(int t=0;t<Q;t++)for(int m=0;m<N;m++)for(int d=0;d<N;d++)for(int z=0;z<N;z++)
   if((z+t*d)%N==m)HIT[t][m]|=1ULL<<(d*N+z);
 for(int t=0;t<=Q;t++)for(int a=0;a<W;a++)for(int b=0;b<W;b++){
  for(int i=t;i<Q;i++)FUT[t][a][b]+=((a%N+i*(a/N))%N)!=((b%N+i*(b/N))%N);
 }
 for(int ui=0;ui<U;ui++)for(int c=0;c<N;c++)for(int mask=0;mask<(1<<N);mask++){
  uint8_t result=0;
  for(int z=0;z<N;z++)if((mask>>z)&1)result|=1<<((units[ui]*z+c)%N);
  rowLut[ui][c][mask]=result;
 }
}
int main(int argc,char**argv){
 if(argc>1)Q=atoi(argv[1]);
 if(argc>2)LIMIT=strtoull(argv[2],nullptr,10);
 if(argc>3)heuristic=atoi(argv[3]);
 if(argc>4)USE_AFFINE=atoi(argv[4])!=0;
 if(argc>5)USE_PAIR=atoi(argv[5])!=0;
 setup();cache.reserve(min(LIMIT,(size_t)2000000));started=chrono::steady_clock::now();
 try{
  bool found=solve({ALL,0,0});
  double seconds=chrono::duration<double>(chrono::steady_clock::now()-started).count();
  cout<<"STATUS "<<(found?"SAT":"UNSAT")<<" n="<<N<<" Q="<<Q<<" heuristic="<<heuristic<<" affine="<<USE_AFFINE<<" pair="<<USE_PAIR<<" states="<<expanded<<" memo="<<cache.size()<<" pair_pruned="<<pair_pruned<<" seconds="<<seconds<<"\n";
  if(found){string file=argc>6?argv[6]:"oic28_n7_tree.json";ofstream out(file);emit(out,{ALL,0,0});out<<"\n";cout<<"TREE "<<file<<" nodes="<<tree_nodes<<"\n";}
 }catch(exception const&e){
  double seconds=chrono::duration<double>(chrono::steady_clock::now()-started).count();
  cerr<<"STATUS CAP n="<<N<<" Q="<<Q<<" states="<<expanded<<" pair_pruned="<<pair_pruned<<" seconds="<<seconds<<" msg="<<e.what()<<"\n";return 2;
 }
 return 0;
}
