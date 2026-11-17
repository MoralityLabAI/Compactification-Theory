// OIC-27 research: exact n=6 one-substitution adaptive binary sensor game.
// This solver uses two 36-bit candidate masks; no 32-bit truncation.
// CLI: horizon node_cap heuristic pair_prune tree_path (tree_path only for SAT)
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
constexpr int N=6, W=N*N, MAXQ=40;
constexpr uint64_t ALL=(1ULL<<W)-1ULL, ROW=(1ULL<<N)-1ULL;
struct State {uint64_t zero,one;};
struct Key {uint64_t zero,one;uint8_t t;
 bool operator==(Key const& rhs)const{return zero==rhs.zero&&one==rhs.one&&t==rhs.t;}
};
struct Hash { size_t operator()(Key const& k) const noexcept {
 uint64_t a=k.zero * 0x9e3779b97f4a7c15ULL;
 uint64_t b=k.one * 0xbf58476d1ce4e5b9ULL;
 uint64_t c=(uint64_t)k.t * 0x94d049bb133111ebULL;
 uint64_t x=a^(b+(a<<6)+(a>>2))^c;
 x ^= x>>30; x*=0xbf58476d1ce4e5b9ULL; x^=x>>27; x*=0x94d049bb133111ebULL; x^=x>>31;
 return (size_t)x;
}};
struct Canon {State s; int offset; Key k;};
struct Action {int m,score,skew;State children[2];};
int Q=14,heuristic=1; bool use_pair=true;
uint64_t HIT[MAXQ][N]{}, DM[N]{};
uint8_t FUT[MAXQ+1][W][W]{};
unordered_map<Key,int8_t,Hash> memo;
size_t cap=1500000, visited=0, pairs=0, terminal_count=0;
chrono::steady_clock::time_point started;
inline int pop(uint64_t x){return __builtin_popcountll(x);}
inline uint64_t shift_rows(uint64_t mask,int c){
 if(!c) return mask;
 uint64_t ans=0;
 for(int d=0;d<N;d++){
  uint64_t row=(mask>>(d*N))&ROW;
  ans|=(((row<<c)|(row>>(N-c)))&ROW)<<(d*N);
 }
 return ans;
}
Canon canon(State s,int t){
 State best=s;int off=0;
 for(int c=1;c<N;c++){
  State v{shift_rows(s.zero,c),shift_rows(s.one,c)};
  if(v.zero<best.zero||(v.zero==best.zero&&v.one<best.one)){best=v;off=c;}
 }
 return {best,off,{best.zero,best.one,(uint8_t)t}};
}
inline int shifts(uint64_t mask){int r=0;for(int d=0;d<N;d++)r+=((mask&DM[d])!=0);return r;}
inline bool terminal(State s){return shifts(s.zero|s.one)<=1;}
inline int terminal_shift(State s){uint64_t mask=s.zero|s.one;for(int d=0;d<N;d++) if(mask&DM[d])return d;return -1;}
inline State next(State s,int t,int m,int y){
 uint64_t hits=HIT[t][m],match=y?hits:(ALL^hits),mismatch=ALL^match;
 return {s.zero&match,(s.one&match)|(s.zero&mismatch)};
}
bool pair_obstructed(State s,int t){
 int ids[W],budgets[W],count=0;
 uint64_t bits=s.zero|s.one;
 while(bits){int id=__builtin_ctzll(bits);bits &= bits-1;
  ids[count]=id;budgets[count]=((s.zero>>id)&1ULL)?1:0;count++;}
 for(int i=0;i<count;i++) for(int j=i+1;j<count;j++) {
  int a=ids[i],b=ids[j];if(a/N==b/N)continue;
  if(FUT[t][a][b] <= budgets[i]+budgets[j])return true;
 }
 return false;
}
bool solve(int t,State state){
 auto c=canon(state,t);
 auto f=memo.find(c.k);if(f!=memo.end())return f->second>=0;
 if(++visited>cap)throw runtime_error("node cap");
 State s=c.s;
 if(terminal(s)){memo.emplace(c.k,N);terminal_count++;return true;}
 if(t>=Q){memo.emplace(c.k,-1);return false;}
 if(use_pair&&pair_obstructed(s,t)){memo.emplace(c.k,-1);pairs++;return false;}
 vector<Action> acts;acts.reserve(N);
 for(int m=0;m<N;m++){
  Action a; a.m=m;a.children[0]=next(s,t,m,0);a.children[1]=next(s,t,m,1);
  int count0=pop(a.children[0].zero|a.children[0].one),count1=pop(a.children[1].zero|a.children[1].one);
  int shift0=shifts(a.children[0].zero|a.children[0].one),shift1=shifts(a.children[1].zero|a.children[1].one);
  if(heuristic==0)a.score=m;
  else if(heuristic==1)a.score=max(count0,count1)*100+abs(count0-count1);
  else if(heuristic==2)a.score=(shift0*shift0+shift1*shift1)*100+count0*count0+count1*count1;
  else if(heuristic==3)a.score=max(shift0,shift1)*1000+max(count0,count1);
  else a.score=shift0*shift0+shift1*shift1;
  a.skew=abs(count0-count1);acts.push_back(a);
 }
 if(heuristic) stable_sort(acts.begin(),acts.end(),[](auto const&a,auto const&b){return a.score<b.score;});
 for(auto &a:acts){
  int count0=pop(a.children[0].zero|a.children[0].one),count1=pop(a.children[1].zero|a.children[1].one);
  int branch=count0>=count1?0:1;
  if(solve(t+1,a.children[branch])&&solve(t+1,a.children[1-branch])){
    memo.emplace(c.k,(int8_t)a.m);return true;
  }
 }
 memo.emplace(c.k,-1);return false;
}
void tree(ostream &out,int t,State state){
 if(terminal(state)){out<<"{\"shift\":"<<terminal_shift(state)<<"}";return;}
 auto c=canon(state,t);auto f=memo.find(c.k);
 if(f==memo.end()||f->second<0||f->second>=N)throw runtime_error("missing winning action");
 int m=(f->second-c.offset+N)%N;
 out<<"{\"query\":"<<m<<",\"children\":{";bool first=true;
 for(int y=0;y<2;y++){
  State child=next(state,t,m,y);
  if(!(child.zero|child.one))continue;
  if(!first)out<<",";first=false;
  out<<"\""<<y<<"\":";tree(out,t+1,child);
 }
 out<<"}}";
}
void init(){
 for(int d=0;d<N;d++)for(int z=0;z<N;z++)DM[d]|=1ULL<<(d*N+z);
 for(int t=0;t<Q;t++)for(int m=0;m<N;m++)for(int d=0;d<N;d++)for(int z=0;z<N;z++)
  if((z+t*d)%N==m)HIT[t][m]|=1ULL<<(d*N+z);
 for(int t=0;t<=Q;t++)for(int a=0;a<W;a++)for(int b=0;b<W;b++){
  int v=0;for(int i=t;i<Q;i++)v+=((a%N+i*(a/N))%N)!=((b%N+i*(b/N))%N);
  FUT[t][a][b]=v;
 }
}
int main(int argc,char**argv){
 if(argc>1)Q=atoi(argv[1]);if(argc>2)cap=strtoull(argv[2],nullptr,10);
 if(argc>3)heuristic=atoi(argv[3]);if(argc>4)use_pair=atoi(argv[4])!=0;
 if(Q<0||Q>MAXQ)throw runtime_error("invalid horizon");
 init();memo.reserve(min((size_t)2000000,cap));started=chrono::steady_clock::now();
 try {
  bool feasible=solve(0,{ALL,0});
  auto sec=chrono::duration<double>(chrono::steady_clock::now()-started).count();
  cout<<"STATUS "<<(feasible?"SAT":"UNSAT")<<" Q="<<Q<<" visited="<<visited
      <<" memo="<<memo.size()<<" pairprune="<<pairs<<" terminal="<<terminal_count
      <<" time="<<setprecision(5)<<sec<<"s"<<endl;
  if(feasible){
   string file=argc>5?argv[5]:("oic27_n6_q"+to_string(Q)+"_tree.json");
   ofstream out(file);tree(out,0,{ALL,0});out<<"\n";cout<<"TREE "<<file<<"\n";
  }
 }catch(exception const&e){
  auto sec=chrono::duration<double>(chrono::steady_clock::now()-started).count();
  cerr<<"STATUS CAP "<<e.what()<<" Q="<<Q<<" visited="<<visited<<" memo="<<memo.size()
      <<" pairprune="<<pairs<<" time="<<sec<<"s"<<endl;return 2;
 }
}
