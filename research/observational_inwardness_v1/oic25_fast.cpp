// OIC-25 experimental exact five-phase robust sensor search.
// Requires n=5 and e=1 raw bit substitution; no erasures.
// Every state stores candidate hidden worlds with 0 or 1 faults used.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
using namespace std;
struct State { uint32_t zero, one; };
struct Cand { State s; int c; uint64_t key; };
static constexpr int n=5,worlds=25;
static constexpr uint32_t ALL=(1u<<25)-1u;
int Q=12;
uint32_t HIT[32][5]{};
uint32_t DM[5]{};
uint8_t FUT[33][25][25]{};
unordered_map<uint64_t,int8_t> memo;
size_t node_cap=30000000,nodes=0,failedPairs=0;
int heuristicMode=1;bool usePairPrune=true;
chrono::steady_clock::time_point start;
inline int pop(uint32_t x){return __builtin_popcount(x);}
inline uint32_t rotateMask(uint32_t mask,int c){
    if(!c) return mask;
    uint32_t ans=0;
    for(int d=0;d<n;d++){
        uint32_t bits=(mask>>(d*n))&31u;
        uint32_t rot=((bits<<c)|(bits>>(n-c)))&31u;
        ans |= rot<<(d*n);
    }
    return ans;
}
inline uint64_t pack(int t,State s){return ((uint64_t)t<<50) | ((uint64_t)s.zero<<25)|s.one;}
inline Cand canon(State s,int t){
    uint64_t best=pack(t,s);int c=0;State ret=s;
    for(int i=1;i<5;i++){
        State now{rotateMask(s.zero,i),rotateMask(s.one,i)};
        uint64_t k=pack(t,now);
        if(k<best){best=k;c=i;ret=now;}
    }
    return {ret,c,best};
}
inline bool terminal(State s){
    uint32_t combined=s.zero|s.one;
    int total=0;
    for(int d=0;d<5;d++) if(combined&DM[d]) total++;
    return total<=1;
}
inline int terminalShift(State s){
    uint32_t combined=s.zero|s.one;
    for(int d=0;d<5;d++) if(combined&DM[d]) return d;
    return -1;
}
inline State successor(State s,int t,int m,int y){
    uint32_t hits=HIT[t][m],a=y?hits:(ALL^hits),notA=ALL^a;
    return {s.zero&a, (s.one&a) | (s.zero&notA)};
}
inline bool impossibleByPairs(State s,int t){
    // Two candidate worlds from different shifts can remain mutually
    // compatible if every possible future address sequence produces
    // <= remaining error budget sum distinguishing clean reports.
    int inds[25];int counts[25];int N=0;
    uint32_t unionMask=s.zero|s.one;
    while(unionMask){int id=__builtin_ctz(unionMask);unionMask&=unionMask-1;inds[N]=id;counts[N]=(s.zero>>id)&1;N++;}
    for(int i=0;i<N;i++)for(int j=i+1;j<N;j++){
        int a=inds[i],b=inds[j];
        if(a/n==b/n)continue;
        if(FUT[t][a][b] <= counts[i]+counts[j])return true;
    }
    return false;
}
struct Action{int m;int score;State children[2];};
bool search(int t,State initial){
    auto c=canon(initial,t);
    auto found=memo.find(c.key); if(found!=memo.end())return found->second>=0;
    if(++nodes>node_cap) throw runtime_error("node cap exceeded");
    State s=c.s;
    if(terminal(s)){memo[c.key]=5;return true;}
    if(t>=Q){memo[c.key]=-1;return false;}
    if(usePairPrune && impossibleByPairs(s,t)){failedPairs++;memo[c.key]=-1;return false;}
    vector<Action> actions;
    for(int m=0;m<5;m++){
        Action act;act.m=m;
        act.children[0]=successor(s,t,m,0);
        act.children[1]=successor(s,t,m,1);
        if(heuristicMode==0) act.score=m;
        else {
            const auto a=act.children[0],b=act.children[1];
            int total0=pop(a.zero)+pop(a.one), total1=pop(b.zero)+pop(b.one);
            int shifts0=0,shifts1=0;
            for(int d=0;d<5;d++) {shifts0+=!!((a.zero|a.one)&DM[d]);shifts1+=!!((b.zero|b.one)&DM[d]);}
            if(heuristicMode==1) act.score= max(total0,total1)*100 + abs(total0-total1);
            else if(heuristicMode==2)act.score=(shifts0*shifts0+shifts1*shifts1)*100 + total0*total0+total1*total1;
            else act.score= max(shifts0,shifts1)*100 + max(total0,total1);
        }
        actions.push_back(act);
    }
    if(heuristicMode)stable_sort(actions.begin(),actions.end(),[](const Action&a,const Action&b){return a.score<b.score;});
    for(auto &a:actions){
        int branch=(pop(a.children[0].zero|a.children[0].one) >= pop(a.children[1].zero|a.children[1].one))?0:1;
        if(search(t+1,a.children[branch]) && search(t+1,a.children[1-branch])){
            memo[c.key]=(int8_t)a.m;
            return true;
        }
    }
    memo[c.key]=-1;
    return false;
}
void outputTree(ostream &out,int t,State state){
    if(terminal(state)){out<<"{\"shift\":"<<terminalShift(state)<<"}";return;}
    auto c=canon(state,t);
    auto f=memo.find(c.key);
    if(f==memo.end()||f->second<0||f->second>4) throw runtime_error("missing winning action during extraction");
    int m=(f->second-c.c+n)%n;
    out<<"{\"query\":"<<m<<",\"children\":{";
    bool first=true;
    for(int y=0;y<2;y++){
        auto child=successor(state,t,m,y);
        if(!(child.zero|child.one))continue;
        if(!first)out<<",";
        first=false;
        out<<"\""<<y<<"\":";outputTree(out,t+1,child);
    }
    out<<"}}";
}
void init(){
    for(int d=0;d<5;d++) for(int z=0;z<5;z++) DM[d]|=(1u<<(d*5+z));
    for(int t=0;t<Q;t++)for(int m=0;m<5;m++){
       uint32_t mask=0;
       for(int d=0;d<5;d++)for(int z=0;z<5;z++)if((z+t*d)%5==m)mask|=(1u<<(d*5+z));
       HIT[t][m]=mask;
    }
    for(int t=0;t<=Q;t++)for(int a=0;a<25;a++)for(int b=0;b<25;b++){
        int cnt=0;
        for(int i=t;i<Q;i++)if(((a%5+i*(a/5))%5)!=((b%5+i*(b/5))%5))cnt++;
        FUT[t][a][b]=cnt;
    }
}
int main(int argc,char**argv){
    if(argc>1) Q=atoi(argv[1]);
    if(argc>2) node_cap=strtoull(argv[2],nullptr,10);
    if(argc>3) heuristicMode=atoi(argv[3]);
    if(argc>4) memo.reserve(strtoull(argv[4],nullptr,10));
    if(argc>5) usePairPrune=(atoi(argv[5])!=0);
    if(Q>32 || Q<0) throw runtime_error("Q out of range");
    init();start=chrono::steady_clock::now();
    try{
      bool ok=search(0,{ALL,0});
      double sec=chrono::duration<double>(chrono::steady_clock::now()-start).count();
      cout<<"STATUS "<<(ok?"SAT":"UNSAT")<<" Q="<<Q<<" nodes="<<nodes<<" memo="<<memo.size()<<" pairprune="<<failedPairs<<" seconds="<<sec<<endl;
      if(ok) {
        string path=(argc>6 ? argv[6] : "winner_n5_q"+to_string(Q)+".json");
        ofstream out(path);outputTree(out,0,{ALL,0});out<<'\n';cout<<"TREE "<<path<<endl;
      }
    }catch(const exception& ex){
      double sec=chrono::duration<double>(chrono::steady_clock::now()-start).count();
      cerr<<"STATUS CAP "<<ex.what()<<" Q="<<Q<<" nodes="<<nodes<<" memo="<<memo.size()<<" pairprune="<<failedPairs<<" seconds="<<sec<<endl;
      return 2;
    }
}
