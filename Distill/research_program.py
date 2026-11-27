"""DISTILL-06: public finite longitudinal research programs.

Four linked investigations, a shared observation nuisance, a transport premise,
local exact audits, and a strict out-of-scope project. All are synthetic models.
No provider output, hidden target feedback, or native scientific claims.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from math import prod
from .core import canonical, digest, object_keys, require, string
from .instances import Rng

SCHEMA = "distill.program.v1"
ACTION_SCHEMA = "distill.program_action.v1"
PROJECTS = ("anchor", "transfer_a", "transfer_b", "out_of_scope")
BITS = ("anchor", "law", "local_a", "local_b", "local_c", "nuisance")
RULES = (
    "Four projects are investigated in order: anchor, transfer_a, transfer_b, out_of_scope. "
    "A hidden world consists of six bits. Each bit is independent under the public prior; "
    "a row's unnormalized mass is the product of its six selected prior weights. "
    "Anchor truth is anchor. If law=1, transfer_a truth is anchor XOR offsets[0] and "
    "transfer_b truth is anchor XOR offsets[1]; if law=0 their truths are local_a and local_b. "
    "Out_of_scope truth is local_c regardless of law. "
    "Screen returns current truth XOR nuisance; for the shifted_control family at out_of_scope only, "
    "it instead returns truth XOR nuisance XOR law. Thus screens share a nuisance and must not be treated "
    "as independent new evidence. Control reveals nuisance, law_audit reveals law, direct_audit reveals "
    "current truth with a local certificate. Transport checks a positive law_audit receipt and a direct anchor "
    "certificate, then certifies the current in-scope truth using its offset; it is never legal out of scope. "
    "The law audit and control may each be performed once per program, screen once per project, "
    "direct_audit or transport only before obtaining a current certificate. All experiment costs are credits "
    "and equal utility deductions. Recommend:0/1 advances the project and earns correct_reward if right, "
    "wrong_reward if wrong. Certify:0/1 advances and earns certificate_reward only with a matching "
    "trusted certificate; defer advances for zero. Publication and recommendation truth/payoff is not revealed "
    "during the program. A final decision is immutable. Unused credits carry forward. "
    "The host lists legal actions, observations and certificate provenance. Optimize whole-program net utility, "
    "not just the immediate project. Do not claim these synthetic mechanisms are empirical findings."
)

@dataclass(frozen=True)
class World:
    bits: tuple[int, ...]
    mass: int
    truths: tuple[int, ...]

@dataclass(frozen=True)
class State:
    stage: int
    mask: int
    budget: int
    screened: bool = False
    control: int = -1
    law: int = -1
    anchor: int = -1
    cert: int = -1

def new_spec(seed, ordinal, family="shared_control", difficulty="frontier"):
    require(type(seed) is int and 0<=seed<2**63 and type(ordinal) is int and 0<=ordinal<10**7,
            "invalid program coordinates")
    require(family in ("shared_control","shifted_control") and difficulty in ("standard","frontier"),
            "unsupported program family or difficulty")
    r=Rng([SCHEMA,seed,ordinal,family,difficulty])
    # Positive, independently factored priors. Law is likely but not guaranteed.
    weights=[[r.between(1,4),r.between(1,4)] for _ in BITS]
    weights[1]=[r.between(1,3),r.between(3,6)]
    costs={"screen":1,"control":r.between(1,2),"law_audit":r.between(1,2),
           "direct_audit":3,"transport":1}
    spec={"schema_version":SCHEMA,"family":family,"difficulty":difficulty,
          "prior_weights":dict(zip(BITS,weights)),"offsets":[r.below(2),r.below(2)],
          "budget":r.between(8,10) if difficulty=="frontier" else r.between(10,12),
          "costs":costs,"correct_reward":6,"wrong_reward":-12,"certificate_reward":10,
          "scope":"synthetic public-development planning model"}
    validate_spec(spec)
    return spec

def validate_spec(spec):
    object_keys(spec,{"schema_version","family","difficulty","prior_weights","offsets","budget",
                      "costs","correct_reward","wrong_reward","certificate_reward","scope"},"program")
    require(spec["schema_version"]==SCHEMA and spec["family"] in ("shared_control","shifted_control")
            and spec["difficulty"] in ("standard","frontier"),"bad program schema")
    require(spec["scope"]=="synthetic public-development planning model","invalid scope")
    object_keys(spec["prior_weights"],set(BITS),"prior weights")
    for v in spec["prior_weights"].values():
        require(isinstance(v,list) and len(v)==2 and all(type(x) is int and 1<=x<=8 for x in v),"bad prior")
    require(isinstance(spec["offsets"],list) and len(spec["offsets"])==2
            and all(type(x) is int and x in (0,1) for x in spec["offsets"]),"bad offsets")
    require(type(spec["budget"]) is int and 1<=spec["budget"]<=14,"bad budget")
    object_keys(spec["costs"],{"screen","control","law_audit","direct_audit","transport"},"costs")
    require(all(type(v) is int and 1<=v<=6 for v in spec["costs"].values()),"bad costs")
    require(type(spec["correct_reward"]) is int and 1<=spec["correct_reward"]<=20 and
            type(spec["wrong_reward"]) is int and -40<=spec["wrong_reward"]<0 and
            type(spec["certificate_reward"]) is int and spec["correct_reward"]<spec["certificate_reward"]<=30,
            "bad rewards")

class Model:
    """Exact finite environment and public belief-state transition kernel."""
    def __init__(self,spec,*,allow_transport=True):
        validate_spec(spec)
        require(type(allow_transport) is bool,"bad transport switch")
        self.spec=spec
        self.allow_transport=allow_transport
        self.worlds=[]
        for bits in product((0,1),repeat=6):
            a,l,d1,d2,d3,n=bits
            mass=prod(spec["prior_weights"][name][b] for name,b in zip(BITS,bits))
            truths=(a,a^spec["offsets"][0] if l else d1,a^spec["offsets"][1] if l else d2,d3)
            self.worlds.append(World(bits,mass,truths))
        self.full_mask=(1<<len(self.worlds))-1
        self.truth_masks=tuple(self._mask(lambda w,j=j:w.truths[j]) for j in range(4))
        self.nuisance_mask=self._mask(lambda w:w.bits[5])
        self.law_mask=self._mask(lambda w:w.bits[1])
        self.screen_masks=tuple(self._mask(lambda w,j=j:w.truths[j]^w.bits[5]^
                (w.bits[1] if j==3 and spec["family"]=="shifted_control" else 0)) for j in range(4))
        self._mass_cache={0:0}
    def _mask(self,f):
        return sum(1<<i for i,w in enumerate(self.worlds) if f(w))
    def initial(self):
        return State(0,self.full_mask,self.spec["budget"])
    def mass(self,mask):
        if mask not in self._mass_cache:
            self._mass_cache[mask]=sum(w.mass for i,w in enumerate(self.worlds) if mask>>i&1)
        return self._mass_cache[mask]
    def cost(self,action):
        return self.spec["costs"].get(action,0)
    def legal(self,s):
        if s.stage==4:
            return ()
        a=["recommend:0","recommend:1","defer"]
        if s.cert>=0:
            a.append("certify:"+str(s.cert))
        def add(action,condition):
            if condition and self.cost(action)<=s.budget:
                a.append(action)
        add("screen",not s.screened)
        add("control",s.control<0)
        add("law_audit",s.law<0)
        add("direct_audit",s.cert<0)
        add("transport",self.allow_transport and s.stage in (1,2) and s.anchor>=0 and s.law==1 and s.cert<0)
        return tuple(a)
    def terminal(self,action):
        return action=="defer" or action.startswith(("recommend:","certify:"))
    def truth_mass(self,s,b):
        m=s.mask&self.truth_masks[s.stage]
        return self.mass(m if b else s.mask^m)
    def reward_sum(self,s,action):
        m=self.mass(s.mask)
        if action=="defer":
            return 0
        if action.startswith("certify:"):
            return m*self.spec["certificate_reward"]
        if action.startswith("recommend:"):
            good=self.truth_mass(s,int(action[-1]))
            return good*self.spec["correct_reward"]+(m-good)*self.spec["wrong_reward"]
        return -m*self.cost(action)
    def branches(self,s,action):
        """Return (outcome, next_state); each branch has a disjoint nonempty mask."""
        require(action in self.legal(s),"illegal research action")
        if self.terminal(action):
            return [(None,State(s.stage+1,s.mask,s.budget,control=s.control,law=s.law,anchor=s.anchor))]
        budget=s.budget-self.cost(action)
        if action=="transport":
            bit=s.anchor^self.spec["offsets"][s.stage-1]
            truth=self.truth_masks[s.stage]
            require((s.mask&truth if bit else s.mask&~truth)==s.mask,"invalid transport premise")
            return [(bit,replace(s,budget=budget,cert=bit))]
        mask={"screen":self.screen_masks[s.stage],"control":self.nuisance_mask,
              "law_audit":self.law_mask,"direct_audit":self.truth_masks[s.stage]}[action]
        out=[]
        for bit,part in ((0,s.mask&~mask),(1,s.mask&mask)):
            if not part:
                continue
            kwargs={"screen":{"screened":True},"control":{"control":bit},"law_audit":{"law":bit},
                    "direct_audit":{"cert":bit}}[action]
            if action=="direct_audit" and s.stage==0:
                kwargs["anchor"]=bit
            out.append((bit,replace(s,mask=part,budget=budget,**kwargs)))
        require(sum(t.mask for _,t in out)==s.mask,"partition failure")
        return out
    def realized_reward(self,s,action,world):
        if action=="defer":
            return 0
        if action.startswith("certify:"):
            require(int(action[-1])==world.truths[s.stage],"false certificate")
            return self.spec["certificate_reward"]
        if action.startswith("recommend:"):
            return self.spec["correct_reward"] if int(action[-1])==world.truths[s.stage] else self.spec["wrong_reward"]
        return -self.cost(action)

class Program:
    """Host-owned evidence ledger; actual world never appears in observations."""
    def __init__(self,spec,world_index,*,allow_transport=True):
        self.model=Model(spec,allow_transport=allow_transport)
        require(type(world_index) is int and 0<=world_index<len(self.model.worlds),"bad evaluator world")
        self._world_index=world_index
        self._world=self.model.worlds[world_index]
        self.state=self.model.initial()
        self.events=[]
        self._utility=0
        self.program_id=digest(spec)
    def _receipt(self,action,bit,stage):
        evidence={"program_id":self.program_id,"sequence":len(self.events),"project":PROJECTS[stage],
                  "action":action,"observation":bit}
        if action=="transport":
            refs=[]
            for kind in ("law_audit","direct_audit"):
                eligible=[e["receipt"]["receipt_id"] for e in self.events if e["action"]==kind
                          and (kind!="direct_audit" or e["project"]=="anchor")]
                require(len(eligible)==1,"transport missing receipt")
                refs+=eligible
            evidence["premises"]=refs
            evidence["scope"]=["transfer_a","transfer_b"]
        elif action=="law_audit":
            evidence["scope"]=["transfer_a","transfer_b"]
        elif action=="direct_audit":
            evidence["scope"]=[PROJECTS[stage]]
        return {**evidence,"receipt_id":digest(evidence)}
    def observe(self):
        s=self.state
        require(s.stage<4,"program complete")
        legal=[{"action":a,"cost":self.model.cost(a)} for a in self.model.legal(s)]
        x={"schema_version":"distill.program_observation.v1","program_id":self.program_id,
           "spec":self.model.spec,"rules":RULES,"project":PROJECTS[s.stage],"stage":s.stage,
           "remaining_credits":s.budget,"compatible_worlds":s.mask.bit_count(),
           "current_truth_masses":[self.model.truth_mass(s,b) for b in (0,1)],
           "ledger":self.events,"legal_actions":legal}
        # Copy the public snapshot. The evaluator index, true world and accumulated
        # terminal rewards are deliberately absent.
        import json
        x=json.loads(canonical(x))
        return {**x,"state_id":digest(x)}
    def step(self,action):
        before=self.observe()
        s=self.state
        require(action in self.model.legal(s),"illegal research action")
        found=[(bit,t) for bit,t in self.model.branches(s,action) if t.mask>>self._world_index&1]
        require(len(found)==1,"world branch mismatch")
        bit,t=found[0]
        self._utility+=self.model.realized_reward(s,action,self._world)
        event={"state_id":before["state_id"],"project":PROJECTS[s.stage],"action":action,
               "cost":self.model.cost(action),"observation":bit,"remaining_credits":t.budget,
               "receipt":self._receipt(action,bit,s.stage) if bit is not None else None}
        self.events.append(event)
        self.state=t
        return event
    def result(self):
        require(self.state.stage==4,"program not complete")
        return {"program_id":self.program_id,"net_utility":self._utility,
                "credits_spent":self.model.spec["budget"]-self.state.budget,
                "actions":len(self.events),"certified":sum(e["action"].startswith("certify:") for e in self.events),
                "transports":sum(e["action"]=="transport" for e in self.events),
                "false_recommendations":sum(e["action"].startswith("recommend:") and
                     int(e["action"][-1])!=self._world.truths[PROJECTS.index(e["project"])] for e in self.events),
                "deferred":sum(e["action"]=="defer" for e in self.events),
                "trace_sha256":digest(self.events),"scope":"synthetic evaluator result"}

def prompt(observation):
    return (
        "You are DataGoblin's research policy. Choose one legal next research action. "
        "Keep a coherent research agenda over all four projects; earlier evidence may be reused only within its checked scope. "
        "Account for costs, alternative hypotheses, uncertainty, and the value of preserving credits for later projects. "
        "Use short, checkable public explanations, not private reasoning. Do not assume a hidden world or invent an observation. "
        "Return JSON only: {schema_version:'distill.program_action.v1',state_id:copy from observation,"
        "action:one legal action,agenda:brief research plan,evidence:what the observed evidence establishes,"
        "decision_basis:expected value and budget tradeoff,revision_rule:which observation would change the next plan}. "
        "Use actual JSON double quotes, exactly these fields, and nonempty strings.\n\n"
        +canonical(observation).decode("ascii")
    )

def validate_action(observation,response):
    object_keys(response,{"schema_version","state_id","action","agenda","evidence","decision_basis","revision_rule"},"research action")
    require(response["schema_version"]==ACTION_SCHEMA and response["state_id"]==observation["state_id"],"stale or wrong action binding")
    require(isinstance(response["action"],str) and response["action"] in {a["action"] for a in observation["legal_actions"]},"illegal action")
    for key in ("agenda","evidence","decision_basis","revision_rule"):
        string(response[key],key,4000)
    canonical(response)
    return response["action"]
