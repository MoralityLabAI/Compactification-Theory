"""DISTILL-12 exact public-belief projection kernels and bounded cache.

Logical charges are declared operation counts, not CPU cycles. All reusable
data is scoped to one Model and one actor attempt; no cross-world warm cache.
"""
from collections import OrderedDict
from functools import reduce
from math import gcd
from .core import require
from .program_quotient import budget_cap

OPTIMIZATIONS=("bitset","cache","bitset_cache","bitset_cache_prune")
METRICS=("projection_work","setup_work","cache_hits","cache_misses",
         "cache_evictions","cache_peak","pruned")
CAPACITY=32


class Projector:
    def __init__(self,m,optimization,capacity=CAPACITY):
        require(optimization in OPTIMIZATIONS,"unknown projection optimization")
        require(type(capacity) is int and 1<=capacity<=CAPACITY,"invalid cache capacity")
        self.model=m
        self.bitset=optimization.startswith("bitset")
        self.cached="cache" in optimization
        self.capacity=capacity
        self.cache=OrderedDict()
        self.planes=None
        self.stage_masks={}
        self.stats={k:0 for k in METRICS}

    def _planes(self,m,meter):
        if self.planes is not None:
            return self.planes
        start=meter.used
        try:
            meter.charge(64)  # acquire public prior widths
            width=max(w.mass.bit_length() for w in m.worlds)
            meter.charge(8+4*width+64)  # allocate and visit world slots
            planes=[0]*width
            for i,w in enumerate(m.worlds):
                n=w.mass
                while n:
                    meter.charge(4)  # isolate bit, index, set and clear
                    bit=n&-n
                    planes[bit.bit_length()-1]|=1<<i
                    n^=bit
            meter.charge(4*width)
            result=tuple((1<<j,p) for j,p in enumerate(planes) if p)
            self.planes=result
            return result
        finally:
            self.stats["setup_work"]+=meter.used-start

    def _masks(self,m,stage,meter):
        if stage in self.stage_masks:
            return self.stage_masks[stage]
        start=meter.used
        try:
            meter.charge(8)
            masks=(m.full_mask,)
            truths=m.truth_masks[stage:]+((m.law_mask,) if stage<=1 else ())
            for bit in truths:
                meter.charge(8*len(masks))
                masks=tuple(t for p in masks for t in (p&~bit,p&bit))
            meter.charge(8)
            self.stage_masks[stage]=masks
            return masks
        finally:
            self.stats["setup_work"]+=meter.used-start

    def _scan(self,m,s,meter):
        bits=4-s.stage+int(s.stage<=1)
        # Same declared price as DISTILL-11; outer signature pays its fixed 16.
        meter.charge(64*(bits+2)+8*(1<<bits)+16)
        bins=[0]*(1<<bits)
        for i,w in enumerate(m.worlds):
            if s.mask>>i&1:
                k=0
                for bit in w.truths[s.stage:]:
                    k=2*k+bit
                if s.stage<=1:
                    k=2*k+w.bits[1]
                bins[k]+=w.mass
        return bins

    def _bitset(self,m,s,meter):
        planes=self._planes(m,meter)
        masks=self._masks(m,s.stage,meter)
        meter.charge(8+4*len(masks))
        bins=[]
        for mask in masks:
            part=mask&s.mask
            if not part:
                bins.append(0)
                continue
            meter.charge(4*len(planes))
            bins.append(sum(weight*(part&plane).bit_count()
                            for weight,plane in planes))
        meter.charge(8*len(bins)+16)
        return bins

    def project(self,m,s,meter):
        require(m is self.model and 0<=s.stage<4,"projection scope")
        start=meter.used
        try:
            key=(s.stage,s.mask)
            if self.cached:
                meter.charge(8)
                if key in self.cache:
                    meter.charge(8)
                    self.cache.move_to_end(key)
                    self.stats["cache_hits"]+=1
                    return self.cache[key]
                self.stats["cache_misses"]+=1
            bins=self._bitset(m,s,meter) if self.bitset else self._scan(m,s,meter)
            factor=reduce(gcd,bins)
            require(factor>0,"empty projected belief")
            result=tuple(x//factor for x in bins),factor
            if self.cached:
                full=len(self.cache)>=self.capacity
                meter.charge(8+len(bins)+(8 if full else 0))
                if full:
                    self.cache.popitem(last=False)
                    self.stats["cache_evictions"]+=1
                self.cache[key]=result
                self.stats["cache_peak"]=max(self.stats["cache_peak"],len(self.cache))
            return result
        finally:
            self.stats["projection_work"]+=meter.used-start


def prunable(m,s,a,meter):
    """Deterministic discarded-flag experiment at equal saturated budget.

    After a screen/control with a known outcome the mask, stage, law, anchor,
    and cert are unchanged. Equal capped budget implies equal continuation
    class. The extra experiment and additional incumbent selection have
    nonpositive utility, so no strictly profitable deviation exists.
    """
    if s.stage not in (1,2) or a not in ("screen","control"):
        return False
    meter.charge(8)
    mask=m.screen_masks[s.stage] if a=="screen" else m.nuisance_mask
    part=s.mask&mask
    if part not in (0,s.mask):
        return False
    cap=budget_cap(m,s)
    return min(s.budget,cap)==min(s.budget-m.cost(a),cap)
