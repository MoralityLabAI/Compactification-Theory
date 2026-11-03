"""Goldberger-Wise probe-limit *illustration* of brane-distance stabilization.

The leading small-backreaction schematic potential is proportional to
 exp(-4 z) [v_IR - v_UV exp(-epsilon z)]², z=kL.
This establishes a local stationary minimum and does NOT enforce the
full 5D Israel junctions, bulk scalar stress, or radion normalization.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GoldbergerWiseProbe:
    epsilon:float=0.1
    z_star:float=35.0
    v_uv:float=1.0

    def __post_init__(self):
        if self.epsilon<=0 or self.z_star<=0 or self.v_uv<=0:
            raise ValueError('epsilon, target kL and v_uv must be positive')

    @property
    def v_ir(self):
        return self.v_uv*np.exp(-self.epsilon*self.z_star)

    def stabilized_separation(self):
        return float(np.log(self.v_uv/self.v_ir)/self.epsilon)

    def normalized_potential(self,z):
        """P(z)=exp(4*z_star) V(z)/prefactor, to avoid small numbers."""
        z=np.asarray(z,dtype=float)
        return np.exp(-4*(z-self.z_star))*(self.v_ir-self.v_uv*np.exp(-self.epsilon*z))**2

    def curvature_at_minimum(self):
        return 2*(self.epsilon*self.v_ir)**2
