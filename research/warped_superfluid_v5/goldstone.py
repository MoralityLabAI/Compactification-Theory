"""WS-05: local flat-space condensed-charge equilibrium and its two branches.

This is a SEPARATE controlled homogeneous benchmark.  The actual WS-04
charged warped background is spatially inhomogeneous, time dependent, and
is not asserted to be in this equilibrium state.

L=1/2 (partial x)^2+1/2 (partial v)^2 - V,
V=m2*(x*x+v*v)/2+lam*(x*x+v*v)**2/4.
x+iv=F exp(i omega*t).  Equilibrium: omega**2=m2+lam*F**2.
The quadratic radial/phase determinant reads
(p**2+2 lam F**2 - E**2)*(p**2 - E**2) - 4 omega**2 E**2=0.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HomogeneousCondensate:
    F: float=0.02
    mass2: float=0.5
    coupling: float=0.3

    def __post_init__(self):
        if self.F<=0 or self.coupling<=0 or self.mass2<0:
            raise ValueError('Require F>0, coupling>0, mass²>=0 for this benchmark')

    @property
    def equilibrium_frequency(self):
        return float(np.sqrt(self.mass2+self.coupling*self.F**2))

    @property
    def radial_curvature(self):
        return 2*self.coupling*self.F**2

    @property
    def gap_squared(self):
        return self.radial_curvature+4*self.equilibrium_frequency**2

    @property
    def goldstone_sound_squared(self):
        return self.radial_curvature/self.gap_squared

    def omega_squared_branches(self, momenta):
        p=np.asarray(momenta,dtype=float)
        if np.any(p<0):
            raise ValueError('momentum magnitude must be >= 0')
        m2=self.radial_curvature
        omega2=self.equilibrium_frequency**2
        disc=np.sqrt((m2+4*omega2)**2+16*omega2*p*p)
        upper=p*p+0.5*(m2+4*omega2+disc)
        # Stable evaluation avoids catastrophic cancellation for tiny momenta.
        lower=(p*p*(p*p+m2))/upper
        return lower,upper

    def determinant(self,momentum,energy_squared):
        p=np.asarray(momentum)
        x=np.asarray(energy_squared)
        return ((p*p+self.radial_curvature-x)*(p*p-x)
                -4*self.equilibrium_frequency**2*x)

    def local_frequency_gap(self,omega_coordinate,B):
        """Local diagnostic in warped conformal gauge; not a local equilibrium theorem.

        For x=F, y=0, xt=0, yt=omega_coord*F, observer at fixed y sees
        theta_dot_proper=e^-B omega_coord.  Curved/gradient terms affect the
        complete radial equilibrium equation.
        """
        return float(np.exp(-B)*omega_coordinate-self.equilibrium_frequency)
