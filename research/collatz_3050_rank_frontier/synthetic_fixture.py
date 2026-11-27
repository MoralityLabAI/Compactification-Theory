"""Explicit synthetic BF16 fixtures; no model results."""
import hashlib
import json
import struct
import numpy as np
from activation_io import ARMS, ROLES, WIDTH


def arrays():
    out={}
    for i,role in enumerate(ROLES):
        a=np.zeros((1,3,WIDTH),np.float32)
        a[0,:2,i]=1
        out[role+".pass1_boundary"]=a
        q=np.zeros(WIDTH,np.float32)
        q[i]=1
        out[role+".final_query"]=q
    for role in ARMS:
        q=out["recipient.final_query"].copy()
        if role=="effect":
            q=out["effect_donor.final_query"].copy()
        elif role=="full_donor_prefix":
            q=out["effect_donor.final_query"].copy()
        out["arm."+role+".final_query"]=q
    return out


def write(path,data):
    header={}
    payload=bytearray()
    for key,a in data.items():
        b=(a.astype(np.float32).view(np.uint32)>>16).astype("<u2").tobytes()
        header[key]={"dtype":"BF16","shape":list(a.shape),"data_offsets":[len(payload),len(payload)+len(b)]}
        payload.extend(b)
    h=json.dumps(header,separators=(",",":")).encode()
    raw=struct.pack("<Q",len(h))+h+payload
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
