from pathlib import Path

import numpy as np
import pjlsa
import pytimber

from  pjlsa.pjlsa import TrimTuple

def parseknobs():
    out={}
    for line in open("operation/knobs.txt"):
        if not line.strip().startswith("#") and len(line)>2:
            madname,lsaname,fact,test=line.strip().split(',')
            out[lsaname.strip()]=(madname,float(fact))
    return out



def get_trim_table(bp,params):
  ot=lsa.getOpticTable(bp)
  trims={}
  for pn in params:
    try:
        trim=lsa.getLastTrim(parameter=pn,beamprocess=bp,part="target")
        trims[pn]=trim
    except Exception as ex:
        print(f"Error in {pn}")
  return ot,trims


def mkbeam(nrj):
    return f"""
beam, sequence=lhcb1, bv= 1,
  particle=proton, charge=1, mass=0.938272046,
  energy= {nrj},   npart=1.2e11,kbunch=2556,
  exn=3.5e-6,eyn=3.5e-6;
beam, sequence=lhcb2, bv=-1,
  particle=proton, charge=1, mass=0.938272046,
  energy= {nrj},   npart=1.2e11,kbunch=2556,
  exn=3.5e-6,eyn=3.5e-6;
  """


db=pytimber.LoggingDB()
lsa = pjlsa.LSAClient()

knobs=parseknobs()

allparams=[
 'LHCBEAM/MOMENTUM',
 'LHCBEAM/IP1-XING-H-MURAD',
 'LHCBEAM/IP2-XING-H-MURAD',
 'LHCBEAM/IP5-XING-H-MURAD',
 'LHCBEAM/IP8-XING-H-MURAD',
 'LHCBEAM/IP1-XING-V-MURAD',
 'LHCBEAM/IP2-XING-V-MURAD',
 'LHCBEAM/IP5-XING-V-MURAD',
 'LHCBEAM/IP8-XING-H-MURAD',
 'LHCBEAM/IP1-SEP-H-MM',
 'LHCBEAM/IP2-SEP-H-MM',
 'LHCBEAM/IP5-SEP-H-MM',
 'LHCBEAM/IP8-SEP-H-MM',
 'LHCBEAM/IP1-SEP-V-MM',
 'LHCBEAM/IP2-SEP-V-MM',
 'LHCBEAM/IP5-SEP-V-MM',
 'LHCBEAM/IP8-SEP-V-MM',
 'LHCBEAM/IP1-OFFSET-H-MM',
 'LHCBEAM/IP2-OFFSET-H-MM',
 'LHCBEAM/IP5-OFFSET-H-MM',
 'LHCBEAM/IP8-OFFSET-H-MM',
 'LHCBEAM/IP1-OFFSET-V-MM',
 'LHCBEAM/IP2-OFFSET-V-MM',
 'LHCBEAM/IP5-OFFSET-V-MM',
 'LHCBEAM/IP8-OFFSET-V-MM',
 'LHCBEAM/IP1-ANGLE-H-MURAD',
 'LHCBEAM/IP2-ANGLE-H-MURAD',
 'LHCBEAM/IP5-ANGLE-H-MURAD',
 'LHCBEAM/IP8-ANGLE-H-MURAD',
 'LHCBEAM/IP1-ANGLE-V-MURAD',
 'LHCBEAM/IP2-ANGLE-V-MURAD',
 'LHCBEAM/IP5-ANGLE-V-MURAD',
 'LHCBEAM/IP8-ANGLE-H-MURAD',
 'LHCBEAM1/QPH',
 'LHCBEAM1/QPV',
 'LHCBEAM1/TELE_QPH',
 'LHCBEAM1/TELE_QPV',
 'LHCBEAM2/QPH',
 'LHCBEAM2/QPV',
 'LHCBEAM2/TELE_QPH',
 'LHCBEAM2/TELE_QPV',
 'LHCBEAM2/LANDAU DAMPING']


bp="RAMP_PELP-SQUEEZE-6.5TeV-ATS-1m-2018_V3_V1"
params=
['LHCBEAM/MOMENTUM',
 'LHCBEAM/IP1-XING-V-MURAD',
 'LHCBEAM/IP2-XING-V-MURAD',
 'LHCBEAM/IP5-XING-H-MURAD',
 'LHCBEAM/IP8-XING-H-MURAD',
 'LHCBEAM/IP1-SEP-H-MM',
 'LHCBEAM/IP2-SEP-H-MM',
 'LHCBEAM/IP5-SEP-V-MM',
 'LHCBEAM/IP8-SEP-V-MM',
 'LHCBEAM1/QPH',
 'LHCBEAM1/QPV',
 'LHCBEAM2/QPH',
 'LHCBEAM2/QPV',
 'LHCBEAM2/LANDAU DAMPING']

scenario="pp-lumi"
ot,trims=get_trim_table("RAMP_PELP-SQUEEZE-6.5TeV-ATS-1m-2018_V3_V1",params)
dqx,dqy=-0.04,-0.025
trims["LHCBEAM1/QH_TRIM"]=TrimTuple([0.,0.,1210],[dqx,dqx,0])
trims["LHCBEAM2/QH_TRIM"]=TrimTuple([0.,0.,1210],[dqx,dqx,0])
trims["LHCBEAM1/QV_TRIM"]=TrimTuple([0.,0.,1210],[dqy,dqy,0])
trims["LHCBEAM2/QV_TRIM"]=TrimTuple([0.,0.,1210],[dqy,dqy,0])

steps=[o.time for o in ot]

for k,v in trims.items():
    trims[k]=np.interp(steps,trim.data[0],trim.data[1])

nrj=trims['LHCBEAM/MOMENTUM']
del trims['LHCBEAM/MOMENTUM']
for iii,(time,opid,opname) in enumerate(ot):
    basedir=f"scenarios/{scenario}"
    out=[]
    out.append('call,file="acc-models-lhc/lhc.seq";')
    out.append(mkbeam(nrj[iii]))
    for k,v in trims.items():
        print(iii,k,knobs[k],v)









