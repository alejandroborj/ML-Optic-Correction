import pjlsa
from cpymad.madx import Madx
from glob import glob
from pathlib import Path

def str2mad(lsa,k):
    kk=k.split('/')[0]
    return lsa.findMadStrengthNameByPCName(kk)

def mkknob(lsa,optic,lsaname,madname):
    out=[]
    out.append(f"! start {lsaname}")
    try:
      fct=lsa.getKnobFactors(lsaname,optic)
      for kk,vv in fct.items():
         madv=str2mad(lsa,kk)
         out.append(f"add2expr,var={madv},expr={vv}*{madname};")
      out.append(f"! end {lsaname}\n")
      return "\n".join(out)
    except:
        return ""

knobs={}
for ii in 1,2,5,8:
    for xy in "XY":
        for bb in "12":
            lsaname=f"LHCBEAM{bb}/IP{ii}_ANGLESCAN_{xy}_MURAD"
            madname=f"on_p{xy}ip{ii}b{bb}".lower()
            knobs[lsaname]=madname
            lsaname=f"LHCBEAM{bb}/IP{ii}_SEPSCAN_{xy}_MM"
            madname=f"on_{xy}ip{ii}b{bb}".lower()
            knobs[lsaname]=madname


optics=[ Path(ff).stem for ff in glob("operation/optics/*")]


lsa=pjlsa.LSAClient()
for optic in optics:
    print(optic)
    with open(f"strengths/{optic}_lsaknobs.madx","w") as fh:
        for lsaname,madname in knobs.items():
            knob=mkknob(lsa,optic,lsaname,madname)
            if knob!="":
               print(lsaname,madname)
               fh.write(knob)


