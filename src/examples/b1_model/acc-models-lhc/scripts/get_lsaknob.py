import pjlsa
from cpymad.madx import Madx
from glob import glob
from pathlib import Path
import sys

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

lsa=pjlsa.LSAClient()

try:
  optic=sys.argv[1]
  lsaname=sys.argv[2]
  madname=sys.argv[3]
  knob=mkknob(lsa,optic,lsaname,madname)
  print(knob)
except Exception as ex:
    print("""Usage:
python get_lsaknob.py R2018i_A400C400A400L400 LHCBEAM1/IP1_SEPSCAN_Y_MM on_yip1b2""")
    raise ex
