from cpymad.madx import Madx
import re
import pjlsa

lsa=pjlsa.LSAClient()


kwd=re.compile("[_A-z][_\.A-z0-9]+")

op="acc-models-lhc/operation/optics/R2018i_A590C590A590L590.madx"

madx=Madx()
madx.call(op)

knobs={}
for vname in madx.globals:
    expr=madx.globals.cmdpar[vname].expr
    if expr is not None:
        print(vname,expr)
        for knob in kwd.findall(expr):
            knobs.setdefault(knob,set()).add(vname)

for knob in sorted(knobs):
    print(knob)

