import xml.etree.ElementTree as ET
from pathlib import  Path
import os

tree = ET.parse('operation/lhc-2018.jmdold.xml')
tree = ET.parse('operation/lhc-2020.jmdold.xml')

# Michi: uncomment to enable copying
#opstr=None
opbase=Path("/home/rdemaria/local/jmad-modelpack-lhc/src/java/cern/accsoft/steering/jmad/modeldefs/defs/")
opstr=opbase/"repdata/2020/"

root=tree.getroot()
optics=root.getchildren()[0]

for op in optics.getchildren():
    opname=op.attrib["name"]
    with open(f"operation/optics/{opname}.madx","w") as fh:
        print(fh.name)
        for st in op.getchildren()[0]:
           #print(st.attrib)
           path=Path(st.attrib["path"])
           if "location" in st.attrib:
               npath="acc-models-lhc/toolkit"/path
               fh.write(f'call,file="{npath}";\n')
           else:
               npath="acc-models-lhc/"/path
               fh.write(f'call,file="{npath}";\n')
               if opstr is not None:
                   path.parent.mkdir(parents=True, exist_ok=True)
                   os.system(f"cp {opstr/path} {path}")
        knobpath=Path(f'operation/optics/{opname}_lsaknobs.madx')
        if knobpath.is_file():
            fh.write(f'call,file="acc-models-lhc/{knobpath}";\n')
        else:
            print(f">> not adding knobs file - {knobpath} does not exist !")


