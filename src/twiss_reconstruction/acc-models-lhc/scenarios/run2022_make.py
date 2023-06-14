from lhcmodels import LHCRun, LHCCycle, LHCBeamProcess, LHCKnobs
from lhcmodels import lsa


knobs = LHCKnobs.from_file("../operation/knobs.txt")
knobs.add_knob("nrj", "LHCBEAM/MOMENTUM", 1.0, 1)

lhcrun = LHCRun(2022)
lhcrun.read_cycles()
lhcrun.save_models(knobs)

