import sys
from copy import deepcopy
import re
from pathlib import Path

import numpy as np

import pjlsa
import pytimber


lsa = pjlsa.LSAClient()
cals = pytimber.LoggingDB()


class LHCRun:
    def __init__(self, year):
        self.year = year
        self.t1 = f"{year}-01-01 00:00:00"
        self.t2 = f"{year}-12-31 23:59:59"
        # self.set_fills()
        self.cycle = {}

    def read_cycles(self, cycle_path=Path(".")):
        for cycle_name in open(cycle_path / "cycles.txt"):
            cycle_name = cycle_name.strip()
            self.cycle[cycle_name] = LHCCycle.from_dir(
                cycle_name, cycle_path / cycle_name
            )

    def save_models(self, knobs, cycle_path=Path(".")):
        for cycle_name, cycle in self.cycle.items():
            print(f"Saving {cycle_name}")
            cycle.save_models(knobs, cycle_path / cycle_name)

    def set_fills(self):
        self.fills = {}
        fills = lsa.findBeamProcessHistory(self.t1, self.t2, accelerator="lhc")
        for filln, bp_list in fills.items():
            # beam_processes=[(ts,bp.split('@')[0]) for ts,bp in bp_list]
            beam_processes = [(ts, bp) for ts, bp in bp_list]
            self.fills[filln] = LHCFill(filln, beam_processes)

    def find_beam_processes(self, regexp="", full=True):
        reg = re.compile(regexp)
        out = {}
        for filln, fill in self.fills.items():
            for ts, bp in fill.beam_processes:
                res = reg.match(bp)
                if res:
                    if full and "@" not in bp:
                        out.setdefault(bp, []).append(filln)
        return out

    def hist_beam_processes(self, regexp="", full=True):
        lst = self.find_beam_processes(regexp, full=full)
        return list(sorted((len(v), k) for k, v in lst.items()))

    def get_used_beamprocess(self):
        out = set()
        for fill in self.fills.values():
            out.update(fill.get_used_beamprocess())
        return out

    def __repr__(self):
        return f"LHCRun({self.year})"


class LHCCycle:
    """Sequence of beam process"""

    @classmethod
    def from_dir(cls, name, cycle_path):
        bplist = [
            l.strip() for l in open(cycle_path / "beam_processes.txt").readlines()
        ]
        return cls(name, bplist)

    def __init__(self, name, beam_processes):
        self.name = name
        self.beam_processes = [LHCBeamProcess(bp) for bp in beam_processes]

    def save_models(self, knobs, cycle_path):
        for bp in self.beam_processes:
            bp.save_models(knobs, cycle_path / bp.name)

    def get_fills(self, lhcrun):
        bp_to_match = [bp.name for bp in self.beam_processes]

        def match(fill):
            fillbp = set([bp.split("@")[0] for ts, bp in fill.beam_processes])
            return all([bp in fillbp for bp in bp_to_match])

        return sorted([ff.filln for ff in lhcrun.fills.values() if match(ff)])

    def __repr__(self):
        return f"LHCCyle({self.name!r})"


class LHCFill:
    def __init__(self, filln, beam_processes):
        self.filln = filln
        self.beam_processes = beam_processes

    def get_data(self):
        return cals.getLHCFillData(self.filln)

    def get_start(self):
        return self.beam_processes[0][0]

    def bp_in_fill(self, beam_process):
        for ts, bp in self.beam_processes:
            if beam_process == bp:
                return True
        else:
            return False

    def get_used_beamprocess(self, segments=False):
        out = set()
        for _, bp in self.beam_processes:
            if segments or "@" not in bp:
                out.add(bp)
        return out

    def __repr__(self):
        return f"LHCFill({self.filln})"


class LHCBeamProcess:
    def __init__(self, name):
        self.name = name

    def get_optic_table(self):
        return lsa.getOpticTable(self.name)

    def save_models(self, knobs, model_path):
        settings = knobs.get_settings()
        modelseq = self.get_modelseq(settings)
        modelseq.save_models(knobs, model_path)

    def get_modelseq(self, settings, presettings=None):
        optable = lsa.getOpticTable(self.name)

        models = []
        index = []
        for op in optable:
            models.append(LHCModel("lhc.seq", op.name, {}))
            index.append(op.time)

        modelseq = LHCModelSeq(index, models)

        trims = {}
        for pp in settings:
            try:
                print(f"Extracting last trim {pp} for {self.name}")
                trims[pp] = lsa.getLastTrim(pp, self.name, part="target")
            except:
                print(f"Error extracting last trim {pp} for {self.name}")

        for pp, trim in trims.items():
            indexes, values = trim.data
            modelseq.apply_trim(pp, indexes, values)

        self.modelseq = modelseq

        return modelseq

    def get_trims(self, params, lhcrun=None):
        import jpype

        if lhcrun is None:
            t1 = None
            t2 = None
        else:
            t1 = lhcrun.t1
            t2 = lhcrun.t2

        out = []
        for param in params:
            try:
                print(f"getting {param}")
                trims = lsa.getTrims(param, beamprocess=self.name, start=t1, end=t2)[
                    param
                ]
                for ts, trim in zip(*trims):
                    out.append([ts, param, trim])
            except jpype.JException as ex:
                print("Error extracting parameter '%s': %s" % (param, ex))
            except KeyError as ex:
                print("Empty response for '%s': %s" % (param, ex))
        out.sort()
        return out

    def get_model_history(self, params, lhcrun, presettings):
        optable = lsa.getOpticTable(self.name)

        models = []
        index = []
        for op in optable:
            predict = {"opname": op.name}
            predict.update(presettings)
            models.append(LHCModel(predict))
            index.append(op.time)

        models = LHCModels(index, models)

        trims = self.get_trims(params, lhcrun)

        for filln, fill in lhcrun.fills.items():
            if fill.bp_in_fill(self.name):
                trims.append([fill.get_start(), "FILLN", filln])

        trims.sort()

        out = {}

        cfill = 0
        for ts, setting, val in trims:
            print(cfill, ts, setting, val)
            if setting == "FILLN":
                out[cfill] = models.copy()
                cfill = val
            else:
                models.apply_trim(setting, val[0], val[1])
        out[cfill] = models.copy()
        return out

    def __repr__(self):
        return f"LHCBeamProcess({self.name!r})"


class LHCModelSeq:
    def __init__(self, steps, models):
        if len(steps) != len(models):
            raise ValueError
        self.steps = np.array(steps)
        self.models = models

    def save_models(self, knobs, models_path):
        for index, model in zip(self.steps, self.models):
            model.save_settings(knobs, models_path / str(index) / "settings.madx")
            model.save_model(models_path / str(index) / "model.madx")

    def apply_trim(self, name, index, values):
        ivalues = np.interp(self.steps, index, values)
        for value, model in zip(ivalues, self.models):
            model.settings[name] = value

    def apply_value(self, name, value):
        for model in self.models:
            model.settings[name] = value

    def copy(self):
        return LHCModels(self.index.copy(), [m.copy() for m in self.models])


class LHCModel:
    def __init__(self, sequence, optics, settings):
        self.sequence = sequence
        self.optics = optics
        self.settings = settings

    def copy(self):
        return LHCModel(self.settings.copy())

    def to_mad(self, knobs):
        out = []
        for k, v in sorted(self.settings.items()):
            if k in knobs.lsa:
                out.append(knobs.to_mad(k, v))
        return "\n".join(out)

    def save_settings(self, knobs, filename):
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as fh:
            fh.write(self.to_mad(knobs))

    def mk_beam(self, pc):
        return f"""
beam, sequence=lhcb1, bv= 1, particle=proton, charge=1, mass=0.938272046,
pc= {pc},   npart=1.2e11,kbunch=2556, ex=5.2126224777777785e-09,ey=5.2126224777777785e-09;
beam, sequence=lhcb2, bv=-1, particle=proton, charge=1, mass=0.938272046,
pc= {pc},   npart=1.2e11,kbunch=2556, ex=5.2126224777777785e-09,ey=5.2126224777777785e-09;

"""

    def save_model(self, filename):
        filename.parent.mkdir(parents=True, exist_ok=True)
        settings = filename.parent / "settings.madx"
        pc = self.settings["LHCBEAM/MOMENTUM"]
        with open(filename, "w") as fh:
            fh.write('call, file="acc-models-lhc/lhc.seq";\n')
            fh.write(self.mk_beam(pc))
            fh.write(f'call,file="acc-models-lhc/operation/optics/{self.optics}.madx";\n')
            fh.write(f'call,file="acc-models-lhc/scenarios/{settings}";\n')


class LHCKnobs:
    @classmethod
    def from_file(cls, fn):
        lst = []
        for ln in open(fn):
            try:
                lst.append(LHCKnob(*ln.strip().split(", ")))
            except:
                pass
        return cls(lst)

    def __init__(self, knob_list):
        self.mad = {}
        self.lsa = {}
        for knob in knob_list:
            self.mad[knob.mad] = knob
            self.lsa[knob.lsa] = knob

    def mad_value(self, lsa_name, lsa_value):
        return self.lsa[lsa_name].mad_value(lsa_value)

    def lsa_value(self, mad_name, mad_value):
        return self.mad[mad_name].lsa_value(mad_value)

    def to_mad(self, lsa_name, lsa_value):
        return self.lsa[lsa_name].to_mad(lsa_value)

    def add(self, knob):
        self.mad[knob.mad] = knob
        self.lsa[knob.lsa] = knob

    def remove(self, lsa=None, mad=None):
        if lsa is not None:
            knob = self.lsa[lsa]
        elif mad is not None:
            knob = self.lsa[lsa]
        del self.mad[knob.mad]
        del self.lsa[knob.lsa]

    def add_knob(self, mad, lsa, scaling, test):
        self.add(LHCKnob(mad, lsa, scaling, test))

    def get_settings(self):
        return list(self.lsa.keys())


class LHCKnob:
    def __init__(self, mad, lsa, scaling, test):
        self.mad = mad
        self.lsa = lsa
        self.scaling = float(scaling)
        self.test = float(test)

    def mad_value(self, lsa_value):
        return lsa_value * self.scaling

    def lsa_value(self, mad_value):
        return mad_value / self.scaling

    def to_mad(self, lsa_value):
        return f"{self.mad}={self.mad_value(lsa_value)};"
