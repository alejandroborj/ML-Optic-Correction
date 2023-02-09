from lhcmodels import LHCRun, LHCCycle, LHCBeamProcess, LHCKnobs
from lhcmodels import lsa



knobs = LHCKnobs.from_file("../operation/knobs.txt")
knobs.add_knob("nrj", "LHCBEAM/MOMENTUM", 1.0, 1)


lhcrun = LHCRun(2022)
lhcrun.set_fills()

lhcrun.hist_beam_processes("PHYSICS")


pp_lumi = LHCCycle("pp_lumi",
    [
        "RAMP-SQUEEZE-6.8TeV-ATS-1.3m_V1",
        "SQUEEZE-6.8TeV-1.3m-60cm_V1",
        "SQUEEZE-6.8TeV-60cm-30cm_V1",
    ]
)
pp_lumi.get_fills(lhcrun)

pp_vdm = LHCCycle("pp_vdm",
    [
        'RAMP-DESQUEEZE-6.8TeV-19_2m-VdM-2022_V1',
        'QCHANGE-6.8TeV-2022-VdM_V1',
        'PHYSICS-6.8TeV-2022-VdM_V1'
    ]
)


presettings = {
    "RAMP-SQUEEZE-6.8TeV-ATS-1.3m_V1": {
        "sequence": "lhc.seq",
        "dqx.b1": 0.05,
        "dqy.b1": 0.05,
        "dqx.b2": 0.05,
        "dqy.b2": 0.05,
        "dqpx.b1": 13,
        "dqpy.b1": 13,
        "dqpx.b2": 13,
        "dqpy.b2": 13,
    }
}

settings = [
    "LHCBEAM/MOMENTUM",
    "LHCBEAM/IP1-XING-V-MURAD",
    "LHCBEAM/IP2-XING-V-MURAD",
    "LHCBEAM/IP5-XING-H-MURAD",
    "LHCBEAM/IP8-XING-H-MURAD",
    "LHCBEAM/IP1-SEP-H-MM",
    "LHCBEAM/IP2-SEP-H-MM",
    "LHCBEAM/IP5-SEP-V-MM",
    "LHCBEAM/IP8-SEP-V-MM",
    "LHCBEAM/IP2-ANGLE-H-MURAD",
    "LHCBEAM/IP8-ANGLE-V-MURAD",
    "RFBEAM1/TOTAL_VOLTAGE",
    "RFBEAM2/TOTAL_VOLTAGE",
]


# for bp in bp_set:
bp = LHCBeamProcess("RAMP_PELP-SQUEEZE-6.5TeV-ATS-1m-2018_V3_V1")
out = bp.get_model_history(settings, lhcrun, presettings.get(bp.name, {}))


optable = lsa.getOpticTable(bp.name)

fill = lhcrun.fills[7334]

fill = lhcrun.fills[7304]


for ts, pp in fill.beam_processes:
    print((ts - fill.get_start()) / 3600, pp)
