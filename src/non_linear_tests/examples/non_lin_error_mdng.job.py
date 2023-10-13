#%%
from pymadng import MAD
import tfs

import matplotlib.pyplot as plt

import time
import os
import numpy as np

orginal_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

with MAD() as mad:
    mad.load("MAD.utility", "assertf")

    mad.MADX.load("'lhc_as-built.seq'", "'lhc_as-built.mad'")
    mad.MADX.load("'opticsfile.21'", "'opticsfile.21.mad'")
    mad.MADX.load("'lhc_unset_vars.mad'") # Load a list of unset variables to prevent warnings

    mad.load("MADX", "lhcb1", "nrj")

    mad.assertf("#lhcb1 == 6694",
        "'invalid number of elements %d in LHCB1 (6694 expected)'", "#lhcb1")
    
    mad.lhcb1.beam = mad.beam(particle="'proton'", energy=mad.nrj)
    mad.MADX_env_send("""
    !ktqx1_r2 = -ktqx1_l2 ! remove the link between these 2 vars
    !kqsx3_l2 = -0.0015
    !kqsx3_r2 = +0.0015
    !MADX['KQSX3.R1_old'] = MADX['KQSX3.R1']
    !MADX['KQSX3.L1_old'] = MADX['KQSX3.L1']
    ! Power MQSX
    !MADX['KQSX3.R2'] =  1e-3  ! was KQSX3.R2 =  10E-4;
    !MADX['KQSX3.L2'] = -1e-3  ! was KQSX3.L2 = -10E-4;

    ktsx1_r2 = -ktsx1_l2
    kqsx3_r2 =  -1e-3
    kqsx3_l2 =  1e-3
    """)
    t0 = time.time()
    mad["tbl", "flw"] = mad.twiss(sequence=mad.lhcb1, method=4, coupling=True)
    print(mad.tbl)
    # plt.plot(mad.tbl.s, mad.tbl.beta11)
    # plt.show()
    mad.tbl.write("'before_tune_correction_n'")

    print("Values before matching")
    print("dQx.b1=", mad.MADX.dqx_b1)
    print("dQy.b1=", mad.MADX.dqy_b1)

    mad.send("""
    expr1 = \\t, s -> t.q1 - 62.30980
    expr2 = \\t, s -> t.q2 - 60.32154
    function twiss_and_send()
        local mtbl, mflow = twiss {sequence=lhcb1, method=4}
        py:send({mtbl.s, mtbl.beta11})
        return mtbl, mflow
    end
    """)
    match_rtrn = mad.match(
        command=mad.twiss_and_send,
        variables = [
            {"var":"'MADX.dqx_b1'", "name":"'dQx.b1'", "'rtol'":1e-6},
            {"var":"'MADX.dqy_b1'", "name":"'dQy.b1'", "'rtol'":1e-6},
        ],
        equalities = [
            {"expr": mad.expr1, "name": "'q1'", "tol":1e-3},
            {"expr": mad.expr2, "name": "'q2'", "tol":1e-3},
        ],
        objective={"fmin": 1e-3}, maxcall=100, info=2,
    )
    mad.send("py:send(nil)")
    tws_result = mad.recv ()

    #object_methods = [tws_result for tws_result in dir(object) if callable(getattr(object, tws_result))]
    #print(len(tws_result))
    #print([result for result in tws_result])
    
    x = tws_result[0]
    y = tws_result[1]

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'b-')
    while tws_result:
        line1.set_xdata(tws_result[0])
        line1.set_ydata(tws_result[1])
        fig.canvas.draw()
        fig.canvas.flush_events()
        tws_result = mad.recv()

    mad["status", "fmin", "ncall"] = match_rtrn
    del match_rtrn

    print("Values after matching")
    print("dQx.b1=", mad.MADX.dqx_b1)
    print("dQy.b1=", mad.MADX.dqy_b1)

    mad.twiss("tbl", sequence=mad.lhcb1, method=4, chrom=True)
    mad.tbl.write("'after_tune_correction_n'")
    t1 = time.time()
    print("pre-tracking time: " + str(t1 - t0) + "s")



os.chdir(orginal_dir)
 

# %%    

import tfs
import matplotlib.pyplot as plt

tw_quad = tfs.read_tfs("before_tune_correction_n_quad.tfs")
tw_sex = tfs.read_tfs("before_tune_correction_n_sex.tfs")

plt.plot(tw_quad.s, tw_quad.beta11, label="QUAD", alpha=0.7)
plt.plot(tw_sex.s, tw_sex.beta11, label="SEX", alpha=0.7)
plt.legend()
plt.show()
plt.plot(tw_quad.s, tw_quad.beta12, label="QUAD", alpha=0.7)
plt.plot(tw_sex.s, tw_sex.beta12, label="SEX", alpha=0.7)
plt.legend()
plt.show()
plt.plot(tw_quad.s, tw_quad.beta21, label="QUAD", alpha=0.7)
plt.plot(tw_sex.s, tw_sex.beta21, label="SEX", alpha=0.7)
plt.legend()
plt.show()
plt.plot(tw_quad.s, tw_quad.beta22, label="QUAD", alpha=0.7)
plt.plot(tw_sex.s, tw_sex.beta22, label="SEX", alpha=0.7)
plt.legend()
plt.show()

# %%
