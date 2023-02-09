"""
Usage python generate_lumiknobs.py >toolkit/match-lumiknobs.madx
"""


from collections import defaultdict
import re


def parse_knob(fn):
    out = defaultdict(list)
    for l in open(fn):
        if "add2expr" in l and "expr=0.0*" not in l:
            _, vv, kk = l.split("=")
            vvv = re.match(r"([^,]+),", vv).groups()[0]
            kkk = re.search(r"on_[a-z0-9]+", kk).group()
            out[kkk].append(vvv)
    return out


def gen_match(kname, correctors):
    xy, ipn, beam = re.match("on_(.)ip(.)b(.)", kname).groups()
    seq = f"sequence=lhcb{beam}"
    ip = f"range=ip{ipn}"
    end = f"range=s.ds.r{ipn}.b{beam}"
    out = []
    for corr in correctors:
        out.append(f"add2expr,var={corr},expr={kname}_{corr}*{kname};")
    out.append(f"use, {seq},range=e.ds.l{ipn}.b{beam}/s.ds.r{ipn}.b{beam};")
    out.append(f"{kname}=1;")
    out.append(f"match, {seq}, betx=1, bety=1;")
    out.append(f"constraint, {seq}, {ip},  {xy}  = 0.001;")
    out.append(f"constraint, {seq}, {ip},  p{xy} = 0;")
    out.append(f"constraint, {seq}, {end}, {xy}  = 0;")
    out.append(f"constraint, {seq}, {end}, p{xy} = 0;")
    for corr in correctors:
        out.append(f"vary, name={kname}_{corr},step=1.e-12;")
    out.append(f"{kname}_{correctors[0]}=1e-10;")
    out.append(f"jacobian, calls = 10, tolerance=1.e-30,bisec=5;")
    out.append(f"endmatch;")
    out.append(f"{kname}=0;")
    out.append(f"tar_{kname}=tar;")
    return "\n".join(out)


#knobs = parse_knob("strengths/R2017a_A65C65A10mL300_lsaknobs.madx")

knobs = {
    "on_xip1b1": ["acbch5.r1b1", "acbyhs4.r1b1", "acbch6.l1b1", "acbyh4.l1b1"],
    "on_xip1b2": ["acbyh4.r1b2", "acbch6.r1b2", "acbyhs4.l1b2", "acbch5.l1b2"],
    "on_yip1b1": ["acbyvs4.l1b1", "acbcv5.l1b1", "acbyv4.r1b1", "acbcv6.r1b1"],
    "on_yip1b2": ["acbcv5.r1b2", "acbyv4.l1b2", "acbcv6.l1b2", "acbyvs4.r1b2"],
    "on_xip2b1": ["acbchs5.r2b1", "acbyhs4.l2b1", "acbyh5.l2b1", "acbyh4.r2b1"],
    "on_xip2b2": ["acbyh4.l2b2", "acbch5.r2b2", "acbyhs4.r2b2", "acbyhs5.l2b2"],
    "on_yip2b1": ["acbcv5.r2b1", "acbyv4.l2b1", "acbyvs5.l2b1", "acbyvs4.r2b1"],
    "on_yip2b2": ["acbcvs5.r2b2", "acbyv4.r2b2", "acbyv5.l2b2", "acbyvs4.l2b2"],
    "on_xip5b1": ["acbch6.l5b1", "acbyhs4.r5b1", "acbyh4.l5b1", "acbch5.r5b1"],
    "on_xip5b2": ["acbch6.r5b2", "acbyh4.r5b2", "acbch5.l5b2", "acbyhs4.l5b2"],
    "on_yip5b1": ["acbyvs4.l5b1", "acbcv6.r5b1", "acbyv4.r5b1", "acbcv5.l5b1"],
    "on_yip5b2": ["acbyvs4.r5b2", "acbcv5.r5b2", "acbcv6.l5b2", "acbyv4.l5b2"],
    "on_xip8b1": ["acbyhs4.l8b1", "acbch5.l8b1", "acbch6.r8b1", "acbyh4.r8b1"],
    "on_xip8b2": ["acbyh4.l8b2", "acbyhs4.r8b2", "acbyh5.r8b2", "acbchs5.l8b2"],
    "on_yip8b1": ["acbyv4.l8b1", "acbyvs4.r8b1", "acbcvs5.l8b1", "acbyv5.r8b1"],
    "on_yip8b2": ["acbyvs5.r8b2", "acbyvs4.l8b2", "acbyv4.r8b2", "acbcv5.l8b2"],
}

for kk,cc in knobs.items():
    print(f"! Generate knob for  {kk}")
    print(gen_match(kk,cc))
    print(f"! End generate knob for {kk}")

for kk in knobs:
    print(f"value,tar_{kk};")

print(f"tar_lumiknob={'+'.join(f'tar_{kk}' for kk in knobs)};")
print(f"value,tar_lumiknob;")
print('if(tar_lumiknob>1e-29){print,text="lumiknob failed";stop;};')



