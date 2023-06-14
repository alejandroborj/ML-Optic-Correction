from cpymad.madx import Madx
import subprocess

mad=Madx(stdout=subprocess.DEVNULL)
mad.call("lhc.seq")

for el in mad.elements:
    mad.show(el)

ff='.l .r .lr a12 a23 a34 a45 a56 a67 a78 a81'.split()

for vv in sorted(mad.globals):
    if vv.startswith('k') or vv.startswith('a'):
      for fff in ff:
        if fff in vv:
            print(f"{vv}=0;")
            break

