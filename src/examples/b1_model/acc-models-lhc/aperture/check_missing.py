from cpymad.madx import Madx
mad= Madx(stdout=False)
mad.call('../lhc.seq')

def find_missing_names(seq,apfile):
    apnames=set()
    for ll in open(apfile):
        if 'APERTURE' in ll:
          apnames.add(ll.split(',')[0].lower())
    seqnames=set([ee for ee in seq.element_names() if '$' not in ee])
    return seqnames-apnames,apnames-seqnames


nodef1,noelem1=find_missing_names(mad.sequence.lhcb1,"aperture.b1.madx")
nodef2,noelem2=find_missing_names(mad.sequence.lhcb2,"aperture.b2.madx")

print("Add aperture definition for")
print("\n".join(sorted(nodef1.union(nodef2))))
print("\n\n\n")

print("remove aperture definition for")
print("\n".join(sorted(noelem1.union(noelem2))))
print("\n\n\n")
