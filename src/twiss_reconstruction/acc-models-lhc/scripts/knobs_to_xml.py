from xml.etree import ElementTree

xml = ElementTree.parse('operation/lhc.jmd.xml')

doc = xml.getroot()                                                                                                    

for optic in doc.find('optics').iter('optic'): 
     print(optic.get('name')) 
     call = ElementTree.Element('call-file') 
     call.set('path','operation/optics/%s_lsaknobs.madx'%optic.get('name')) 
     optic.find('init-files').append(call) 

xml.write('operation/lhc.jmd.xml.new')
