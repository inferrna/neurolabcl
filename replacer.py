import re
s = open("neurolab/tool.py", "r").read()

s = re.sub('^([ \t\r\f\v]*)(.+?)\.shape = (.*?), (.*?)$', '\g<1>\g<2> = \g<2>.reshape((\g<3>, \g<4>,)) #replacement for \"\\g<0>\"', s, flags=re.MULTILINE)
s = re.sub('^([ \t\r\f\v]*)(.+?)\.shape = (\S+?)$', '\g<1>\g<2> = \g<2>.reshape(\g<3>) #replacement for \"\\g<0>\"', s, flags=re.MULTILINE)

f = open("neurolab/newtool.py", "w")
f.write(s)
f.close()
