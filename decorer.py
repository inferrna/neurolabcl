import re
s = open("neurolab/mynp.py", "r").read()

s = re.sub('^(def .*)$', '@chkfunc\n\g<1>', s, flags=re.MULTILINE)
s = re.sub('^([ \t\r\f\v]+)(def .*)$', '\g<1>@chkmethod\n\g<1>\g<2>', s, flags=re.MULTILINE)

f = open("neurolab/newmynp.py", "w")
f.write(s)
f.close()
