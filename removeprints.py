# git diff --no-prefix 3084125 eb4fcca  > /tmp/diff.patch
# egrep "\+\+\+|print" /tmp/diff.patch > /tmp/print.txt

p = open("/tmp/print.txt", "r").read().split("+++")
pp = [a.split("+") for a in p if "\n+" in a]
for x in pp:
    fnm = x[0].replace(" ", "").replace("\n", "")
    s = open(fnm, "r").read()
    for xx in x[1:]:
        s = s.replace(xx, "")
    if x[1:]:
        f = open(fnm, "w")
        f.write(s)
        f.close()
