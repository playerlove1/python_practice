#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py
"""

# The pickle file has to be using Unix new lines otherwise at least Python 3.4's C pickle parser fails with exception: pickle.UnpicklingError: the STRING opcode argument must be quoted
# I think that some git versions may be changing the Unix new lines ('\n') to DOS lines ('\r\n').

#用來轉換原始的.pkl檔


#original = "word_data.pkl"
#destination = "word_data_unix.pkl"
original = "email_authors.pkl"
destination = "email_authors_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:    
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))