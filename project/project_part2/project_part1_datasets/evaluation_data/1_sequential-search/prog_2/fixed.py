def search(x, seq):
    for i in range(len(seq)): # fixed the loop range
        if x <= seq[i]:   # removed unnecessary condition causing index error     
            return i

    return len(seq) # fix premature return, return the length of the sequence if x is greater than all elements in the sequence