def search(x, seq):
    for i, elem in enumerate(seq):
        if x <= elem:
            return i
        
    return len(seq) # fix premature return, return the length of the sequence if x is greater than all elements in the sequence 