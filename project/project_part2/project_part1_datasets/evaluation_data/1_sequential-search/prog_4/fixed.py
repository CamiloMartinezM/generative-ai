def search(x, seq):
    if seq == () or seq == []: # fixed the condition to check if the sequence is empty
        return 0
    else:
        for i, elem in enumerate (seq):
            if x<=elem:
                return i
            elif x>seq[-1]:
                return len(seq)