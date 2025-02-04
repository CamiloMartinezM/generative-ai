def top_k(lst, k):
    new = []
    for _ in range(k): # fixed the choice of loop, original code had a wrong syntax
        max_value = max(lst)
        new.append(max_value)
        lst.remove(max_value) # fixed the removal of max value from the list, pop needs an index not a value
    return new