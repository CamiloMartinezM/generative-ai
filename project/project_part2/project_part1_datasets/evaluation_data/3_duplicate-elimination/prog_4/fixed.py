def remove_extras(lst):
    i = 0 
    while i < len(lst): # Old approach was modifying the list while iterating over it
        if lst[i] in lst[:i]: # Original code had a typo by using "ls" instead of "lst"
            lst.pop(i)
        else:
            i += 1
    return lst
