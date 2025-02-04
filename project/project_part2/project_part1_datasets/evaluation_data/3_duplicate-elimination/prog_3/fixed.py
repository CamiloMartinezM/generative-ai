def remove_extras(lst):
    # lst.sort() # Remove sorting, it is not required and changes the order of elements
    store = []
    for ele in lst:
        if ele not in store:
            store += [ele]
    return store