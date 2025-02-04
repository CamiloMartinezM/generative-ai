def top_k(lst, k):
    sort = []
    while len(sort) < k:
        largest = lst[0]
        for element in lst:
            if element > largest: # Fixed loop logic, earlier it was modifying the list while iterating
                largest = element
        lst.remove(largest)
        sort.append(largest)
    return sort