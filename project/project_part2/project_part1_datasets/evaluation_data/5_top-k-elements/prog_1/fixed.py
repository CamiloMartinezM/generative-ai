def top_k(lst, k):
    result = []
    while k > 0: # Fix condition, k >= 0 -> k > 0 to get k elements
        big = max(lst)
        result.append(big)
        lst.remove(big)
        k -= 1
    return result
