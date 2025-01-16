def merge(one, two):
    new_tup = []
    # The variables 'left' and 'right' were not defined in this scope, 
    # changed to 'one' and 'two' to match the function parameters.
    while one and two:
        # To sort in descending order by age, the comparison should be '>' instead of '<'.
        if one[0][1] > two[0][1]:
            new_tup.append(one.pop(0))
        else:
            new_tup.append(two.pop(0))
    # After the while loop, we need to extend the list with the remaining items.
    new_tup.extend(one or two)
    return new_tup

def sort_age(lst):
    n = len(lst)
    if n < 2:
        return lst
    # The division operator should be '//' for integer division, not '/'.
    left = lst[:n//2]
    right = lst[n//2:]
    return merge(sort_age(left), sort_age(right))