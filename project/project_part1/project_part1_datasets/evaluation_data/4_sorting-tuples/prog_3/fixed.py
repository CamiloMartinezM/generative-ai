def sort_age(lst):
    new = []
    while lst:
        largest = lst[0]
        for ele in lst:
            if ele[1] > largest[1]:  # Compare the age part of the tuple
                largest = ele
        lst.remove(largest)  # Remove from lst, not a
        new.append(largest)
    return new  # Return statement should be outside the while loop