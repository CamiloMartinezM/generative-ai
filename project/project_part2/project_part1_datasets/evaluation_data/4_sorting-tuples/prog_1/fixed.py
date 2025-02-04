def sort_age(lst):
    product = []
    while lst:
        largest = lst[0]  # We are looking for the largest (oldest) person, not the smallest.
        for i in lst:
            if i[1] > largest[1]:  # Change the comparison operator to '>' to find the largest.
                largest = i
        lst.remove(largest)
        product.append(largest)
    return product