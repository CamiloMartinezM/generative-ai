def sort_age(lst):
    lst.sort(key=lambda x: x[1], reverse=True) # sort() returns None, so we need to sort the list in place, and then return the list
    return lst

# Using the sorted() function instead of the sort() method is also a valid fix
