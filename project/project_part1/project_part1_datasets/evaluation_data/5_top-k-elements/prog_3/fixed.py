def top_k(lst, k):
    new_list = []  # Renamed variable to new_list to avoid confusion with the built-in list type
    while len(new_list) < k:  # Changed the condition to check the length of new_list instead of lst, earlier it was checking the length of lst which was a typo for list
        a = max(lst)
        lst.remove(a)
        new_list.append(a)  # Changed from 'new' to 'new_list' to match the variable name
    return new_list  # Return the correct variable new_list