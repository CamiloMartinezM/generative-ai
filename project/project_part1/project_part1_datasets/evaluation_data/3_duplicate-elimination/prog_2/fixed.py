def remove_extras(lst):
    new = []
    for x in lst:
        # Check if x is already in the new list, earlier code was removing duplicates completely, not just extras
        if x not in new:
            # Append x to the new list, earlier append was used incorrectly
            new.append(x)
    return new