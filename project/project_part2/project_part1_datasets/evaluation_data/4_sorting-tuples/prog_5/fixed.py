def sort_age(lst):
    final=[]
    while lst:
        old=lst[0]
        for i in lst:
            if old[1]<i[1]:
                old=i
        final.append(old) # Fix indentation, append and remove should be outside the for loop
        lst.remove(old) # Fix indentation, append and remove should be outside the for loop
    return final
