def count_dates(date, possible_birthdays):
    count = 0
    for i in possible_birthdays:
        if i[1] == date:
            count += 1
    return count
    
def unique_day(date, possible_birthdays):
    if count_dates(date, possible_birthdays) == 1:
        return True
    else:
        return False
    
def count_months(month, possible_birthdays):
    count = 0
    for i in possible_birthdays:
        if i[0] == month:
            count += 1
    return count
    
def unique_month(month, possible_birthdays):
    if count_months(month, possible_birthdays) == 1:
        return True
    else:
        return False


def contains_unique_day(month, possible_birthdays):
    days_in_month = ()
    for i in possible_birthdays:
        if i[0] == month:
            days_in_month += (i,)
    for x in range(len(days_in_month)):
        if unique_day(days_in_month[x][1], possible_birthdays):
            return True
        
    return False # return only after checking for all, not prematurely

