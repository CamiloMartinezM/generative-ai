def unique_day(day, possible_birthdays):
    result = ()
    for p in possible_birthdays:
        pd = p[1]
        if day == pd:
            result = result + (day,)
    if len(result) > 1 or len(result) == 0: # also return false if the counter is 0. Unique day should only return true if the day occurs exactly once
        return False
    return True

def unique_month(month, possible_birthdays): # complete the implementation of the function
    count = 0
    for birthday in possible_birthdays:
        if birthday[0] == month:
            count += 1
    return count == 1

def contains_unique_day(month, possible_birthdays): # complete the implementation of the function
    for birthday in possible_birthdays:
        if birthday[0] == month:
            if unique_day(birthday[1], possible_birthdays):
                return True
    return False