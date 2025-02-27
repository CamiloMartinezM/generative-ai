def unique_day(date, possible_birthdays):
    counter = 0
    for birthdate in possible_birthdays:
        if str(date) == birthdate[1]:
            counter += 1
    if counter > 1 or counter == 0: # also return false if the counter is 0. Unique day should only return true if the day occurs exactly once
        return False
    else:
        return True
            

def unique_month(month, possible_birthdays):
    counter = 0
    for birthdate in possible_birthdays:
        if month == birthdate[0]:
            counter += 1
    if counter > 1 or counter == 0: # also return false if the counter is 0. Unique month should only return true if the month occurs exactly once
        return False
    else:
        return True

def contains_unique_day(month, possible_birthdays):
    counter = 0
    for birthdate in possible_birthdays:
        if month == birthdate[0]:
            tp = unique_day(birthdate[1], possible_birthdays)
            if tp == True:
                counter += 1
    if counter >= 1:
        return True
    else:
        return False
    
