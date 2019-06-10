prev = ""
with open('food_info.csv','r') as f:
    for line in f:
        s = line
        if s != prev:
            with open('food_info_cleaned.csv','a') as f2:
                f2.write(s)
        prev = line
