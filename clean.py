# Remove recipes with placeholder images

infile = open('food_info.csv','r')
outfile = open('food_info_cleaned.csv','w+')

for i,line in enumerate(infile):
    if not line.startswith(r"./images/no-recipe-card-"):
        outfile.write(line)
infile.close()
outfile.close()
