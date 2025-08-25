import csv

count_c = 0
count_d = 0
count_z = 0
count_n = 0

with open("labels.csv", newline='') as infile:
    reader = csv.reader(infile)
    header = next(reader)

    for row in reader:
        if row[1].strip().lower() == 'c':
            #print(row[0])
            count_c+=1
            
        if row[1].strip().lower() == 'd':
            #print(row[0])
            count_d+=1
        if row[1].strip().lower() == 'z':
            #print(row[0])
            count_z+=1
        if row[1].strip().lower() == 'n':
            #print(row[0])
            count_n+=1

print(f'c - Amount of car crashes: {count_c}')
print(f'd - Amount of do not include: {count_d}')
print(f'z - Amount of no cars: {count_z}')
print(f'n - Amount of normal videos: {count_n}')
print(f'Total amount of labeled videos: {count_c+count_d+count_z+count_n}')