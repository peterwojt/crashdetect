import csv

with open('crashes_in_videos.csv', newline='') as f:
    reader = csv.reader(f)
    row_count = sum(1 for row in reader) - 1  # subtract 1 if there's a header
print(f"Total rows: {row_count}")
