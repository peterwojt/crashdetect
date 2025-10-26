import os
import re
from collections import Counter

# Directory containing the crash video files
folder = "crashes/2025-2024-2023"  # <-- change this to your folder path

# Regex patterns to extract intersection names
# Adjusted to catch patterns like "__148_SE_28__", "_108_NE_4-_-_Center_", etc.
patterns = [
    re.compile(r"__([A-Za-z0-9_]+)__"),         # pattern like "__148_SE_28__"
    re.compile(r"([0-9]+_[A-Z]{1,2}_[A-Za-z0-9_-]+)_\d{4}-\d{2}-\d{2}")  # pattern with date after
]

counter = Counter()

for filename in os.listdir(folder):
    if not filename.lower().endswith(".mp4"):
        continue
    
    intersection = None
    for pattern in patterns:
        match = pattern.search(filename)
        if match:
            intersection = match.group(1)
            break
    
    if not intersection:
        # fallback: try to extract something before date-like pattern
        parts = re.split(r"_\d{4}-\d{2}-\d{2}", filename)
        if len(parts) > 1:
            intersection = parts[0]
    
    if intersection:
        counter[intersection] += 1
    else:
        print(f"⚠️ Could not parse intersection from: {filename}")

# Print results
print("\n=== Crash Counts by Intersection ===")
for intersection, count in counter.most_common():
    print(f"{intersection}: {count}")

if counter:
    top = counter.most_common(1)[0]
    print(f"\n🚦 Most crashes: {top[0]} ({top[1]} crashes)")
else:
    print("No valid intersections found.")
