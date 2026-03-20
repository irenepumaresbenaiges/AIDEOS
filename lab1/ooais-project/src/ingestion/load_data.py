with open("data/raw/observations.csv") as f:
	lines = f.readlines()
print("Number of records:", len(lines)-1)
objects = [line.split(",")[1] for line in lines[1:]]
print("\nObjects:", set(objects))
object_counts = {}
for obj in objects:
    object_counts[obj] = object_counts.get(obj, 0) + 1
print("\nObject Occurrences:")
for obj, count in object_counts.items():
    print(f" - {obj}: {count}")
temperatures = []
for line in lines[1:]:
    val = line.split(",")[2]
    if val != "INVALID":
        temperatures.append(float(val))
if temperatures:
    avg_temp = sum(temperatures) / len(temperatures)
    print(f"\nAverage Temperature: {avg_temp:.2f}ºC")
else:
    print("\nAverage Temperature: N/A (No valid data)")