# Take numbers from the user
x = []

n = int(input("Enter the number of elements: "))

for i in range(n):
    num = int(input(f"Enter element {i+1}: "))
    x.append(num)

print("\nList:", x)
print("Minimum:", min(x))
print("Maximum:", max(x))
print("Sum:", sum(x))
print("Average:", sum(x) / len(x))
print("Total Length:", len(x))
print("Sorted List:", sorted(x))