import numpy as np

# Rows = days, Columns = products
sales = np.array([
    [100, 150, 200],   # Day 1
    [120, 180, 220],   # Day 2
    [130, 160, 250],   # Day 3
    [110, 170, 230]    # Day 4
])

print("Daily Product Sales:\n", sales)

# axis=0: product-wise results (column-wise)
print("\nProduct-wise Mean:", np.mean(sales, axis=0))
print("Product-wise Median:", np.median(sales, axis=0))
print("Product-wise Variance:", np.var(sales, axis=0))
print("Product-wise Standard Deviation:", np.std(sales, axis=0))

# axis=1: day-wise results (row-wise)
print("\nDay-wise Mean:", np.mean(sales, axis=1))
print("Day-wise Median:", np.median(sales, axis=1))
print("Day-wise Variance:", np.var(sales, axis=1))
print("Day-wise Standard Deviation:", np.std(sales, axis=1))