# Function definition
def table(n):
    for i in range(1, 11):
        print(n, "x", i, "=", n * i)

# Main function
def main():
    num = int(input("Enter a number: "))
    table(num)

# Function call
main()