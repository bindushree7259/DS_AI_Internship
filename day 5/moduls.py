import math
import random
import datetime
import calendar

# Math module
number = 25
print("Square root:", math.sqrt(number))
print("Factorial:", math.factorial(5))
print("Value of PI:", math.pi)

# Random module
print("Random number:", random.randint(1, 100))
print("Random choice:", random.choice(["Apple", "Banana", "Mango"]))

# Datetime module
now = datetime.datetime.now()
print("Current date and time:", now)
print("Current date:", now.date())

# Calendar module
year = now.year
month = now.month
print("\nCurrent month's calendar:")
print(calendar.month(year, month))