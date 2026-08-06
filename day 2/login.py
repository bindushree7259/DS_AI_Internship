print("Welcome to Fruit Shopping")

cart = []

# Add fruits until user enters done
print("\nEnter fruits (type 'done' to stop adding):")

while True:
    fruit = input("Enter fruit: ")

    if fruit.lower() == "done":
        break

    cart.append(fruit)
    print(fruit, "added successfully!")

# CRUD Operations
while True:
    print("\n1. View Cart")
    print("2. Update Fruit")
    print("3. Delete Fruit")
    print("4. Checkout")

    choice = input("Enter your choice: ")

    # READ
    if choice == "1":
        print("\nShopping Cart:", cart)

    # UPDATE
    elif choice == "2":
        old_fruit = input("Enter fruit to update: ")

        if old_fruit in cart:
            new_fruit = input("Enter new fruit: ")
            index = cart.index(old_fruit)
            cart[index] = new_fruit
            print("Fruit updated successfully!")
        else:
            print("Fruit not found!")

    # DELETE
    elif choice == "3":
        fruit = input("Enter fruit to delete: ")

        if fruit in cart:
            cart.remove(fruit)
            print("Fruit deleted successfully!")
        else:
            print("Fruit not found!")

    # CHECKOUT
    elif choice == "4":
        print("\nCart Type : List")
        print(cart)

        print("\nTotal Items :", len(cart))

        cart_tuple = tuple(cart)

        print("\nCart Type : Tuple")
        print(cart_tuple)

        print("\nCheckout Complete!")
        print("Thank you for shopping!")
        break

    else:
        print("Invalid choice!")