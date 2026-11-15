print("*" * 80)
print(" welcome to recommendation assistant ")
print("*" * 80)

print("\nselect your interest")
print("1. movie")
print("2. book")
print("3. music")

choice = input("\nenter your choice (1/2/3): ")

if choice == "1":
    print("\nchoose movie genre: ")
    print("a. action")
    print("b. comedy")
    print("c. drama")
    genre = input("enter genre (a/b/c): ").strip().lower()
    
    if genre == "a":
        print("\nrecommended movie : RRR")
    elif genre == "b":
        print("\nrecommended movie : Hera Pheri")
    elif genre == "c":
        print("\nrecommended movie : Tare Zameen Par")
    else:
        print("\ninvalid choice")

elif choice == "2":
    print("\nchoose book category: ")
    print("a. mythology")
    print("b. fiction")
    print("c. motivation")
    category = input("\nenter category (a/b/c): ").strip().lower()
    
    if category == "a":
        print("recommended book : The Palace of Illusions")
    elif category == "b":
        print("recommended book : The Guide")
    elif category == "c":
        print("recommended book : Wings of Fire")
    else:
        print("invalid category")

elif choice == "3":
    print("\nchoose music type:")
    print("a. motivational")
    print("b. devotional")
    print("c. party")
    music = input("enter type (a/b/c): ").strip().lower()
    
    if music == "a":
        print("recommended song: Chak De India")
    elif music == "b":
        print("recommended song : Achyutam Keshavam")
    elif music == "c":
        print("recommended song : Kala Chashma")
    else:
        print("invalid choice")

else:
    print("invalid choice")

print("\nthank you for using recommendation assistant")