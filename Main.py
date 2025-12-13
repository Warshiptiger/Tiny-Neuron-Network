from Train import train, test
from network import TinyNN
from utils import load_csv

def main():
    model = TinyNN(lr=0.1)
    data = load_csv("data.csv")

    while True:
        print("1. Train")
        print("2. Test")
        print("3. Predict")
        print("4. Save")
        print("5. Load")
        print("6. Quit")

        choice = input("Choice: ")

        if choice == "1":
            train(model, data)
        elif choice == "2":
            test(model, data)
        elif choice == "3":
            x = float(input("x: "))
            y = float(input("y: "))
            print("Prediction:", model.forward([x, y]))
        elif choice == "4":
            model.save("weights.txt")
            print("Saved.")
        elif choice == "5":
            model.load("weights.txt")
            print("Loaded.")
        elif choice == "6":
            break

if __name__ == "__main__":
    main()
