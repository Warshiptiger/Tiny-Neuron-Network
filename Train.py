import random
import math

def train(model, data, epochs=100):
    for epoch in range(epochs):
        random.shuffle(data)

        total_loss = 0
        correct = 0
        for item in data:
            x = item[0]
            y = item[1]
            y_hat = model.forward(x)

            model.backward(x, y)
            loss_part1 = y * math.log(y_hat + 1e-9)
            loss_part2 = (1 - y) * math.log(1 - y_hat + 1e-9)
            loss = -(loss_part1 + loss_part2)

            total_loss = total_loss + loss
            if y_hat >= 0.5:
                pred = 1
            else:
                pred = 0

            if pred == y:
                correct = correct + 1
        accuracy = correct / len(data)
        print("Epoch:", epoch + 1)
        print("Loss:", round(total_loss, 4))
        print("Accuracy:", round(accuracy * 100, 2), "%")
        print(" ")

def test(model, data):
    correct = 0
    for item in data:
        x = item[0]
        y = item[1]
        y_hat = model.forward(x)
        if y_hat >= 0.5:
            pred = 1
        else:
            pred = 0
        if pred == y:
            correct = correct + 1

    accuracy = correct / len(data)
    print("Accuracy:", round(accuracy * 100, 2), "%")
