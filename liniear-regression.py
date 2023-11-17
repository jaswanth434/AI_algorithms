import matplotlib.pyplot as plt
import random

def generate_random_points(num_points, x_range, y_range):
    return [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(num_points)]

def linear_regression_analysis(points):
    n = len(points)
    x_total = sum(x for x, _ in points)
    y_total = sum(y for _, y in points)

    x_avg = x_total / n
    y_avg = y_total / n

    sigma_x = sum((x - x_avg) ** 2 for x, _ in points) / (n - 1)
    covariance_xy = sum((x - x_avg) * (y - y_avg) for x, y in points) / (n - 1)

    slope = covariance_xy / sigma_x
    intercept = y_avg - slope * x_avg

    return slope, intercept

def predict(slope, intercept, x_value):
    return slope * x_value + intercept

def plot_graph(random_points, slope, intercept, x_value=None, predicted_y=None):
    x_values = [x for x, _ in random_points]
    y_values = [y for _, y in random_points]
    plt.scatter(x_values, y_values, color='blue', label='Data Points')

    if x_value is not None and predicted_y is not None:
        plt.scatter([x_value], [predicted_y], color='magenta', edgecolor='black', zorder=5, label='Predicted Point')

    regression_line = [predict(slope, intercept, x) for x in sorted(x_values)]
    plt.plot(sorted(x_values), regression_line, color='red', label='Regression Line')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression Analysis')
    plt.legend()
    plt.show()

num_points = int(input("Enter the number of points: "))
x_min = float(input("Enter the minimum x value: "))
x_max = float(input("Enter the maximum x value: "))
y_min = float(input("Enter the minimum y value: "))
y_max = float(input("Enter the maximum y value: "))

x_range = (x_min, x_max)
y_range = (y_min, y_max)

random_points = generate_random_points(num_points, x_range, y_range)

slope, intercept = linear_regression_analysis(random_points)
print("Slope:", slope)
print("Intercept:", intercept)

while True:
    try:
        x_value = float(input(f"Enter an x value between {x_min} and {x_max} for prediction (or type 'exit' to quit): "))
        if x_min <= x_value <= x_max:
            predicted_y = predict(slope, intercept, x_value)
            print(f"Predicted y for x = {x_value}: {predicted_y}")
            plot_graph(random_points, slope, intercept, x_value, predicted_y)
        else:
            print(f"Please enter a value within the range {x_min} to {x_max}.")
    except ValueError as e:
        if str(e) == "could not convert string to float: 'exit'":
            print("Exiting the program.")
            break
        else:
            print("Invalid input. Please enter a valid number or 'exit' to quit.")
