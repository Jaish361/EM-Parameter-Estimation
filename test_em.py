from data_generator import generate_dataset
from em_algorithm import run_em


# Generate data
X = generate_dataset(200)

# Run EM
result = run_em(
    X,
    max_iterations=20,
    tolerance=0.0001
)

# Display results
print("EM RESULTS")
print("=" * 30)

print("Iterations:", result["iterations"])

print("\nComponent 1")
print("Mean:", result["mean1"])
print("Variance:", result["variance1"])
print("Weight:", result["weight1"])

print("\nComponent 2")
print("Mean:", result["mean2"])
print("Variance:", result["variance2"])
print("Weight:", result["weight2"])