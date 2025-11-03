import pandas as pd
import matplotlib.pyplot as plt

# PART 1: Load and Preview the Dataset
iris_df = pd.read_csv('iris.csv')

# Print a sample of 10 rows
print("Sample of 10 rows:")
print(iris_df.sample(n=10, random_state=42))

# Use info function to examine feature names  
print("\nDataset Info:")
iris_df.info()

# Drop Virginica samples
iris_binary = iris_df[iris_df['iris_type'] != 'Iris-virginica']

# PART 2: Explore Data Visually

# Check class imbalance
print("\nClass distribution:")
print(iris_binary['iris_type'].value_counts())

# Create figure with 2 scatter plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot 1: Sepal Length vs Sepal Width
for iris_type in iris_binary['iris_type'].unique():
    subset = iris_binary[iris_binary['iris_type'] == iris_type]
    ax1.scatter(subset['sepal_length'], subset['sepal_width'], label=iris_type)
ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Sepal Width')
ax1.set_title('Sepal Length vs Sepal Width')
ax1.legend()

# Scatter plot 2: Petal Length vs Petal Width  
for iris_type in iris_binary['iris_type'].unique():
    subset = iris_binary[iris_binary['iris_type'] == iris_type]
    ax2.scatter(subset['petal_length'], subset['petal_width'], label=iris_type)
ax2.set_xlabel('Petal Length')
ax2.set_ylabel('Petal Width')
ax2.set_title('Petal Length vs Petal Width')
ax2.legend()

plt.tight_layout()
plt.show()

"""
ANALYSIS QUESTIONS:

Q1: How balanced are the two classes?
A: Perfectly balanced as there are 50 samples each.

Q2: Which feature pair shows the best separation?
A: Petal Length versus Petal Width shows the best separation between the pair. 

Q3: Does the data appear linearly separable?
A: Yes, using petal features makes this easier to notice.

Q4: What challenges might a simple model (like a perceptron) face?
A: Very few, and it is ideal for a perceptron based on linear separability
"""