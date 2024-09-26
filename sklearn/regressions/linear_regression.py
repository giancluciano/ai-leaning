from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)


# Use only one feature [bmi body mass index]
diabetes_X = diabetes_X[:, None, 2]
# terget diabetes_y is a quantitative measure of disease progression one year after baseline

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

reg = linear_model.LinearRegression()
reg.fit(diabetes_X_train, diabetes_y_train)
print(f"coefficients: {reg.coef_}") 

# Make predictions using the testing set
diabetes_y_pred = reg.predict(diabetes_X_test)


# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()