# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA


#------Part 1----------
#----------------------
# Step 2
# Reading the data
dataset = pd.read_csv("winequalityN-lab6.csv")

# Step 3
def transform_quality(quality):
    if quality >= 6:
        return 1  # High-quality wine
    else:
        return 0  # Low-quality wine

dataset['quality'] = dataset['quality'].apply(transform_quality)

# Step 4
# Dropping the first column
dataset = dataset.drop(columns=['type'])

# Step 5
# Setting data and labels columns
labels = dataset.iloc[:, -1]
data = dataset.iloc[:, :-1]

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=0)

# Step 6
# Creating a pipeline with StandardScaler and LogisticRegression
pipeline = make_pipeline(StandardScaler(), LogisticRegression())

# Training the model
pipeline.fit(X_train, Y_train)

# Testing the model
Y_pred = pipeline.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Calculating confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculating F1 score
f1 = f1_score(Y_test, Y_pred)
print("F1 Score:", f1)

# Getting the probabilities for the positive class
Y_prob = pipeline.predict_proba(X_test)[:, 1]

# Calculating false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)

# Plotting ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Calculating AUC
auc = roc_auc_score(Y_test, Y_prob)
print("AUC:", auc)






#------Part 2----------
#----------------------
def transform_quality(quality):
    if quality >= 6:
        return 1  # High-quality wine
    else:
        return 0  # Low-quality wine

dataset['quality'] = dataset['quality'].apply(transform_quality)

#Step 4
#dropping the first column
dataset = dataset.drop(columns=['type'])

#Step 5
#Setting data and labels columns
labels = dataset.iloc[:, -1]
data = dataset.iloc[:, :-1]

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=0)

#Part 2 Step 1
#define a standard Scarlar to normalize inputs
scaler = StandardScaler()

#Step 2
#defining the classifer and the pipeline
l_reg = LogisticRegression(max_iter=10000)
pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=2))

#step 3
X_train_pca = pca_pipe.fit_transform(X_train)
X_test_pca = pca_pipe.transform(X_test)
print('x_train_pca is: ',X_train_pca)
print('x_test_pca is: ', X_test_pca)

#Step 4
clf = make_pipeline(StandardScaler(), l_reg)

#Step 5
clf.fit(X_train_pca, Y_train)

#Step 6
y_pred_pca = clf.predict(X_test_pca)

#Step 7
disp = DecisionBoundaryDisplay.from_estimator(
    clf, X_train_pca, response_method="predict",
        xlabel='X1', ylabel='X2',
        alpha=0.5,
)

#Step 8
disp.ax_.scatter(X_train_pca[:, 0], X_train_pca[:, 1] , c=Y_train)
plt.show()

#accuracy
acc = accuracy_score(Y_test, y_pred_pca)
print("Accuracy: ", acc)