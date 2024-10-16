import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder






# Question 1
dataset_name = "heart.csv"
dataset = pd.read_csv(dataset_name)

data = dataset.iloc[:, :-1]
labels = dataset.iloc[:,-1]

fig, ax = plt.subplots(ncols=4,nrows=4, figsize=(20,10))


for i in range(13):
    data.hist(ax=ax.flatten()[i])


fig.tight_layout()
plt.show()













# Question 3
dataset_name = "heart.csv"
dataset = pd.read_csv(dataset_name)

data = dataset.iloc[:, :-1]
labels = dataset.iloc[:,-1]

fig, ax = plt.subplots(ncols=4,nrows=4, figsize=(20,10))


for i in range(13):
    ax.flatten()[i].hist(data.iloc[:, i])
    ax.flatten()[i].set_title(data.columns[i], fontsize=15)


fig.tight_layout()
plt.show()















# Question 4

dataset = pd.read_csv("heart.csv")
data = dataset.iloc[:, :-1]
labels = dataset.iloc[:, -1]
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20,10))


data.plot(ax=ax.flatten()[0:13], kind='density', subplots=True, sharex=False)

fig.tight_layout()
plt.show()















# Question 5

dataset = pd.read_csv("heart.csv")
data = dataset.iloc[0:13, :-1]
labels = dataset.iloc[0:13, -1]
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20,10))


data.plot(ax=ax.flatten()[0:13], kind='box', subplots=True, sharex=False, sharey=False)

fig.tight_layout()
plt.show()














# Question 6

dataset = pd.read_csv("heart.csv")
data = dataset.iloc[:, :-1]
labels = dataset.iloc[:, -1]
fig, ax = plt.subplots(ncols=13, nrows=13, figsize=(30,30))

pd.plotting.scatter_matrix(data, ax=ax)

fig.tight_layout()
plt.show()















# Question 8
# Load the dataset
dataset_name = 'winequalityN.csv'
dataset = pd.read_csv(dataset_name)

# Drop the first column containing string values
dataset = dataset.drop(dataset.columns[0], axis=1)
label_encoder = LabelEncoder()
dataset['quality'] = label_encoder.fit_transform(dataset['quality'] >= 8)

# Separate the data from the lables
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Transform data with 2 components for PAC and T-SNE
scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca')
x_tsne = tsne.fit_transform(x)


# Plot t-SNE and PAC in 2 graphs in the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 10))
colors = ['pink', 'red']
legend_labels = ['Low-Quality', 'High-Quality']

ax_tsne = axes[0]
for i in range(len(legend_labels)):
    ax_tsne.scatter(x_tsne[y == i, 0], x_tsne[y == i, 1], c=colors[i], s=60)
ax_tsne.set_title('t-SNE of Wine Quality Dataset', fontsize=15)
ax_tsne.legend(legend_labels, fontsize=12)
ax_tsne.set_xlabel('t-SNE Component 1', fontsize=12)
ax_tsne.set_ylabel('t-SNE Component 2', fontsize=12)

ax_pca = axes[1]
for i in range(len(legend_labels)):
    ax_pca.scatter(x_pca[y == i, 0], x_pca[y == i, 1], c=colors[i], s=60)
ax_pca.set_title('PCA of Wine Quality Dataset', fontsize=15)
ax_pca.legend(legend_labels, fontsize=12)
ax_pca.set_xlabel('PCA Component 1', fontsize=12)
ax_pca.set_ylabel('PCA Component 2', fontsize=12)

plt.tight_layout()
plt.show()













# Question 9

# Load the dataset
dataset_name = 'winequalityN.csv'
dataset = pd.read_csv(dataset_name)

# Drop the first column containing string values
dataset = dataset.drop(dataset.columns[0], axis=1)
label_encoder = LabelEncoder()
dataset['quality'] = label_encoder.fit_transform(dataset['quality'] >= 8)

# Separate the data from the lables
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Transform data with 11 components for PAC and T-SNE
scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=11)
x_pca = pca.fit_transform(x)

tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca')
x_tsne = tsne.fit_transform(x)


# Plot t-SNE and PAC in 2 graphs in the figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))
colors = ['pink', 'red']
legend_labels = ['Low-Quality', 'High-Quality']

ax_tsne = axes[0]
for i in range(len(legend_labels)):
    ax_tsne.scatter(x_tsne[y == i, 0], x_tsne[y == i, 1], c=colors[i], s=60)
ax_tsne.set_title('t-SNE of Wine Quality Dataset', fontsize=15)
ax_tsne.legend(legend_labels, fontsize=12)
ax_tsne.set_xlabel('t-SNE Component 8', fontsize=12)
ax_tsne.set_ylabel('t-SNE Component 9', fontsize=12)

ax_pca = axes[1]
for i in range(len(legend_labels)):
    ax_pca.scatter(x_pca[y == i, 7], x_pca[y == i, 8], c=colors[i], s=60)
ax_pca.set_title('PCA of Wine Quality Dataset', fontsize=15)
ax_pca.legend(legend_labels, fontsize=12)
ax_pca.set_xlabel('Principal Component 8', fontsize=12)
ax_pca.set_ylabel('Principal Component 9', fontsize=12)

plt.tight_layout()
plt.show()