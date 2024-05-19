import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
dataset_path = r'C:\Users\ahmed\Desktop\FACULTY - SEM 2 - 2024\Face Recog\Face-Recognition\DataSet'
data_matrix = []
labels = []

# Generate the DataMatrix and the Label vector
for subject_id in range(1, 41):
    subject_folder = os.path.join(dataset_path, f's{subject_id}')
    for image_name in os.listdir(subject_folder):
        if image_name.endswith('.pgm'):
            image_path = os.path.join(subject_folder, image_name)
            image = Image.open(image_path)
            image_array = np.array(image)
            image_vector = image_array.flatten()
            data_matrix.append(image_vector)
            labels.append(subject_id)

D = np.array(data_matrix)
y = np.array(labels)

# Checking the Data Matrix and the Label vector
print(f"Data matrix shape: {D.shape}")
print(f"Label vector shape: {y.shape}")

# Split the data into training and testing sets
D_train = D[::2]
D_test = D[1::2]
y_train = y[::2]
y_test = y[1::2]

print(f"Training data / Labels shape: {D_train.shape}/{y_train.shape}")
print(f"Test data / Labels shape: {D_test.shape}/{y_test.shape}")

# Standardize the data
scaler = StandardScaler()
D_train_scaled = scaler.fit_transform(D_train)
D_test_scaled = scaler.transform(D_test)

# PCA Implementation
def pca_manual(D, alpha):
    # Step 1: Compute the mean
    mu = np.mean(D, axis=0)

    # Step 2: Center the data
    Z = D - mu

    # Step 3: Compute the covariance matrix
    cov_matrix = np.cov(Z, rowvar=False)

    # Step 4: Compute eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

    # Step 5: Take only the real part of eigenvalues and eigenvectors
    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real

    # Step 6: Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]

    # Step 7: Compute the fraction of total variance
    cumulative_variance = np.cumsum(eig_vals) / np.sum(eig_vals)

    # Step 8: Choose the smallest r such that f(r) >= alpha
    num_components = np.searchsorted(cumulative_variance, alpha) + 1

    # Step 9: Reduced basis
    U_r = eig_vecs[:, :num_components]

    # Step 10: Project data onto the reduced dimensionality space
    A = np.dot(Z, U_r)

    return A, U_r, num_components

# LDA Implementation
def lda_manual(D_train_scaled, y_train):
    num_classes = len(np.unique(y_train))
    mean_vectors = [np.mean(D_train_scaled[y_train == cl], axis=0) for cl in range(1, num_classes + 1)]
    overall_mean = np.mean(D_train_scaled, axis=0)

    # Between-class scatter matrix Sb
    Sb = np.zeros((D_train_scaled.shape[1], D_train_scaled.shape[1]))
    for i, mean_vec in enumerate(mean_vectors, start=1):
        n = D_train_scaled[y_train == i, :].shape[0]
        mean_vec = mean_vec.reshape(D_train_scaled.shape[1], 1)
        overall_mean = overall_mean.reshape(D_train_scaled.shape[1], 1)
        Sb += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    # Within-class scatter matrix Sw
    Sw = np.zeros((D_train_scaled.shape[1], D_train_scaled.shape[1]))
    for cl, mean_vec in zip(range(1, num_classes + 1), mean_vectors):
        class_scatter = np.zeros((D_train_scaled.shape[1], D_train_scaled.shape[1]))
        for row in D_train_scaled[y_train == cl]:
            row = row.reshape(D_train_scaled.shape[1], 1)
            mean_vec = mean_vec.reshape(D_train_scaled.shape[1], 1)
            class_scatter += (row - mean_vec).dot((row - mean_vec).T)
        Sw += class_scatter

    # Solve the generalized eigenvalue problem for the matrix inv(Sw)Sb
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))

    # Take only the real part of the eigenvectors
    eig_vecs = eig_vecs.real

    # Sort eigenvalues and corresponding eigenvectors
    eig_pairs = [(eig_vals[i], eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # Select the top num_classes - 1 eigenvectors
    W = np.hstack([eig_pairs[i][1].reshape(D_train_scaled.shape[1], 1) for i in range(num_classes - 1)])

    # Project the training and test data onto the new LDA space
    D_train_lda = D_train_scaled.dot(W)
    D_test_lda = D_test_scaled.dot(W)

    return D_train_lda, D_test_lda

# Loop over different values of alpha
alphas = [0.8, 0.85, 0.9, 0.95]
k_values = [1, 3, 5, 7]
pca_accuracies = {k: [] for k in k_values}
lda_accuracies = {k: [] for k in k_values}

for alpha in alphas:
    # PCA
    D_train_pca, U_r, num_components = pca_manual(D_train_scaled, alpha)
    D_test_pca = np.dot(D_test_scaled - np.mean(D_train_scaled, axis=0), U_r)

    for k in k_values:
        # Nearest Neighbor Classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(D_train_pca, y_train)

        # Predict and calculate accuracy
        y_pred = knn.predict(D_test_pca)
        accuracy = np.mean(y_pred == y_test)
        pca_accuracies[k].append(accuracy)

        print(f"Alpha: {alpha}, k: {k}, Number of components: {num_components}, PCA Accuracy: {accuracy}")

# LDA
D_train_lda, D_test_lda = lda_manual(D_train_scaled, y_train)

for k in k_values:
    # Nearest Neighbor Classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(D_train_lda, y_train)

    # Predict and calculate accuracy
    y_pred = knn.predict(D_test_lda)
    accuracy = np.mean(y_pred == y_test)
    lda_accuracies[k].append(accuracy)

    print(f"k: {k}, LDA Accuracy: {accuracy}")

# Plot PCA accuracies
for k in k_values:
    plt.plot(alphas, pca_accuracies[k], marker='o', label=f'k={k}')
plt.xlabel('Alpha')
plt.ylabel('Classification Accuracy')
plt.title('PCA: Relation between Alpha and Classification Accuracy for different k values')
plt.legend()
plt.show()

# Plot LDA accuracies
plt.plot(k_values, [lda_accuracies[k][0] for k in k_values], marker='o')
plt.xlabel('k')
plt.ylabel('Classification Accuracy')
plt.title('LDA: Classification Accuracy for different k values')
plt.show()
