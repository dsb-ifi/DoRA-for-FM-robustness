import torch.nn as nn
import torch.nn.functional as F

class Joint_entropy(nn.Module):
    def __init__(self, alpha=2.0, sigma=None, k=5):
        super(Joint_entropy, self).__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.k = k

    def sigma_estimation(self, X):
        """Estimate sigma using the median distance."""
        pairwise_distances = torch.cdist(X, X, p=2) ** 2
        distances = pairwise_distances.flatten().detach().cpu().numpy()
        med = np.median(distances[distances > 0])  # Exclude zero distances
        if med <= 0:
            med = np.mean(distances[distances > 0])
        if med < 1E-2:
            med = 1E-2
        return 0.1*med

    def normalized_distance_matrix_RBF(self, X, sigma=None):
        """Calculate the normalized distance matrix using the RBF kernel."""
        # If sigma is not provided, estimate it
        if sigma is None:
            sigma = self.sigma_estimation(X)

        # Normalize the input and calculate pairwise squared Euclidean distances
        X_normalized = F.normalize(X, p=2, dim=1)
        pairwise_distances = torch.cdist(X_normalized, X_normalized, p=2) ** 2
        # Apply the RBF kernel
        K = torch.exp(-pairwise_distances / (2 * sigma ** 2))
        return K

    def top_k_eigenvalues(self, matrix, k):
        # Use SVD to get the top k eigenvalues
        _, S, _ = torch.svd(matrix)
        return S

    def joint_entropy(self, X, Y, alpha, sigma=None, k=5):
        """Calculate the joint entropy between X and Y."""
        K_X = self.normalized_distance_matrix_RBF(X, sigma)
        K_X = K_X / torch.trace(K_X)
        K_Y = self.normalized_distance_matrix_RBF(Y, sigma)
        K_Y = K_Y / torch.trace(K_Y)
        # Compute the Hadamard product of the two kernels
        K = K_X * K_Y
        K = K / torch.trace(K)
        top_k_eigenvalues = self.top_k_eigenvalues(K, k)
        eigenvalues_pow = top_k_eigenvalues ** alpha
        joint_entropy = 1 / (1 - alpha) * torch.log2(torch.sum(eigenvalues_pow))
        return joint_entropy

    def forward(self, X, Y):
        joint_entropy_XY = self.joint_entropy(X, Y, self.alpha, self.sigma, self.k)
        # Return the negative value, so that the entropy will be maximized instead of minimized.
        return -joint_entropy_XY