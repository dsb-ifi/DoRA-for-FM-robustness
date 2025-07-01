
import torch
import numpy as np

class MMD(torch.nn.Module):
    def __init__(self, kernel="rbf", degree=2, coef0=1, sigma_scale=0.5):
        """
        Initialize the MMD class.

        Parameters:
        - kernel: str, 'rbf' or 'poly'. Choose the kernel type for MMD computation.
        - degree: int, Degree of the polynomial kernel function. Ignored for 'rbf'.
        - coef0: int, Independent term in polynomial kernel. Ignored for 'rbf'.
        - sigma_scale: float, Scaling factor for sigma in RBF kernel using the median heuristic.
        """
        super(MMD, self).__init__()
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.sigma_scale = sigma_scale

    def sigma_estimation(self, X):
        """
        Estimate sigma using the median distance heuristic for RBF kernel.

        Parameters:
        - X: Tensor, input samples of shape (n_samples, n_features).

        Returns:
        - Estimated sigma value for kernel.
        """
        pairwise_distances = torch.cdist(X, X, p=2) ** 2  # Squared Euclidean distances
        distances = pairwise_distances.flatten().detach().cpu().numpy()
        med = np.median(
            distances[distances > 0]
        )  # Exclude zero distances (self-distances)
        if med <= 0:
            med = np.mean(distances[distances > 0])
        if med < 1e-2:  # Prevent sigma from being too small
            med = 1e-2
        return self.sigma_scale * med

    def rbf_kernel(self, X, Y, sigma):
        """
        Compute the RBF kernel between two sets of samples.

        Parameters:
        - X: Tensor, shape (n_samples_X, n_features)
        - Y: Tensor, shape (n_samples_Y, n_features)
        - sigma: float, kernel width.

        Returns:
        - Tensor, RBF kernel matrix.
        """
        pairwise_distances = torch.cdist(X, Y, p=2) ** 2  # Squared Euclidean distances
        return torch.exp(-pairwise_distances / (2 * sigma**2))

    def poly_kernel(self, X, Y):
        """
        Compute the Polynomial kernel between two sets of samples.

        Parameters:
        - X: Tensor, shape (n_samples_X, n_features)
        - Y: Tensor, shape (n_samples_Y, n_features)

        Returns:
        - Tensor, Polynomial kernel matrix.
        """
        return (torch.mm(X, Y.T) * self.coef0 + 1) ** self.degree

    def forward(self, X, Y):
        """
        Compute Maximum Mean Discrepancy (MMD) between two samples X and Y.

        Parameters:
        - X: Tensor, shape (n_samples_X, n_features), input samples from distribution P.
        - Y: Tensor, shape (n_samples_Y, n_features), input samples from distribution Q.

        Returns:
        - mmd_value: float, the estimated MMD value between X and Y.
        """
        # Estimate sigma using the median heuristic for RBF kernel
        if self.kernel == "rbf":
            sigma = self.sigma_estimation(torch.cat([X, Y], dim=0))
            XX = self.rbf_kernel(X, X, sigma).mean()
            YY = self.rbf_kernel(Y, Y, sigma).mean()
            XY = self.rbf_kernel(X, Y, sigma).mean()
        elif self.kernel == "poly":
            XX = self.poly_kernel(X, X).mean()
            YY = self.poly_kernel(Y, Y).mean()
            XY = self.poly_kernel(X, Y).mean()
        else:
            raise ValueError("Unknown kernel type. Use 'rbf' or 'poly'.")

        # MMD computation
        mmd_value = XX + YY - 2 * XY
        return mmd_value


"""
Example initialisation w standard parameters:
mmd_calculator_rbf = MMD(kernel='rbf', sigma_scale=0.5)
print("MMD (RBF kernel, sigma_scale=0.5):", mmd_calculator_rbf(a, b).item())
"""
