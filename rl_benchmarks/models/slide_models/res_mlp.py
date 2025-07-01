
import torch.nn as nn
import torch.nn.functional as F

######### Define the transformation z'=h(z) #############
class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden_dim=256, metadata_cols=3):
        """
        Initialize the ResidualMLP class.

        Parameters:
        - dim: int, input and output dimension of the MLP (input: batch * dim).
        - hidden_dim: int, hidden layer dimension.
        """
        super(ResidualMLP, self).__init__()
        self.metadata_cols = metadata_cols

        # Define MLP layers
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        """
        Forward pass with residual connection.

        Parameters:
        - x: Tensor, shape (batch, dim) # eg 16,1000,771
        - mask: Optional[Tensor], shape (batch, n_tiles, 1) containing booleans. True if padded

        Returns:
        - out: Tensor, shape (batch, dim), same shape as input
        """
        # Apply MLP layers with ReLU activation
        #print("res mlp", x.dtype, x.shape)
        x = x[..., self.metadata_cols:]
        residual = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)

        # Add residual connection
        out = out + residual
        return out
