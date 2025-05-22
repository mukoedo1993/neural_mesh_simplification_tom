import torch
import torch.nn as nn

from .layers import TriConv


class FaceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, k):
        super(FaceClassifier, self).__init__()
        self.k = k
        self.num_layers = num_layers

        self.triconv_layers = nn.ModuleList(
            [
                TriConv(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )

        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, pos, batch=None):
        # Handle empty input
        if x.size(0) == 0 or pos.size(0) == 0:
            return torch.tensor([], device=x.device)

        # If pos is 3D (num_faces, 3, 3), compute centroids
        if pos.dim() == 3:
            pos = pos.mean(dim=1)  # Average vertex positions to get face centers

        # Construct k-nn graph based on triangle centers
        edge_index = self.custom_knn_graph(pos, self.k, batch)

        # Apply TriConv layers
        for i in range(self.num_layers):
            x = self.triconv_layers[i](x, pos, edge_index)
            x = torch.relu(x)

        # Final classification
        x = self.final_layer(x)
        logits = x.squeeze(-1)  # Remove last dimension

        # Apply softmax normalization per batch
        if batch is None:
            # Global normalization using softmax
            probs = torch.softmax(logits, dim=0)
        else:
            # Per-batch normalization
            probs = torch.zeros_like(logits)
            for b in range(int(batch.max().item()) + 1):
                mask = batch == b
                probs[mask] = torch.softmax(logits[mask], dim=0)

        return probs

    def custom_knn_graph(self, x, k, batch=None):
        if x.size(0) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=x.device)

        batch_size = 1 if batch is None else int(batch.max().item()) + 1
        edge_index = []

        for b in range(batch_size):
            if batch is None:
                x_batch = x
            else:
                mask = batch == b
                x_batch = x[mask]

            if x_batch.size(0) > 1:
                distances = torch.cdist(x_batch, x_batch)
                distances.fill_diagonal_(float("inf"))
                _, indices = distances.topk(min(k, x_batch.size(0) - 1), largest=False)

                source = (
                    torch.arange(x_batch.size(0), device=x.device)
                    .view(-1, 1)
                    .expand(-1, indices.size(1))
                )
                edge_index.append(
                    torch.stack([source.reshape(-1), indices.reshape(-1)])
                )

        if edge_index:
            edge_index = torch.cat(edge_index, dim=1)

            # Make the graph symmetric
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

        return edge_index
