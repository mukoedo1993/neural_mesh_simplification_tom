import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

from neural_mesh_simplification.models.edge_predictor import EdgePredictor
from neural_mesh_simplification.models.layers.devconv import DevConv


@pytest.fixture
def sample_mesh_data():
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]], dtype=torch.long
    )
    return Data(x=x, edge_index=edge_index)


def test_edge_predictor_initialization():
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=15)
    assert isinstance(edge_predictor.devconv, DevConv)
    assert isinstance(edge_predictor.W_q, nn.Linear)
    assert isinstance(edge_predictor.W_k, nn.Linear)
    assert edge_predictor.k == 15


def test_edge_predictor_forward(sample_mesh_data):
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=2)
    simplified_adj_indices, simplified_adj_values = edge_predictor(
        sample_mesh_data.x, sample_mesh_data.edge_index
    )

    assert isinstance(simplified_adj_indices, torch.Tensor)
    assert isinstance(simplified_adj_values, torch.Tensor)
    assert simplified_adj_indices.shape[0] == 2  # 2 rows for source and target indices
    assert (
        simplified_adj_values.shape[0] == simplified_adj_indices.shape[1]
    )  # Same number of values as edges


def test_edge_predictor_output_range(sample_mesh_data):
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=2)
    _, simplified_adj_values = edge_predictor(
        sample_mesh_data.x, sample_mesh_data.edge_index
    )

    assert (simplified_adj_values >= 0).all()  # Values should be non-negative


def test_edge_predictor_symmetry(sample_mesh_data):
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=2)
    simplified_adj_indices, simplified_adj_values = edge_predictor(
        sample_mesh_data.x, sample_mesh_data.edge_index
    )

    # Create a sparse tensor from the output
    n = sample_mesh_data.x.shape[0]
    adj_matrix = torch.sparse_coo_tensor(
        simplified_adj_indices, simplified_adj_values, (n, n)
    )
    dense_adj = adj_matrix.to_dense()

    assert torch.allclose(dense_adj, dense_adj.t(), atol=1e-6)


def test_edge_predictor_connectivity(sample_mesh_data):
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=2)
    simplified_adj_indices, _ = edge_predictor(
        sample_mesh_data.x, sample_mesh_data.edge_index
    )

    # Check if all nodes are connected
    unique_nodes = torch.unique(simplified_adj_indices)
    assert len(unique_nodes) == sample_mesh_data.x.shape[0]


def test_edge_predictor_different_input_sizes():
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=5)

    # Test with a larger graph
    x = torch.rand(10, 3)
    edge_index = torch.randint(0, 10, (2, 30))
    simplified_adj_indices, simplified_adj_values = edge_predictor(x, edge_index)

    assert simplified_adj_indices.shape[0] == 2
    assert simplified_adj_values.shape[0] == simplified_adj_indices.shape[1]
    assert torch.max(simplified_adj_indices) < 10


def test_attention_scores_shape(sample_mesh_data):
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=2)

    # Get intermediate features
    knn_edges = knn_graph(sample_mesh_data.x, k=2, flow="target_to_source")
    extended_edges = torch.cat([sample_mesh_data.edge_index, knn_edges], dim=1)
    features = edge_predictor.devconv(sample_mesh_data.x, extended_edges)

    # Test attention scores
    attention_scores = edge_predictor.compute_attention_scores(
        features, sample_mesh_data.edge_index
    )

    assert attention_scores.shape[0] == sample_mesh_data.edge_index.shape[1]
    assert torch.allclose(
        attention_scores.sum(),
        torch.tensor(
            len(torch.unique(sample_mesh_data.edge_index[0])), dtype=torch.float32
        ),
    )


def test_simplified_adjacency_shapes():
    # Create a simple graph
    x = torch.rand(5, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    attention_scores = torch.rand(edge_index.shape[1])

    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=15)
    indices, values = edge_predictor.compute_simplified_adjacency(
        attention_scores, edge_index
    )

    assert indices.shape[0] == 2
    assert indices.shape[1] == values.shape[0]
    assert torch.max(indices) < 5


def test_empty_input_handling():
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=15)
    x = torch.rand(5, 3)
    empty_edge_index = torch.empty((2, 0), dtype=torch.long)

    # Test forward pass with empty edge_index
    with pytest.raises(ValueError, match="Edge index is empty"):
        indices, values = edge_predictor(x, empty_edge_index)

    # Test compute_simplified_adjacency with empty edge_index
    empty_attention_scores = torch.empty(0)
    with pytest.raises(ValueError, match="Edge index is empty"):
        indices, values = edge_predictor.compute_simplified_adjacency(
            empty_attention_scores, empty_edge_index
        )


def test_feature_transformation():
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=2)
    x = torch.rand(5, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    # Get intermediate features
    knn_edges = knn_graph(x, k=2, flow="target_to_source")
    extended_edges = torch.cat([edge_index, knn_edges], dim=1)
    features = edge_predictor.devconv(x, extended_edges)

    # Check feature dimensions
    assert features.shape == (5, 64)  # [num_nodes, hidden_channels]

    # Check transformed features through attention layers
    q = edge_predictor.W_q(features)
    k = edge_predictor.W_k(features)
    assert q.shape == (5, 64)
    assert k.shape == (5, 64)
