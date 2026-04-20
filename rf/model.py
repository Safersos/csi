import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# ---------------------------------------------
# GRAPH ATTENTION NETWORK (GAT) BLOCKS
# ---------------------------------------------

class LearnableGAT(nn.Module):
    def __init__(self, in_features, out_features, num_nodes=17, dropout=0.0):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 1. "Noisy Hips" Directed Initialization
        # Start with strong independent self-connections
        base_adj = torch.eye(num_nodes) * 5.0 + torch.randn(num_nodes, num_nodes) * 0.05
        # (Removed Hip Bias: Forcing all nodes to sink into 11/12 caused fatal Graph Over-smoothing)
        
        self.adj_matrix = nn.Parameter(base_adj)
        
        # 2. Standard Dropout to force edge sparsity (AlphaDropout destroys Softmax attention weights)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x geometry: [Batch, Window, Nodes, Features]
        adj = torch.softmax(self.adj_matrix, dim=-1) # [Nodes, Nodes] attention weighting
        
        # Engages Sparsity during training so the GAT stops acting like an average pooling layer
        adj = self.dropout_layer(adj)
        
        # Message Passing: Propagates physical connections across the learned topology
        out = torch.einsum('ij,bwjf->bwif', adj, x)
        
        # Residual Connection [CRITICAL]: Prevents deep GAT from over-smoothing into the center of mass
        return self.activation(self.linear(out)) + x

class BiomechanicGATHead(nn.Module):
    def __init__(self, d_model=128, num_nodes=17, node_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        
        # Bottleneck Isolation: Protects the shared 128-d Temporal Transformer from the 544-d physical explosion!
        self.node_proj = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_nodes * node_dim)
        )
        
        # Hybrid GAT Depth
        self.gat1 = LearnableGAT(node_dim, node_dim, num_nodes)
        self.gat2 = LearnableGAT(node_dim, node_dim, num_nodes)
        
        # Rigid coordinate finalization per node
        self.coord_proj = nn.Linear(node_dim, 2)
        
    def forward(self, x):
        # x: [Batch, Window, d_model]
        B, W, _ = x.shape
        nodes = self.node_proj(x)
        nodes = nodes.view(B, W, self.num_nodes, self.node_dim)
        
        out = self.gat1(nodes)
        out = self.gat2(out)
        
        # Output geometry matched perfectly for PhysicsLoss: [Batch, Window, 17, 2]
        return self.coord_proj(out)

class SaferPINN(nn.Module):
    def __init__(self, w_seq=60, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        
        # Spatial Subcarrier Fusion (Phase & Magnitude Cross-Correlation)
        self.conv_fusion = nn.Conv1d(2, 32, kernel_size=5, padding=2)
        
        # 32 channels * 213 raw subcarriers
        self.feature_proj = nn.Linear(32 * 213, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=w_seq)
        
        # Temporal Attention Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ---------------------------------------------
        # THE DUAL-HEAD ARCHITECTURE WITH GAT (PHASE VI)
        # ---------------------------------------------
        
        # Head 1: The Navigator (Global Trajectory)
        # Predicts the macro center-of-mass (Mid-Hip) across the temporal window
        self.head_navigator = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        
        # Head 2: The Biomechanic (Local Graph)
        # Predicts 17 joints with Learnable Edge Propagation
        self.head_biomechanic = BiomechanicGATHead(d_model=d_model, num_nodes=17, node_dim=32)

    def forward(self, x):
        """
        x: [B, W, 213, 2]
        """
        B, W, S, C = x.shape
        
        # Flatten time into batch to push spatially into Conv1D
        x_cnn = x.view(B * W, S, C).permute(0, 2, 1) # [B*W, 2, 213]
        
        fused = torch.relu(self.conv_fusion(x_cnn))  # [B*W, 32, 213]
        fused = fused.view(B, W, -1)                 # [B, W, 32*213]
        
        # Map physical bandwidth down to hidden temporal dimensionality
        tf_input = self.feature_proj(fused)          # [B, W, d_model]
        tf_input = self.pos_encoder(tf_input)
        
        tf_out = self.transformer(tf_input)          # [B, W, d_model]
        
        # --- Dual Head Branching ---
        root_pred = self.head_navigator(tf_out)      # [B, W, 2]
        local_pred = self.head_biomechanic(tf_out)   # [B, W, 17, 2]
        
        return root_pred, local_pred
