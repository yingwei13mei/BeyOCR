# multimodal_fusion.py
import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    """
    Implements cross-modal attention between text and visual features.
    """
    
    def __init__(self, text_dim=768, visual_dim=2048, hidden_dim=512):
        super().__init__()
        
        # Projection layers
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        
        # Attention mechanisms
        self.text_to_visual_attention = nn.MultiheadAttention(hidden_dim, 8)
        self.visual_to_text_attention = nn.MultiheadAttention(hidden_dim, 8)
        
        # Output layer
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, text_features, visual_features):
        """
        Apply cross-modal attention between text and visual features.
        
        Args:
            text_features: Features from text modality [batch_size, seq_len, text_dim]
            visual_features: Features from visual modality [batch_size, regions, visual_dim]
            
        Returns:
            Fused multimodal representation
        """
        # Project features to common space
        text_proj = self.text_projection(text_features)  # [batch, seq_len, hidden]
        visual_proj = self.visual_projection(visual_features)  # [batch, regions, hidden]
        
        # Reshape for attention (attention expects [seq_len, batch, hidden])
        text_proj = text_proj.permute(1, 0, 2)
        visual_proj = visual_proj.permute(1, 0, 2)
        
        # Cross-attention
        text_attended, _ = self.text_to_visual_attention(
            query=text_proj, 
            key=visual_proj, 
            value=visual_proj
        )
        
        visual_attended, _ = self.visual_to_text_attention(
            query=visual_proj, 
            key=text_proj, 
            value=text_proj
        )
        
        # Return to original shape [batch, seq_len/regions, hidden]
        text_attended = text_attended.permute(1, 0, 2)
        visual_attended = visual_attended.permute(1, 0, 2)
        
        # Pool attended features
        text_pooled = torch.mean(text_attended, dim=1)  # [batch, hidden]
        visual_pooled = torch.mean(visual_attended, dim=1)  # [batch, hidden]
        
        # Concatenate and fuse
        multimodal = torch.cat([text_pooled, visual_pooled], dim=1)  # [batch, hidden*2]
        fused = self.fusion_layer(multimodal)  # [batch, hidden]
        
        return fused
