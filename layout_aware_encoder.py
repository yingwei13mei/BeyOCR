# layout_aware_encoder.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class LayoutAwareBERT(nn.Module):
    """
    BERT model enhanced with 2D positional information for document understanding.
    """
    
    def __init__(self, bert_model_name="bert-base-uncased"):
        super().__init__()
        
        # Load pretrained BERT
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Spatial embedding layers
        self.x_embedding = nn.Embedding(1000, 128)  # Quantized x-coordinate embedding
        self.y_embedding = nn.Embedding(1000, 128)  # Quantized y-coordinate embedding
        self.width_embedding = nn.Embedding(100, 128)  # Width embedding
        self.height_embedding = nn.Embedding(100, 128)  # Height embedding
        
        # Project spatial embeddings to BERT hidden dimension
        bert_dim = self.bert.config.hidden_size
        self.spatial_projection = nn.Linear(512, bert_dim)
        
        # Final layer
        self.output_layer = nn.Linear(bert_dim, bert_dim)
        
    def forward(self, input_ids, attention_mask, bbox):
        """
        Forward pass with text and layout information.
        
        Args:
            input_ids: Token IDs from BERT tokenizer [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            bbox: Bounding box coordinates for each token [batch_size, seq_len, 4]
                Each bbox contains (x_min, y_min, x_max, y_max) normalized to [0, 999]
                
        Returns:
            Document representation enhanced with layout information
        """
        # Get BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = bert_outputs.last_hidden_state  # [batch, seq_len, bert_dim]
        
        # Extract and quantize bbox coordinates
        x_min = bbox[:, :, 0].clamp(0, 999).long()  # [batch, seq_len]
        y_min = bbox[:, :, 1].clamp(0, 999).long()  # [batch, seq_len]
        
        # Calculate width and height and quantize
        width = (bbox[:, :, 2] - bbox[:, :, 0]).clamp(0, 99).long()  # [batch, seq_len]
        height = (bbox[:, :, 3] - bbox[:, :, 1]).clamp(0, 99).long()  # [batch, seq_len]
        
        # Get spatial embeddings
        x_emb = self.x_embedding(x_min)  # [batch, seq_len, 128]
        y_emb = self.y_embedding(y_min)  # [batch, seq_len, 128]
        w_emb = self.width_embedding(width)  # [batch, seq_len, 128]
        h_emb = self.height_embedding(height)  # [batch, seq_len, 128]
        
        # Combine spatial embeddings
        spatial_emb = torch.cat([x_emb, y_emb, w_emb, h_emb], dim=2)  # [batch, seq_len, 512]
        spatial_emb = self.spatial_projection(spatial_emb)  # [batch, seq_len, bert_dim]
        
        # Combine with text embeddings
        combined_embeddings = text_embeddings + spatial_emb  # [batch, seq_len, bert_dim]
        
        # Final transformation
        output = self.output_layer(combined_embeddings)  # [batch, seq_len, bert_dim]
        
        return output
