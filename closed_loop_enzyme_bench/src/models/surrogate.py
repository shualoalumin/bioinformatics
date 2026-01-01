import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, EsmModel
from dataclasses import dataclass
from typing import List
import numpy as np
from tqdm import tqdm

@dataclass
class SurrogateConfig:
    esm2_model_id: str = "facebook/esm2_t6_8M_UR50D"
    pool: str = "mean"
    epochs: int = 10
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    hidden_dim: int = 64

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

def get_embeddings(sequences: List[str], model_id: str, pool_mode: str, device: str) -> torch.Tensor:
    print(f"Loading ESM-2 model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmModel.from_pretrained(model_id).to(device)
    model.eval()
    
    embeddings = []
    batch_size = 16
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
            batch_seqs = sequences[i:i+batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            outputs = model(**inputs)
            # last_hidden_state: [batch, seq_len, dim]
            # mask: [batch, seq_len]
            mask = inputs.attention_mask.bool()
            
            if pool_mode == "mean":
                # Exclude [CLS] and [SEP]? ESM uses standard BERT tokens.
                # Simply mean over valid tokens
                emb = []
                for j in range(len(batch_seqs)):
                    # mask[j] includes special tokens.
                    # We can just take mean over all non-pad tokens for simplicity
                    # or strictly exclude CLS/EOS. Let's do simple mean over valid mask.
                    seq_len = mask[j].sum()
                    token_embs = outputs.last_hidden_state[j, :seq_len, :]
                    emb.append(token_embs.mean(dim=0))
                embeddings.append(torch.stack(emb))
            elif pool_mode == "cls":
                embeddings.append(outputs.last_hidden_state[:, 0, :])
            else:
                raise ValueError(f"Unknown pool mode: {pool_mode}")
                
    return torch.cat(embeddings).cpu()

def train_surrogate(X: torch.Tensor, y: torch.Tensor, cfg: SurrogateConfig) -> nn.Module:
    input_dim = X.shape[1]
    model = MLP(input_dim, cfg.hidden_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    print(f"Training surrogate on {len(X)} samples for {cfg.epochs} epochs...")
    model.train()
    for ep in range(cfg.epochs):
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(cfg.device), by.to(cfg.device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        # print(f"Epoch {ep+1}: Loss {total_loss/len(X):.4f}")
        
    return model

def predict_surrogate(model: nn.Module, X: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            preds.append(model(bx))
    return torch.cat(preds).cpu()
