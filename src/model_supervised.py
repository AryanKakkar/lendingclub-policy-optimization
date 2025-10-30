import torch, torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def train_mlp(X_tr, y_tr, X_va, y_va, pos_weight=None, epochs=20, bs=4096, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss() if pos_weight is None else \
              nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(),
                                      torch.from_numpy(y_tr).float()),
                        batch_size=bs, shuffle=True)

    best_auc, best_state = -1, None
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        with torch.no_grad():
            va_logits = model(torch.from_numpy(X_va).float().to(device)).cpu().numpy().ravel()
            va_probs  = 1/(1+np.exp(-va_logits))
            auc_val   = roc_auc_score(y_va, va_probs)
        if auc_val > best_auc:
            best_auc, best_state = auc_val, model.state_dict().copy()
        print(f"Epoch {ep:02d} | Val AUC: {auc_val:.4f}")
    model.load_state_dict(best_state)
    return model, best_auc

def predict_proba(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device)).cpu().numpy().ravel()
    return 1/(1+np.exp(-logits))
