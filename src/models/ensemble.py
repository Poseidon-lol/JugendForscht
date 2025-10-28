# src/models/ensemble.py

from pathlib import Path
import sys

# Ensure the project root (with src/) is on sys.path
PROJECT_ROOT = Path().resolve()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "src").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("Could not locate project root containing src/")

import os
import torch
import numpy as np
from src.models.mpnn import MPModel, MoleculeDataset, train_one, evaluate
from src.utils.device import ensure_state_dict_on_cpu, get_device, move_to_device

from pathlib import Path
import sys



# -----------------------------
# Ensemble Training
# -----------------------------
def train_ensemble(df, model_dir, n_models=3, epochs=20, batch_size=16, lr=1e-3, device=None):
    """
    Train n_models on the same dataset, save each model to disk.
    """
    device_spec = get_device(device)
    os.makedirs(model_dir, exist_ok=True)
    
    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}")
        dataset = MoleculeDataset(df)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Model init
        node_dim = dataset[0].x.size(1)
        edge_dim = dataset[0].edge_attr.size(1)
        out_dim = dataset[0].y.size(1)
        model = MPModel(node_in_dim=node_dim, edge_in_dim=edge_dim, out_dim=out_dim).to(device_spec.target)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            loss = train_one(model, loader, optimizer, device_spec)
            if (epoch+1) % 5 == 0 or epoch == 0:
                val_mae, _, _ = evaluate(model, loader, device_spec)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss:.4f}, Val MAE: {val_mae:.4f}")
        
        # Save model
        path = os.path.join(model_dir, f"ensemble_model_{i}.pt")
        torch.save(ensure_state_dict_on_cpu(model, device_spec), path)
        print(f"Saved model {i} to {path}")

# -----------------------------
# Ensemble Prediction
# -----------------------------
def ensemble_predict(model_dir, dataset, device=None):
    """
    Load all ensemble models and predict on dataset, return mean + std.
    """
    device_spec = get_device(device)
    n_models = len([f for f in os.listdir(model_dir) if f.endswith('.pt')])
    
    preds = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    for i in range(n_models):
        path = os.path.join(model_dir, f"ensemble_model_{i}.pt")
        model = MPModel(node_in_dim=dataset[0].x.size(1),
                        edge_in_dim=dataset[0].edge_attr.size(1),
                        out_dim=dataset[0].y.size(1))
        state = torch.load(path, map_location=device_spec.map_location)
        model.load_state_dict(state)
        model = model.to(device_spec.target)
        model.eval()
        
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                batch = move_to_device(batch, device_spec)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                all_preds.append(out.cpu().numpy())
        preds.append(np.concatenate(all_preds, axis=0))
    
    preds = np.stack(preds, axis=0)  # shape: (n_models, n_samples, n_targets)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std
