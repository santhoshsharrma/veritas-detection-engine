import numpy as np
import torch
import torch.nn.functional as F

from src.eval.metrics import binary_metrics, roc_auc


@torch.no_grad()
def predict(model, loader, device):
    """Return (P(AI-generated), labels) as numpy arrays."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train(model, train_loader, val_loader, device, epochs=3, lr=3e-4, ckpt_path="best.pt"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_auc, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(y)
            n += len(y)
        probs, labels = predict(model, val_loader, device)
        m = binary_metrics(labels, probs)
        auc = roc_auc(labels, probs)
        history.append({"epoch": epoch, "train_loss": running / n, "val_auc": auc, **m})
        print(f"epoch {epoch}: loss {running / n:.4f} | val acc {m['accuracy']:.4f} "
              f"fpr {m['fpr']:.4f} auc {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), ckpt_path)
    return history
