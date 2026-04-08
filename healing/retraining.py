import torch
import torch.nn as nn

def heal_model(model, data_loader=None):
    print("⚠ Healing activated: Reinitializing + quick retraining")

    # Reinitialize damaged layer
    nn.init.kaiming_normal_(model.conv.weight)
    if model.conv.bias is not None:
        nn.init.zeros_(model.conv.bias)

    # 🔥 QUICK MICRO-RETRAINING
    if data_loader is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for i, (x, y) in enumerate(data_loader):
            if i >= 5:   # only 5 mini-batches (FAST)
                break
            out = model(x)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("✅ Quick self-healing retraining done")
