import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image

from models.cnn import CNN
from attacks.fault_injection import inject_fault
from evaluation.metrics import accuracy


# ===============================
# Device Setup
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("\n==============================")
print("SELF-HEALING NEURAL NETWORK")
print("==============================\n")


# ===============================
# Load MNIST Data
# ===============================
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)


# ===============================
# Model Setup
# ===============================
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()


# ===============================
# 1. Train Baseline Model
# ===============================
print("[1] Training Baseline Model...")

model.train()
for epoch in range(3):   # Fast + Stable
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Baseline Training Completed.")

torch.save(model.state_dict(), "model.pth")
print("Model saved!")

# ===============================
# Baseline Evaluation
# ===============================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

baseline_acc = accuracy(correct, total)
print("Baseline Accuracy (Clean):", round(baseline_acc, 4))


# ===============================
# 2. Inject Fault
# ===============================
print("\n[2] Injecting Internal Fault...")
inject_fault(model)


# ===============================
# Faulty Evaluation
# ===============================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

faulty_acc = accuracy(correct, total)
print("Accuracy After Fault:", round(faulty_acc, 4))


# ===============================
# 3. Self-Healing Phase
# ===============================
print("\n[3] Activating Self-Healing...")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(1):
    for i, (x, y) in enumerate(train_loader):
        if i >= 50:   # Small recovery training
            break

        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Self-healing completed.")


# ===============================
# Evaluation After Healing
# ===============================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

healed_acc = accuracy(correct, total)
print("Accuracy After Self-Healing:", round(healed_acc, 4))


# ===============================  
# Final Comparison
# ===============================
print("\n==============================")
print("FINAL COMPARISON")
print("==============================")
print("Baseline Accuracy     :", round(baseline_acc, 2))
print("Faulty Model Accuracy :", round(faulty_acc, 2))
print("Healed Model Accuracy :", round(healed_acc, 2))
print("==============================")


# ======================================================
# 5. Single Image Demonstration
# ======================================================
print("\n[5] Single Image Self-Healing Demonstration")

# -------- Proper MNIST-style preprocessing --------
img = Image.open("samples/digit9.png").convert("L")

img_np = np.array(img)
img_np = np.array(Image.fromarray(img_np).resize((28, 28)))

# Invert colors to match MNIST
img_np = 255 - img_np

# Normalize
img_np = img_np / 255.0

# Convert to tensor
img_tensor = torch.tensor(img_np, dtype=torch.float32)
img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)


# -------- BEFORE FAULT --------
model.eval()
with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    normal_pred = probs.argmax(1).item()
    normal_conf = probs.max().item()

print("\n--- BEFORE FAULT ---")
print("Predicted Digit :", normal_pred)



# -------- Inject Fault Again --------
print("\nInjecting Fault Again for Demo...")
inject_fault(model)

model.eval()
with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    faulty_pred = probs.argmax(1).item()
    faulty_conf = probs.max().item()

print("\n--- AFTER FAULT ---")
print("Predicted Digit :", faulty_pred)



# -------- Heal Again --------
print("\nActivating Healing...")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(1):
    for i, (x, y) in enumerate(train_loader):
        if i >= 30:
            break

        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    healed_pred = probs.argmax(1).item()
    healed_conf = probs.max().item()

print("\n--- AFTER SELF-HEALING ---")
print("Predicted Digit :", healed_pred)

