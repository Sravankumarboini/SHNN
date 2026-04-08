from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import CNN
from attacks.fault_injection import inject_fault
from evaluation.metrics import accuracy


# -------------------------------
# Setup
# -------------------------------
app = Flask(__name__)
CORS(app)

device = torch.device("cpu")

# -------------------------------
# Load Model
# -------------------------------
model = CNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

loss_fn = torch.nn.CrossEntropyLoss()

# -------------------------------
# Dataset
# -------------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=128)


# -------------------------------
# Preprocess Image
# -------------------------------
def preprocess(img):

    img = img.convert("L")
    img = img.resize((28, 28))

    img_np = np.array(img)

    # invert colors (MNIST format)
    img_np = 255 - img_np

    # normalize
    img_np = img_np / 255.0

    img_tensor = torch.tensor(img_np, dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    return img_tensor


# -------------------------------
# Prediction Function
# -------------------------------
def predict(img_tensor):

    with torch.no_grad():

        output = model(img_tensor)

        probs = F.softmax(output, dim=1)

        pred = probs.argmax(1).item()
        conf = probs.max().item()

    return int(pred), float(conf)


# -------------------------------
# Home Page
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# Prediction API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict_api():

    # Reset model for every request
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    file = request.files["file"]
    img = Image.open(file.stream)

    img_tensor = preprocess(img)

    # ===============================
    # BEFORE FAULT
    # ===============================
    normal_pred, normal_conf = predict(img_tensor)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):

            if i >= 50:
                break

            out = model(x)
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    baseline_acc = accuracy(correct, total)

    # ===============================
    # AFTER FAULT
    # ===============================
    inject_fault(model)

    faulty_pred, faulty_conf = predict(img_tensor)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):

            if i >= 20:
                break

            out = model(x)
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    faulty_acc = accuracy(correct, total)

    # ===============================
    # SELF HEALING
    # ===============================
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for i, (x, y) in enumerate(train_loader):

        if i >= 30:
            break

        out = model(x)

        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    healed_pred, healed_conf = predict(img_tensor)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):

            if i >= 20:
                break

            out = model(x)
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    healed_acc = accuracy(correct, total)

    # ===============================
    # RETURN JSON
    # ===============================
    return jsonify({
        "before_fault": [int(normal_pred), round(normal_conf, 2)],
        "after_fault": [int(faulty_pred), round(faulty_conf, 2)],
        "after_healing": [int(healed_pred), round(healed_conf, 2)],
        "baseline_acc": float(round(baseline_acc, 2)),
        "faulty_acc": float(round(faulty_acc, 2)),
        "healed_acc": float(round(healed_acc, 2))
    })


# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)