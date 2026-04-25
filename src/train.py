import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.tinyvgg import TinyVGG
from utils.helpers import accuracy_fn, save_model
import time

def main():
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28))
    ])

    # Dataset
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    # Dataloaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # Model
    model = TinyVGG(input_shape=1, hidden_units=32, output_shape=10).to(device)

    # Loss + Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = 0

        start = time.time()

        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end = time.time()

        # Evaluation
        model.eval()
        test_acc = 0
        with torch.inference_mode():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                test_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Test Accuracy: {test_acc/len(test_loader):.2f}%")
        print(f"Epoch time: {end - start:.2f} seconds\n")

    # Save model
    save_model(model, target_dir="models", model_name="tinyvgg_fashionmnist.pth")

if __name__ == "__main__":
    main()
