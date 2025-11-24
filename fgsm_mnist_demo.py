# -*- coding: utf-8 -*-
"""
FGSM Demonstration: Prediction Algorithms, Attacks & Countermeasures
Author: Pratik Ashok Paranjape

What this script shows:
1. Train a CNN on MNIST (prediction algorithm).
2. Evaluate accuracy on clean test data.
3. Craft adversarial examples with FGSM and measure robustness (accuracy vs ε).
4. Visualize original vs adversarial image and perturbation.
5. Apply COUNTERMEASURES:
   - Adversarial Training (retrain on adversarial examples).
   - Simple Preprocessing Defense (input quantization).
6. Re-evaluate robustness after defenses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from typing import List, Tuple

# ------------------------------------------------------------
# 1. Model Definition: Simple CNN (Prediction Algorithm)
# ------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # -> 16 x 28 x 28
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)                           # -> 16 x 14 x 14

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # -> 32 x 14 x 14
        self.bn2 = nn.BatchNorm2d(32)                            # -> 32 x 14 x 14
        # pool again -> 32 x 7 x 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------------------------------------------
# 2. FGSM Attack Implementation
# ------------------------------------------------------------

def fgsm_attack(model: nn.Module,
                loss_fn: nn.Module,
                images: torch.Tensor,
                labels: torch.Tensor,
                epsilon: float) -> torch.Tensor:
    """
    Generate FGSM adversarial examples.

    x_adv = x + epsilon * sign(∇_x J(θ, x, y))
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True

    outputs = model(images)
    loss = loss_fn(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.data.sign()
    adv_images = images + epsilon * grad_sign
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()

# ------------------------------------------------------------
# 3. Simple Preprocessing Defense (Input Quantization)
# ------------------------------------------------------------

def preprocessing_defense(x: torch.Tensor, levels: int = 16) -> torch.Tensor:
    """
    Quantize input to reduce effect of small perturbations.
    levels = number of quantization bins (e.g., 16).
    """
    x = x.clone()
    x = torch.clamp(x, 0, 1)
    x = torch.round(x * (levels - 1)) / (levels - 1)
    return x

# ------------------------------------------------------------
# 4. Training & Evaluation Utilities
# ------------------------------------------------------------

def train(model, trainloader, optimizer, loss_fn, epochs=1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[Standard Train] Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f}")


def adversarial_train_epoch(model, trainloader, optimizer, loss_fn, epsilon: float):
    """
    One epoch of adversarial training:
    - Generate FGSM adversarial examples on the fly
    - Train on a mix of clean + adversarial images
    """
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Generate adversarial examples for current batch
        adv_inputs = fgsm_attack(model, loss_fn, inputs, labels, epsilon)

        # Combine clean and adversarial images (simple concat)
        combined_inputs = torch.cat([inputs, adv_inputs], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)

        optimizer.zero_grad()
        outputs = model(combined_inputs)
        loss = loss_fn(outputs, combined_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"[Adversarial Train] Epsilon={epsilon} - Loss: {running_loss/len(trainloader):.4f}")


def evaluate(model, testloader, loss_fn) -> float:
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = 100.0 * correct / total
    print(f"[Eval Clean] Loss: {test_loss/len(testloader):.4f} | Accuracy: {acc:.2f}%")
    return acc


def evaluate_fgsm(model, testloader, loss_fn, epsilon_values: List[float],
                  use_preprocessing: bool = False,
                  title_prefix: str = "") -> List[Tuple[float, float]]:
    """
    For each epsilon, run FGSM on the test set and compute adversarial accuracy.
    If use_preprocessing=True, apply preprocessing_defense to attacked images.
    """
    model.eval()
    results = []
    mode = "With Preprocessing" if use_preprocessing else "Raw Attack"
    print(f"\n=== FGSM Evaluation ({mode}) ===")

    for eps in epsilon_values:
        correct = 0
        total = 0
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            adv_inputs = fgsm_attack(model, loss_fn, inputs, labels, eps)

            if use_preprocessing:
                adv_inputs = preprocessing_defense(adv_inputs)

            outputs = model(adv_inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        acc = 100.0 * correct / total
        print(f"{title_prefix} Epsilon={eps:.3f} | Adversarial Accuracy: {acc:.2f}%")
        results.append((eps, acc))
    return results

# ------------------------------------------------------------
# 5. Visualization Helpers
# ------------------------------------------------------------

def imshow(img, title=None):
    img = img.squeeze().cpu().numpy()
    plt.imshow(img, cmap="gray")
    if title is not None:
        plt.title(title)
    plt.axis("off")


def show_adversarial_example(model, testloader, loss_fn, epsilon: float, use_preprocessing: bool = False):
    """
    Show original image, perturbation and adversarial image side-by-side.
    If use_preprocessing=True, show the defended adversarial image instead.
    """
    model.eval()
    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, orig_pred = torch.max(outputs, 1)

    adv_images = fgsm_attack(model, loss_fn, images, labels, epsilon)
    if use_preprocessing:
        adv_images_def = preprocessing_defense(adv_images)
        outputs_adv = model(adv_images_def)
    else:
        outputs_adv = model(adv_images)

    _, adv_pred = torch.max(outputs_adv, 1)

    perturbation = adv_images - images

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 3, 1)
    imshow(images[0], title=f"Original (pred: {orig_pred[0].item()})")

    plt.subplot(1, 3, 2)
    imshow(perturbation[0] * 5 + 0.5, title="Perturbation (amplified)")

    plt.subplot(1, 3, 3)
    final_img = preprocessing_defense(adv_images[0].unsqueeze(0))[0] if use_preprocessing else adv_images[0]
    imshow(final_img, title=f"{'Defended ' if use_preprocessing else ''}Adv (pred: {adv_pred[0].item()})")

    plt.suptitle(f"FGSM example (ε = {epsilon}, {'with preprocessing' if use_preprocessing else 'no defense'})")
    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_epsilon(results_before, results_after=None, label_before="Before Defense", label_after="After Defense"):
    eps_before = [r[0] for r in results_before]
    acc_before = [r[1] for r in results_before]

    plt.figure(figsize=(7, 4))
    plt.plot(eps_before, acc_before, marker="o", label=label_before)

    if results_after is not None:
        eps_after = [r[0] for r in results_after]
        acc_after = [r[1] for r in results_after]
        plt.plot(eps_after, acc_after, marker="s", label=label_after)

    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Adversarial Accuracy (%)")
    plt.title("Model Robustness vs FGSM Attack Strength")
    plt.grid(True)
    plt.legend()
    plt.show()

# ------------------------------------------------------------
# 6. Main: End-to-End Demo (Attack + Countermeasures)
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 6.1 Data
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # 6.2 Model, Loss, Optimizer
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 6.3 Standard Training (Prediction Algorithm)
    print("=== Stage 1: Train Standard Model (No Defense) ===")
    train(model, trainloader, optimizer, loss_fn, epochs=2)
    print("\n=== Evaluation on Clean Test Data ===")
    clean_acc_before = evaluate(model, testloader, loss_fn)

    # 6.4 FGSM Attack Evaluation (no defense)
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    results_before = evaluate_fgsm(model, testloader, loss_fn, epsilons, use_preprocessing=False,
                                   title_prefix="[No Defense]")

    # 6.5 Show one adversarial example (no defense)
    show_adversarial_example(model, testloader, loss_fn, epsilon=0.2, use_preprocessing=False)

    # 6.6 Simple Preprocessing Defense Demo
    print("\n=== Stage 2: Same Model + Preprocessing Defense (Quantization) ===")
    results_preproc = evaluate_fgsm(model, testloader, loss_fn, epsilons, use_preprocessing=True,
                                    title_prefix="[Preprocessing Defense]")
    show_adversarial_example(model, testloader, loss_fn, epsilon=0.2, use_preprocessing=True)

    # 6.7 Adversarial Training (strong countermeasure)
    print("\n=== Stage 3: Adversarial Training (FGSM-based) ===")
    adv_train_eps = 0.2  # training epsilon
    # Reuse optimizer (or re-create with smaller LR for fine-tuning)
    for epoch in range(2):
        adversarial_train_epoch(model, trainloader, optimizer, loss_fn, epsilon=adv_train_eps)

    print("\n=== Evaluation on Clean Test Data AFTER Adversarial Training ===")
    clean_acc_after = evaluate(model, testloader, loss_fn)

    print("\n=== FGSM Evaluation AFTER Adversarial Training ===")
    results_after = evaluate_fgsm(model, testloader, loss_fn, epsilons, use_preprocessing=False,
                                  title_prefix="[Adversarially Trained]")

    # 6.8 Compare robustness curves
    plot_accuracy_vs_epsilon(results_before, results_after,
                             label_before="Before Adversarial Training",
                             label_after="After Adversarial Training")

    print("\nSummary:")
    print(f"Clean Accuracy Before Defense: {clean_acc_before:.2f}%")
    print(f"Clean Accuracy After Adversarial Training: {clean_acc_after:.2f}%")

if __name__ == "__main__":
    main()