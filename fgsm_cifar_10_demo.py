# -*- coding: utf-8 -*-
"""
FGSM Demonstration on CIFAR-10:
Prediction Algorithms, Attacks & Countermeasures
Author: Pratik Ashok Paranjape

This script shows:
1. A CNN prediction model trained on CIFAR-10.
2. FGSM adversarial attacks at various epsilon values.
3. Accuracy vs epsilon (robustness curve).
4. A guaranteed "successful" adversarial example + confidence shift.
5. Preprocessing defense (quantization).
6. Adversarial training as a strong countermeasure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# ------------------------------------------------------------
# 1. Model Definition: Simple CNN for CIFAR-10 (Prediction Algorithm)
# ------------------------------------------------------------

class SimpleCIFARCNN(nn.Module):
    def __init__(self):
        super(SimpleCIFARCNN, self).__init__()
        # Input: 3 x 32 x 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # -> 32 x 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)                            # -> 32 x 16 x 16

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> 64 x 16 x 16
        self.bn2 = nn.BatchNorm2d(64)                             # -> 64 x 16 x 16
        # pool again -> 64 x 8 x 8

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> 128 x 8 x 8
        self.bn3 = nn.BatchNorm2d(128)
        # pool -> 128 x 4 x 4

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))         # 32 x 16 x 16
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))         # 64 x 8 x 8
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))         # 128 x 4 x 4
        x = x.view(x.size(0), -1)                                  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------------------------------------------
# 2. FGSM Attack Implementation
# ------------------------------------------------------------

def fgsm_attack(
    model: nn.Module,
    loss_fn: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float
) -> torch.Tensor:
    """
    Generate FGSM adversarial examples:
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
    Quantize the input to reduce the effect of small perturbations.
    levels: number of quantization bins (e.g., 16).
    """
    x = torch.clamp(x, 0, 1)
    x = torch.round(x * (levels - 1)) / (levels - 1)
    return x

# ------------------------------------------------------------
# 4. Training & Evaluation Utilities
# ------------------------------------------------------------

def train(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int = 1
) -> None:
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


def adversarial_train_epoch(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    epsilon: float
) -> None:
    """
    One epoch of adversarial training:
    - Generate FGSM adversarial examples on the fly.
    - Train on a mix of clean and adversarial images.
    """
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        adv_inputs = fgsm_attack(model, loss_fn, inputs, labels, epsilon)

        combined_inputs = torch.cat([inputs, adv_inputs], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)

        optimizer.zero_grad()
        outputs = model(combined_inputs)
        loss = loss_fn(outputs, combined_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"[Adversarial Train] Epsilon={epsilon} - Loss: {running_loss/len(trainloader):.4f}")


def evaluate(
    model: nn.Module,
    testloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module
) -> float:
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


def evaluate_fgsm(
    model: nn.Module,
    testloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    epsilon_values: List[float],
    use_preprocessing: bool = False,
    title_prefix: str = ""
) -> List[Tuple[float, float]]:
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
# 5. Visualization & Example Helpers
# ------------------------------------------------------------

def imshow(img: torch.Tensor, title: Optional[str] = None) -> None:
    """Show a 3-channel CIFAR-10 image."""
    img = img.cpu().numpy()
    # img shape: (3, H, W) -> (H, W, 3)
    img = img.transpose((1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis("off")


def find_successful_attack_sample(
    model: nn.Module,
    testloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    epsilon: float,
    max_batches: int = 50
):
    """
    Find a test sample where:
    - original prediction is correct
    - adversarial prediction is different (attack succeeded)
    Returns (orig_img, adv_img, label, orig_pred, adv_pred) or None.
    """
    model.eval()
    checked = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, orig_preds = torch.max(outputs, 1)

        adv_images = fgsm_attack(model, loss_fn, images, labels, epsilon)
        outputs_adv = model(adv_images)
        _, adv_preds = torch.max(outputs_adv, 1)

        for i in range(images.size(0)):
            if orig_preds[i] == labels[i] and adv_preds[i] != orig_preds[i]:
                print(f"Found successful attack at epsilon={epsilon}")
                return (
                    images[i:i+1].detach(),
                    adv_images[i:i+1].detach(),
                    int(labels[i].item()),
                    int(orig_preds[i].item()),
                    int(adv_preds[i].item()),
                )

        checked += 1
        if checked >= max_batches:
            break

    print(f"No successful attack found within first {max_batches} batches for epsilon={epsilon}.")
    return None


def show_example_with_confidences(
    model: nn.Module,
    orig_img: torch.Tensor,
    adv_img: torch.Tensor,
    label: int,
    class_names: List[str]
) -> None:
    """
    Show original vs adversarial image and print probability distribution shift.
    """
    model.eval()
    with torch.no_grad():
        out_clean = model(orig_img.to(device))
        out_adv = model(adv_img.to(device))

    probs_clean = F.softmax(out_clean[0], dim=0).cpu()
    probs_adv = F.softmax(out_adv[0], dim=0).cpu()

    _, pred_clean = torch.max(out_clean, 1)
    _, pred_adv = torch.max(out_adv, 1)

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    imshow(orig_img[0], title=f"Original (pred: {class_names[pred_clean.item()]})")

    plt.subplot(1, 3, 2)
    perturbation = (adv_img - orig_img)[0]
    # Amplify perturbation for visibility
    pert_vis = torch.clamp(perturbation * 5 + 0.5, 0, 1)
    imshow(pert_vis, title="Perturbation (amplified)")

    plt.subplot(1, 3, 3)
    imshow(adv_img[0], title=f"Adversarial (pred: {class_names[pred_adv.item()]})")

    plt.suptitle(f"Label={class_names[label]}, pred_clean={class_names[pred_clean.item()]}, pred_adv={class_names[pred_adv.item()]}")
    plt.tight_layout()
    plt.show()

    print("\nTop-3 probabilities (clean):")
    top_clean = torch.topk(probs_clean, 3)
    for score, cls_idx in zip(top_clean.values, top_clean.indices):
        print(f"Class {class_names[int(cls_idx)]} ({int(cls_idx)}): {float(score):.4f}")

    print("\nTop-3 probabilities (adversarial):")
    top_adv = torch.topk(probs_adv, 3)
    for score, cls_idx in zip(top_adv.values, top_adv.indices):
        print(f"Class {class_names[int(cls_idx)]} ({int(cls_idx)}): {float(score):.4f}")


def plot_accuracy_vs_epsilon(
    results_before: List[Tuple[float, float]],
    results_after: Optional[List[Tuple[float, float]]] = None,
    label_before: str = "Before Defense",
    label_after: str = "After Adversarial Training"
) -> None:
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
    plt.title("CIFAR-10 Robustness vs FGSM Attack Strength")
    plt.grid(True)
    plt.legend()
    plt.show()

# ------------------------------------------------------------
# 6. Main: End-to-End Demo (Attack + Countermeasures on CIFAR-10)
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 6.1 CIFAR-10 Data
    transform = transforms.ToTensor()  # keep [0,1] range for simplicity with FGSM

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    class_names = trainset.classes  # ['airplane', 'automobile', ..., 'truck']

    # 6.2 Model, Loss, Optimizer
    model = SimpleCIFARCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ------------------ Stage 1: Standard Training ------------------
    print("=== Stage 1: Train Standard CIFAR-10 Model (No Defense) ===")
    # For a real run, increase epochs (e.g., 10–20) but for demo, keep small:
    train(model, trainloader, optimizer, loss_fn, epochs=3)

    print("\n=== Evaluation on Clean Test Data ===")
    clean_acc_before = evaluate(model, testloader, loss_fn)

    # ------------------ Stage 2: FGSM Raw Attack --------------------
    epsilons = [0.0, 0.01, 0.03, 0.05, 0.1]  # For CIFAR, smaller eps recommended
    results_before = evaluate_fgsm(
        model, testloader, loss_fn, epsilons, use_preprocessing=False,
        title_prefix="[No Defense]"
    )

    # Find a successful adversarial sample for presentation
    print("\n=== Searching for a Successful Attack Example (Raw FGSM) ===")
    epsilon_demo = 0.03
    example = find_successful_attack_sample(model, testloader, loss_fn, epsilon=epsilon_demo, max_batches=80)

    if example is not None:
        orig_img, adv_img, label, orig_pred, adv_pred = example
        print(f"\nSuccessful example: label={class_names[label]}, "
              f"orig_pred={class_names[orig_pred]}, adv_pred={class_names[adv_pred]}")
        show_example_with_confidences(model, orig_img, adv_img, label, class_names)
    else:
        print("Could not find a label-flip example within the limit; still, dataset-level accuracy drop demonstrates attack.")

    # ------------------ Stage 3: Preprocessing Defense --------------
    print("\n=== Stage 3: Same Model + Preprocessing Defense (Quantization) ===")
    results_preproc = evaluate_fgsm(
        model, testloader, loss_fn, epsilons, use_preprocessing=True,
        title_prefix="[Preprocessing Defense]"
    )

    # ------------------ Stage 4: Adversarial Training ---------------
    print("\n=== Stage 4: Adversarial Training (FGSM-based) ===")
    adv_train_eps = 0.03
    for epoch in range(2):
        adversarial_train_epoch(model, trainloader, optimizer, loss_fn, epsilon=adv_train_eps)

    print("\n=== Evaluation on Clean Test Data AFTER Adversarial Training ===")
    clean_acc_after = evaluate(model, testloader, loss_fn)

    print("\n=== FGSM Evaluation AFTER Adversarial Training ===")
    results_after = evaluate_fgsm(
        model, testloader, loss_fn, epsilons, use_preprocessing=False,
        title_prefix="[Adversarially Trained]"
    )

    # Compare robustness before vs after adversarial training
    plot_accuracy_vs_epsilon(
        results_before,
        results_after,
        label_before="Before Adversarial Training",
        label_after="After Adversarial Training"
    )

    print("\nSummary:")
    print(f"Clean Accuracy Before Defense: {clean_acc_before:.2f}%")
    print(f"Clean Accuracy After Adversarial Training: {clean_acc_after:.2f}%")

if __name__ == "__main__":
    main()