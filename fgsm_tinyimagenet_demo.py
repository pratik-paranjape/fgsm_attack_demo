"""
FGSM Demonstration on Tiny ImageNet (PIL-free, robust loader):
Prediction Algorithms, Attacks & Countermeasures
Author: Pratik Ashok Paranjape

This script shows:
1. A CNN prediction model trained on Tiny ImageNet (200 classes, 64x64).
2. FGSM adversarial attacks at various epsilon values.
3. Accuracy vs epsilon (robustness curve).
4. A "successful" adversarial example + confidence shift.
5. Preprocessing defense (input quantization).
6. Adversarial training as a strong countermeasure.

Expected Tiny ImageNet folder structure:
tiny-imagenet-200/
    train/
        n01443537/
            images/
                *.JPEG
        ...
    val/
        images/
        val_annotations.txt
    wnids.txt
"""

import os
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Safe Image Loader (PIL-free, robust)
# ------------------------------------------------------------

def safe_read_image(path: str, resize_to: Tuple[int, int] = (64, 64)) -> Optional[torch.Tensor]:
    """
    Robust image loader using torchvision.io.read_image.
    - Handles corrupted/unreadable images.
    - Forces 3 channels (RGB).
    - Forces fixed resolution via tensor-based resize.
    Returns:
        img tensor [3,H,W] in [0,1] or None if unreadable.
    """
    try:
        img = read_image(path)  # uint8 tensor [C,H,W]
    except Exception:
        return None

    if img.ndim != 3:
        return None

    # Fix channels to 3 (RGB-like)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] == 4:
        img = img[:3, :, :]
    elif img.shape[0] != 3:
        return None

    # Resize to target size
    img = TF.resize(img, resize_to)  # still uint8
    img = img.float() / 255.0        # [0,1]

    return img


# ------------------------------------------------------------
# 2. Tiny ImageNet Dataset Utilities (PIL-free)
# ------------------------------------------------------------

class TinyImageNetTrain(Dataset):
    """
    Tiny ImageNet training dataset using safe_read_image.
    Expects:
      root/train/<wnid>/images/*.JPEG
    """
    def __init__(self, root: str, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        train_dir = "./tiny-imagenet-200/train"
        wnids_file = "./tiny-imagenet-200/wnids.txt"

        wnids = [w.strip() for w in open(wnids_file)]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

        self.samples = []
        for wnid in wnids:
            wnid_dir = os.path.join(train_dir, wnid, "images")
            if not os.path.isdir(wnid_dir):
                continue
            for fname in os.listdir(wnid_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    path = os.path.join(wnid_dir, fname)
                    label = self.class_to_idx[wnid]
                    self.samples.append((path, label))

        print(f"[TinyImageNetTrain] Loaded {len(self.samples)} images, {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = safe_read_image(path)

        # If unreadable/corrupted, fall back to a black image (or you could resample another index)
        if img is None:
            img = torch.zeros(3, 64, 64)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class TinyImageNetVal(Dataset):
    """
    Tiny ImageNet validation dataset using val_annotations.txt + safe_read_image.
    Expects:
      root/val/images/*.JPEG
      root/val/val_annotations.txt
    """
    def __init__(self, root: str, transform=None, class_to_idx=None):
        super().__init__()
        self.root = root
        self.transform = transform

        val_dir = "./tiny-imagenet-200/val"
        img_dir = os.path.join(val_dir, "images")
        ann_path = os.path.join(val_dir, "val_annotations.txt")

        img_to_wnid = {}
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    filename, wnid = parts[0], parts[1]
                    img_to_wnid[filename] = wnid

        if class_to_idx is None:
            wnids_file = os.path.join(root, "wnids.txt")
            wnids = [w.strip() for w in open(wnids_file)]
            class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

        self.class_to_idx = class_to_idx

        self.samples = []
        for filename, wnid in img_to_wnid.items():
            if wnid not in self.class_to_idx:
                continue
            img_path = os.path.join(img_dir, filename)
            label = self.class_to_idx[wnid]
            self.samples.append((img_path, label))

        print(f"[TinyImageNetVal] Loaded {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = safe_read_image(path)

        if img is None:
            img = torch.zeros(3, 64, 64)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# ------------------------------------------------------------
# 3. Model Definition: CNN for Tiny ImageNet
# ------------------------------------------------------------

class TinyImageNetCNN(nn.Module):
    def __init__(self, num_classes: int = 200):
        super(TinyImageNetCNN, self).__init__()
        # Input: 3 x 64 x 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)   # -> 64 x 64 x 64
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)                            # -> 64 x 32 x 32

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> 128 x 32 x 32
        self.bn2 = nn.BatchNorm2d(128)
        # pool -> 128 x 16 x 16

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)# -> 256 x 16 x 16
        self.bn3 = nn.BatchNorm2d(256)
        # pool -> 256 x 8 x 8

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   # 64 x 32 x 32
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))   # 128 x 16 x 16
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))   # 256 x 8 x 8
        x = x.view(x.size(0), -1)                            # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ------------------------------------------------------------
# 4. FGSM Attack & Preprocessing Defense
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


def preprocessing_defense(x: torch.Tensor, levels: int = 16) -> torch.Tensor:
    """
    Quantize input pixels to reduce the effect of small perturbations.
    """
    x = torch.clamp(x, 0, 1)
    x = torch.round(x * (levels - 1)) / (levels - 1)
    return x


# ------------------------------------------------------------
# 5. Training & Evaluation
# ------------------------------------------------------------

def train(
    model: nn.Module,
    trainloader: DataLoader,
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
    trainloader: DataLoader,
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
    testloader: DataLoader,
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
    testloader: DataLoader,
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
        print(f"{title_prefix} Epsilon={eps:.4f} | Adversarial Accuracy: {acc:.2f}%")
        results.append((eps, acc))
    return results


# ------------------------------------------------------------
# 6. Visualization & Example Helpers
# ------------------------------------------------------------

def imshow(img: torch.Tensor, title: Optional[str] = None) -> None:
    """
    Show 3-channel Tiny ImageNet image (tensor [C,H,W]).
    """
    img = img.cpu().numpy()
    img = img.transpose((1, 2, 0))  # C,H,W -> H,W,C
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis("off")


def find_successful_attack_sample(
    model: nn.Module,
    testloader: DataLoader,
    loss_fn: nn.Module,
    epsilon: float,
    max_batches: int = 50
):
    """
    Find a test sample where:
    - original prediction is correct
    - adversarial prediction is different.
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
    label_idx: int,
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
    pert_vis = torch.clamp(perturbation * 5 + 0.5, 0, 1)
    imshow(pert_vis, title="Perturbation (amplified)")

    plt.subplot(1, 3, 3)
    imshow(adv_img[0], title=f"Adversarial (pred: {class_names[pred_adv.item()]})")

    plt.suptitle(
        f"Label={class_names[label_idx]}, "
        f"pred_clean={class_names[pred_clean.item()]}, "
        f"pred_adv={class_names[pred_adv.item()]}"
    )
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
    plt.title("Tiny ImageNet Robustness vs FGSM Attack Strength")
    plt.grid(True)
    plt.legend()
    plt.show()


# ------------------------------------------------------------
# 7. Main: End-to-End Demo (Attack + Countermeasures)
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Adjust this depending on where your "train/val/wnids.txt" live:
    # If this script is inside tiny-imagenet-200/, use data_root = "."
    # If this script is one folder above, use "./tiny-imagenet-200".
    data_root = "."  # or "./tiny-imagenet-200"

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    transform_val = None  # raw tensor [0,1]

    trainset = TinyImageNetTrain(data_root, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    class_to_idx = trainset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    valset = TinyImageNetVal(data_root, transform=transform_val, class_to_idx=class_to_idx)
    testloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=0)

    num_classes = len(class_to_idx)
    model = TinyImageNetCNN(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -------- Stage 1: Standard Training --------
    print("=== Stage 1: Train Standard Tiny ImageNet Model (No Defense) ===")
    train(model, trainloader, optimizer, loss_fn, epochs=3)

    print("\n=== Evaluation on Clean Validation Data ===")
    clean_acc_before = evaluate(model, testloader, loss_fn)

    # -------- Stage 2: FGSM Raw Attack --------
    epsilons = [0.0, 0.002, 0.005, 0.01, 0.02]
    results_before = evaluate_fgsm(
        model, testloader, loss_fn, epsilons, use_preprocessing=False,
        title_prefix="[No Defense]"
    )

    print("\n=== Searching for a Successful Attack Example (Raw FGSM) ===")
    epsilon_demo = 0.01
    example = find_successful_attack_sample(
        model, testloader, loss_fn, epsilon=epsilon_demo, max_batches=80
    )

    if example is not None:
        orig_img, adv_img, label_idx, orig_pred, adv_pred = example
        print(
            f"\nSuccessful example: label={class_names[label_idx]}, "
            f"orig_pred={class_names[orig_pred]}, adv_pred={class_names[adv_pred]}"
        )
        show_example_with_confidences(model, orig_img, adv_img, label_idx, class_names)
    else:
        print("Could not find a label-flip example within the limit; "
              "dataset-level accuracy drop still demonstrates attack.")

    # -------- Stage 3: Preprocessing Defense --------
    print("\n=== Stage 3: Same Model + Preprocessing Defense (Quantization) ===")
    _ = evaluate_fgsm(
        model, testloader, loss_fn, epsilons, use_preprocessing=True,
        title_prefix="[Preprocessing Defense]"
    )

    # -------- Stage 4: Adversarial Training --------
    print("\n=== Stage 4: Adversarial Training (FGSM-based) ===")
    adv_train_eps = 0.01
    for epoch in range(2):
        adversarial_train_epoch(model, trainloader, optimizer, loss_fn, epsilon=adv_train_eps)

    print("\n=== Evaluation on Clean Validation Data AFTER Adversarial Training ===")
    clean_acc_after = evaluate(model, testloader, loss_fn)

    print("\n=== FGSM Evaluation AFTER Adversarial Training ===")
    results_after = evaluate_fgsm(
        model, testloader, loss_fn, epsilons, use_preprocessing=False,
        title_prefix="[Adversarially Trained]"
    )

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