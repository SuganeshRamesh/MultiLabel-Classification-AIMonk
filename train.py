import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from dataset import MultiLabelDataset
import matplotlib.pyplot as plt
import os
import numpy as np

def train_model():
    # Parameters
    IMG_DIR = 'images'
    LABELS_FILE = 'labels.txt'
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Augmentation & Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    full_dataset = MultiLabelDataset(LABELS_FILE, IMG_DIR, transform=transform)
    
    # Calculate pos_weight for imbalance handling
    # We count 1s as positive, 0s as negative, and skip -1 (NA)
    labels_df = full_dataset.df.iloc[:, 1:]
    pos_weights = []
    for col in labels_df.columns:
        num_pos = (labels_df[col] == 1).sum()
        num_neg = (labels_df[col] == 0).sum()
        # pos_weight = neg / pos
        weight = num_neg / (num_pos + 1e-8)
        pos_weights.append(weight)
    
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)
    print(f"Calculated pos_weights for imbalance: {pos_weights}")
    
    if torch.isnan(pos_weights).any():
        print("Error: NaN found in pos_weights. Checking label distribution...")
        # Fallback to ones if something is wrong
        pos_weights = torch.ones(4, dtype=torch.float32).to(DEVICE)


    # Simple Train/Val split (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model: Fine-tuning ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    # 4 output attributes
    model.fc = nn.Linear(num_ftrs, 4)
    model = model.to(DEVICE)

    # Custom Masked BCE Loss with pos_weight
    def masked_bce_loss(outputs, labels, pos_weight):
        # outputs: (batch, 4) logit
        # labels: (batch, 4) with values 0, 1, or -1 (NA)
        
        # Create mask for valid labels (>= 0)
        mask = (labels >= 0).float()
        
        # BCEWithLogitsLoss with per-label weights
        loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        raw_loss = loss_fn(outputs, labels.clamp(min=0)) 
        
        # Apply mask and average
        masked_loss = (raw_loss * mask).sum() / (mask.sum() + 1e-8)
        return masked_loss

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # For plotting
    iteration_losses = []
    
    # Training Loop
    model.train()
    iteration_number = 0
    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = masked_bce_loss(outputs, labels, pos_weights)
            loss.backward()
            optimizer.step()
            
            iteration_losses.append(loss.item())
            running_loss += loss.item()
            iteration_number += 1

            
            if torch.isnan(loss):
                print(f"Error: NaN loss at Epoch {epoch+1}, Step {i}")
                print(f"Outputs range: {outputs.min().item():.4f} to {outputs.max().item():.4f}")
                print(f"Labels summary: {labels.sum().item()}")
                # Optional: break if nan
                # break
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        
        print(f"Epoch {epoch+1} average loss: {running_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth")

    # Final plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(iteration_losses)), iteration_losses)
    plt.xlabel('iteration_number')
    plt.ylabel('training_loss')
    plt.title('Aimonk_multilabel_problem')
    plt.savefig('Aimonk_multilabel_problem.png')
    plt.show()
    print("Loss curve saved as Aimonk_multilabel_problem.png")

if __name__ == '__main__':
    train_model()
