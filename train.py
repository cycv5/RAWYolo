import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import RawDataset, UNet, CombinedModel
from utils import prepare_batch
import os


def train_model(model, train_loader, val_loader, epochs=30):
    train_losses = []
    val_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # unfreeze YOLO parameters
    for param in model.yolo.parameters():
        param.requires_grad = True

    # Optimize all parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        # Training
        for param in model.yolo.parameters():
            param.requires_grad = True
        model.yolo.train()
        model.unet.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)

        for batch, (images, targets) in enumerate(train_progress):
            images = images.float().to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            batch = prepare_batch(images, targets)

            # Calculate loss
            loss, _ = model.yolo.loss(outputs, batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())

        # Validation
        model.yolo.eval()
        model.unet.eval()
        val_loss = 0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False)

        with torch.no_grad():
            for images, targets in val_progress:
                images = images.float().to(device)
                targets = targets.to(device)

                outputs = model(images)
                batch = prepare_batch(images, targets)
                loss, _ = model.yolo.loss(outputs, batch)
                val_loss += loss.item()
                val_progress.set_postfix(loss=loss.item())

        scheduler.step()
        print(
            f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}')
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"combined_model_{epoch+1}.pth")
    print('Finished Training')
    print("Training Loss:")
    print(train_losses)
    print("Validation Loss:")
    print(val_losses)
    return model


if __name__ == '__main__':
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model = CombinedModel()
    train_dataset = RawDataset(image_dir="dataset_split/train/images", annotation_dir="dataset_split/train/labels")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = RawDataset(image_dir="dataset_split/val/images", annotation_dir="dataset_split/val/labels")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    # model.load_state_dict(torch.load("combined_model.pth"))
    trained_model = train_model(model, train_loader, val_loader, epochs=30)
    torch.save(model.state_dict(), "combined_model_30.pth")

