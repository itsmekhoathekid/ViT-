import torch
from utils import *
from models import (
    Optimizer,
    ViT, 
    CEloss
)

from tqdm import tqdm
import argparse
import yaml
import os 
import logging
from speechbrain.nnet.schedulers import NoamScheduler

# Cấu hình logger


def reload_model(model, optimizer, checkpoint_path):
    """
    Reload model and optimizer state from a checkpoint.
    """
    past_epoch = 0
    path_list = [path for path in os.listdir(checkpoint_path)]
    if len(path_list) > 0:
        for path in path_list:
            if ".ckpt" not in path:
                past_epoch = max(int(path.split("_")[-1]), past_epoch)
        
        load_path = os.path.join(checkpoint_path, f"{model.model_name}_epoch_{past_epoch}")
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("No checkpoint found. Starting from scratch.")
    
    return past_epoch+1, model, optimizer


def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🔁 Training", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        img, label = batch
        img = img.to(device)
        label = label.to(device)


        optimizer.zero_grad()

        logits = model(img)  # [B, num_classes]
        
        loss = criterion(logits, label)
        loss.backward()

        optimizer.step()

        curr_lr, _ = scheduler(optimizer.optimizer)

        total_loss += loss.item()

        # === In loss từng batch ===
        progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Average training loss: {avg_loss:.4f}")
    return avg_loss, curr_lr


from torchaudio.functional import rnnt_loss

def evaluate(model, dataloader, optimizer, criterion, device):
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🧪 Evaluating", leave=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in progress_bar:
            img, label = batch
            img = img.to(device)
            label = label.to(device)


            optimizer.zero_grad()

            logits = model(img)  # [B, num_classes]
            
            loss = criterion(logits, label)
            correct += (logits.argmax(1) == label).sum().item()
            total += label.size(0)

            total_loss += loss.item()
            progress_bar.set_postfix(batch_loss=loss.item(), accuracy=100. * correct / total)


    avg_loss = total_loss / len(dataloader)
    logging.info(f"Average validation loss: {avg_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")
    return avg_loss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
def main():
    from torch.optim import Adam
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config['training']
    logg(training_cfg['logg'])

    
    # ==== Load Dataset ====
    train_dataset = HoaVNDataset(
        root_path=training_cfg['train_path'],
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg['num_workers']
    )

    dev_dataset = HoaVNDataset(
        root_path=training_cfg['dev_path'],
        transform=transform
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg['num_workers']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT(
        image_size = config['model']['image_size'],
        patch_size = config['model']['patch_size'],
        in_channels = config['model']['in_channels'],
        num_classes = config['model']['num_classes'],
        d_model = config['model']['d_model'],
        n_heads = config['model']['n_heads'],
        d_ff = config['model']['d_ff'],
        n_layers = config['model']['n_layers'],
        dropout = config['model']['dropout']
    ).to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # add_nan_hook(model)  # Thêm hook để kiểm tra NaN trong model

    # === Khởi tạo loss ===
    criterion = CEloss()

    optimizer = Optimizer(model.parameters(), config['optim'])



    if not config['training']['reload']:
        scheduler = NoamScheduler(
            n_warmup_steps=config['scheduler']['n_warmup_steps'],
            lr_initial=config['scheduler']['lr_initial']
        )
    else:
        scheduler = NoamScheduler(
            n_warmup_steps=config['scheduler']['n_warmup_steps'],
            lr_initial=config['scheduler']['lr_initial']
        )
        scheduler.load(config['training']['save_path'] + '/scheduler.ckpt')

    # === Huấn luyện ===

    start_epoch = 1
    if config['training']['reload']:
        checkpoint_path = config['training']['save_path']
        start_epoch, model, optimizer = reload_model(model, optimizer, checkpoint_path)
    num_epochs = config["training"]["epochs"]

    
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, curr_lr = train_one_epoch(model, train_loader, optimizer, criterion, device,  scheduler)
        val_loss = evaluate(model, dev_loader, optimizer, criterion , device)

        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {curr_lr:.6f}")
        # Save model checkpoint

        model_filename = os.path.join(
            config['training']['save_path'],
            f"{config['model']['model_name']}_epoch_{epoch}"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

        scheduler.save(config['training']['save_path'] + '/scheduler.ckpt')



if __name__ == "__main__":
    main()

# 3