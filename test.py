import torch
from utils import * 
from models import *
from tqdm import tqdm
import argparse
import yaml
import os 
from utils import logg
from sklearn.metrics import classification_report


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(config: dict, device: torch.device, epoch : int):
    checkpoint_path = os.path.join(
        config['training']['save_path'],
        f"{config['model']['model_name']}_epoch_{epoch}"
    )
    print(f"Loading checkpoint from: {checkpoint_path}")
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch to load the model from")
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_cfg = config['training']
    test_dataset = HoaVNDataset(
        root_path=training_cfg['test_path'],
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg['num_workers']
    )

    model = load_model(config, device, args.epoch)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicts = torch.argmax(outputs, dim=1)
            all_preds.extend(predicts.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(classification_report(all_labels, all_preds, digits=4))



if __name__ == "__main__":
    main()
