import torch
from torch import nn
import json
import yaml
from model import BiEncoderModel
from torch.optim import AdamW
from dataset import CustomDataset
from torch.utils.data import DataLoader

# Load the configuration file
with open("/content/drive/MyDrive/projet classification model*/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access parameters
data_path = config["data"]["synthetic_data_path"]
model_name = config["model"]["name"]
max_num_labels = config["model"]["max_num_labels"]
learning_rate = float(config["training"]["learning_rate"])
batch_size = config["training"]["batch_size"]

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = config["training"]["epochs"]


# Custom collate function
def custom_collate_fn(batch):
    # Separate texts and labels from the batch
    # Faire le padding ici
    texts = [item["text"] for item in batch]
    input_ids_text = torch.stack([item['input_ids'] for item in texts]).squeeze(1)
    attention_mask_text = torch.stack([item['attention_mask'] for item in texts]).squeeze(1)


    labels = [item["labels"] for item in batch]
    input_ids_labels = torch.cat([item['input_ids'] for item in labels], dim=0)
    attention_mask_labels = torch.cat([item['attention_mask'] for item in labels], dim=0)

    target_labels = [item["target_labels"] for item in batch]
    target_labels = torch.stack(target_labels, dim=0)


    num_labels = [item["dimension"] for item in batch]
    num_labels = torch.tensor(num_labels).to(device)
    
    return input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, target_labels, num_labels



if __name__ == "__main__":



    model = BiEncoderModel(model_name, max_num_labels).to(device)

    # #Freeze base model parameters
    # for param in model.parameters():
    #   param.requires_grad = False

    # Unfreeze the pooler layer to have a better representation for our classification task
    # for param in model.shared_encoder.pooler.parameters():
    #   param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss(reduction='none') # multiclass classification

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=learning_rate)

    with open(data_path, 'r', encoding='utf-8') as f:
      data = json.load(f)

    # Training loop
    dataset = CustomDataset(data, model_name, max_num_labels)

    # Dataloader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for batch_idx, (input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, target_labels, num_labels) in enumerate(train_loader):

            input_ids_text = input_ids_text.to(device)
            attention_mask_text = attention_mask_text.to(device)
            input_ids_labels = input_ids_labels.to(device)
            attention_mask_labels = attention_mask_labels.to(device)
            target_labels = target_labels.to(device)
            num_labels = num_labels.to(device)

            #Forward pass
            optimizer.zero_grad()
            scores, mask = model.forward(input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, num_labels)  # Scores: [B, max_num_labels], mask: [B, max_num_labels]

            # Compute the loss
            loss = criterion(scores, target_labels)
            #multiplier par mask
            loss = (loss * mask).sum()

            # Backward pass and update weights
            #loss.requires_grad = True # necessary if a part of the architecture is freezed
            loss.backward()
            optimizer.step()

            # Log de la perte
            running_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / (batch_idx + 1):.4f}")