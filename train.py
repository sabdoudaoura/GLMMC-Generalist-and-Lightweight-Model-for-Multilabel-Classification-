import torch 
from torch import nn
import yaml
from model import BiEncoderModel
from torch.optim import AdamW
from dataset import CustomDataset
from torch.utils.data import DataLoader

# Load the configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access parameters
data_path = config["data"]["synthetic_data_path"]
model_name = config["model"]["name"]
max_num_labels = config["model"]["max_num_labels"]
learning_rate = 0.0002#config["training"]["learning_rate"]
batch_size = config["training"]["batch_size"]
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10


# Custom collate function
def custom_collate_fn(batch):
    # Separate texts and labels from the batch
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return texts, labels



if __name__ == "__main__":



    model = BiEncoderModel(model_name, max_num_labels)
    criterion = nn.BCEWithLogitsLoss() # multiclass classification
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.to(device)


    # Training loop
    dataset = CustomDataset(data_path)
    # Dataloader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    
    # for batch_idx, (texts, batch_labels) in enumerate(train_loader):
    #     print(f"{len(texts)}, {len(batch_labels)}")


    
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for batch_idx, (texts, batch_labels) in enumerate(train_loader):
            print(len(texts))
            print(len(batch_labels))
            texts = [text for text in texts]  # Make sure text is in a list
            batch_labels = batch_labels  # Labels are in list of lists

            # # load to device (CPU ou GPU) # useless as gpu only uses numbers
            # texts = [text.to(device) for text in texts]
            # batch_labels = [label.to(device) for label in batch_labels]

            # Forward pass
            optimizer.zero_grad()
            scores, mask = model.forward(texts, batch_labels)  # Scores: [B, max_num_labels], mask: [B, max_num_labels]

            # Création de la target pour le calcul de la perte
            targets = torch.zeros_like(scores)
            for i, labels in enumerate(batch_labels):
                for j, label in enumerate(labels):
                    # Marquer la cible à 1 pour les labels présents
                    targets[i, j] = 1.0

            # Calcul de la perte
            loss = criterion(scores, targets)
            loss = (loss * mask).sum() / mask.sum()  # Appliquer le masque pour exclure les labels non présents dans chaque texte

            # Backward pass et mise à jour des poids
            loss.backward()
            optimizer.step()

            # Log de la perte
            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:  # Afficher les pertes toutes les 10 itérations
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / (batch_idx + 1):.4f}")

