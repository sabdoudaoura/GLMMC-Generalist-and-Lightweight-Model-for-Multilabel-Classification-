import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiEncoderModel(nn.Module):
    def __init__(self, model_name, max_num_labels):
        super(BiEncoderModel, self).__init__()
        # Shared encoder for both text and labels
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_num_labels = max_num_labels  # Maximum labels per sample

    def encode(self, input_ids, attention_mask):
        """
        Encodes a list of texts or labels using the shared encoder.
        """
        outputs = self.shared_encoder(input_ids = input_ids, attention_mask = attention_mask)
        # mask aware pooling
        # last_hidden_state: [B, seq_len, D]
        att_mask = attention_mask.unsqueeze(-1) ### Create a dimention [B, seq_len, 1] for easier broadcasting

        return (outputs.last_hidden_state * att_mask).sum(1) / att_mask.sum(1) ### Global representation for label or sentence # We sum to get a unique embedding for each sentence

    def forward(self, input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, label_counts):
        """
        texts: List of input texts with batch size B 
        batch_labels: List of lists containing labels for each text
        """
        B = input_ids_text.shape[0]

        # Flatten all labels in the batch
        label_embeddings = self.encode(input_ids_labels, attention_mask_labels)  # Shape: [number_of_labels, D]

        # Encode texts
        text_embeddings = self.encode(input_ids_text, attention_mask_text)  # Shape: [B, D]

        # Prepare to recover batch structure
        max_num_label = self.max_num_labels
        padded_label_embeddings = torch.zeros(B, max_num_label, label_embeddings.size(-1)).to(device) #tensor to store the labels [B, max_num_label, 728]
        mask = torch.zeros(B, max_num_label, dtype=torch.bool).to(device) #[B, max_num_label] each element has 5 places for labels. Mask tell us how many labels are required for this element and their location ("1")

        #possibilité de parallelizer ? 
        current = 0
        for i, count in enumerate(label_counts):
            if count > 0:
                end = current + count
                padded_label_embeddings[i, :count, :] = label_embeddings[current:end] # copy count embeddings at the place of the ith element
                mask[i, :count] = 1
                current = end

        # Compute similarity scores between text and each label
        # Each sentence [B, D] -> [B, D, 1] times each label [B, max_num_label, D]
        # text_embeddings: [B, D]
        # padded_label_embeddings: [B, max_num_label, D]
        # scores: [B, max_num_label] # Each score is the similarity sentence and word
        scores = torch.bmm(padded_label_embeddings, text_embeddings.unsqueeze(2)).squeeze(2) 
        #scores = torch.sigmoid(scores) # loss function computes sigmoid #only if loss has no sigmoid

        return scores, mask

    @torch.no_grad()
    def forward_predict(self, texts, batch_labels):
        """
        texts: List of input texts
        labels: List of labels corresponding to the texts
        Returns:
            List of JSON objects with label scores for each text
        """

        texts_tokenized = [self.tokenizer(entry, return_tensors='pt', padding='max_length', truncation=True, max_length=15) for entry in texts]
        batch_labels_tokenized = [self.tokenizer(entry, return_tensors='pt', padding='max_length', truncation=True, max_length=5) for entry in batch_labels]

        input_ids_text = torch.stack([item['input_ids'] for item in texts_tokenized]).squeeze(1).to(device)
        attention_mask_text = torch.stack([item['attention_mask'] for item in texts_tokenized]).squeeze(1).to(device)


        input_ids_labels = torch.cat([item['input_ids'] for item in batch_labels_tokenized], dim=0).to(device)
        attention_mask_labels = torch.cat([item['attention_mask'] for item in batch_labels_tokenized], dim=0).to(device)
        
        label_counts = [len(labels) for labels in batch_labels]

        scores, mask = self.forward(input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, label_counts)
        scores = torch.sigmoid(scores)
        results = []
        for i, text in enumerate(texts):
            text_result = {}
            for j, label in enumerate(batch_labels[i]):
                if mask[i, j]:
                    text_result[label] = float(f"{scores[i, j].item():.2f}")
            results.append({"text": text, "scores": text_result})
        return results

# Example Usage
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    max_num_labels = 4
    model = BiEncoderModel(model_name, max_num_labels)

    texts = ["A celebrity chef has opened a new restaurant specializing in vegan cuisine.",
         "Doctors are warning about the rise in flu cases this season.",
         "The United States has announced plans to build a wall on its border with Mexico."]
    batch_labels = [
        ["Food", "Business", "Politics"],
        ["Health", "Food", "Public Health"],
        ["Immigration", "Religion", "National Security"]
    ]

    # Prediction with JSON output
    predictions = model.forward_predict(texts, batch_labels)
    print("Predictions:", predictions)