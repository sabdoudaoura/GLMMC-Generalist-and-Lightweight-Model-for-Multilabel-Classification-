import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BiEncoderModel(nn.Module):
    def __init__(self, model_name, max_num_labels):
        super(BiEncoderModel, self).__init__()
        # Shared encoder for both text and labels
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_num_labels = max_num_labels  # Maximum labels per sample

    def encode(self, texts_or_labels):
        """
        Encodes a list of texts or labels using the shared encoder.
        """
        inputs = self.tokenizer(texts_or_labels, return_tensors='pt', padding=True, truncation=True)
        outputs = self.shared_encoder(**inputs)
        # mask aware pooling
        # last_hidden_state: [B, seq_len, D]
        att_mask = inputs['attention_mask'].unsqueeze(-1) ### Create a dimention [B, seq_len, 1] for easier broadcasting
        return (outputs.last_hidden_state * att_mask).sum(1) / att_mask.sum(1)  ### Global representation for label or sentence

    def forward(self, texts, batch_labels):
        """
        texts: List of input texts with batch size B
        batch_labels: List of lists containing labels for each text
        """
        B = len(texts)

        # Flatten all labels in the batch
        all_labels = [label for labels in batch_labels for label in labels] # batch labels [["label1", "label2"], ["label3", "label4", "label5"]]
        label_embeddings = self.encode(all_labels)  # Shape: [Num_unique_labels, D]

        # Encode texts
        text_embeddings = self.encode(texts)  # Shape: [B, D]

        # Prepare to recover batch structure
        label_counts = [len(labels) for labels in batch_labels]
        max_num_label = self.max_num_labels
        padded_label_embeddings = torch.zeros(B, max_num_label, label_embeddings.size(-1)) #tensor the stock the labels
        mask = torch.zeros(B, max_num_label, dtype=torch.bool)

        current = 0
        for i, count in enumerate(label_counts):
            if count > 0:
                end = current + count
                padded_label_embeddings[i, :count, :] = label_embeddings[current:end]
                mask[i, :count] = 1
                current = end

        # Compute similarity scores between text and each label
        # text_embeddings: [B, D]
        # padded_label_embeddings: [B, max_num_label, D]
        # scores: [B, max_num_label]
        scores = torch.bmm(padded_label_embeddings, text_embeddings.unsqueeze(2)).squeeze(2)
        scores = torch.sigmoid(scores)

        return scores, mask

    @torch.no_grad()
    def forward_predict(self, texts, labels):
        """
        texts: List of input texts
        labels: List of labels corresponding to the texts
        Returns:
            List of JSON objects with label scores for each text
        """
        scores, mask = self.forward(texts, labels)
        results = []
        for i, text in enumerate(texts):
            text_result = {}
            for j, label in enumerate(labels[i]):
                if mask[i, j]:
                    text_result[label] = float(f"{scores[i, j].item():.2f}")
            results.append({"text": text, "scores": text_result})
        return results

# Example Usage
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    max_num_labels = 5
    model = BiEncoderModel(model_name, max_num_labels)

    texts = ["I love machine learning.", "Deep learning models are powerful."]
    batch_labels = [
        ["AI", "Machine Learning", "building"],
        ["Deep Learning", "Neural Networks", "AI", "building"]
    ]

    # Forward pass
    scores, mask = model.forward(texts, batch_labels)
    print("Scores:", scores)
    print("Mask:", mask)

    # Prediction with JSON output
    # predictions = model.forward_predict(texts, batch_labels)
    # print("Predictions:", predictions)