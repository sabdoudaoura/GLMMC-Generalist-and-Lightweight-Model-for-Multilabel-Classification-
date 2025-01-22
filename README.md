# 🤖 GLMMC: Generalist and Lightweight Model for Multilabel Classification 

GLMMC is a Multilabel Classification Model capable of classifying texts into various predefined entities using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to Large Language Models (LLMs), which, despite their flexibility, are costly and too large for resource-constrained scenarios.

### Usage
```python

from model import BiEncoderModel

texts = ["A celebrity chef has opened a new restaurant specializing in vegan cuisine.", 
         "Doctors are warning about the rise in flu cases this season.", 
         "The United States has announced plans to build a wall on its border with Mexico."]
batch_labels = [
        
        ["Food", "Business", "Politics"],
        ["Health", "Food", "Public Health"],
        ["Immigration", "Religion", "National Security"]
    ]
# Load the model
model = BiEncoderModel("sharki007/bi-encoder-model", max_num_labels=6)
# Prediction with JSON output
predictions = model.forward_predict(texts, batch_labels)
print("Predictions:", predictions)

```


#### Expected Output

```

Predictions: [{'text': 'A celebrity chef has opened a new restaurant specializing in vegan cuisine.', 'scores': {'Food': 0.71, 'Business': 0.64, 'Politics': 0.41}}, 
{'text': 'Doctors are warning about the rise in flu cases this season.', 'scores': {'Health': 0.72, 'Food': 0.49, 'Public Health': 0.7}}, 
{'text': 'The United States has announced plans to build a wall on its border with Mexico.', 'scores': {'Immigration': 0.69, 'Religion': 0.33, 'National Security': 0.72}}]

```


## Author 🧑‍💻
- [Salim ABDOU DAOURA](https://github.com/sabdoudaoura)
