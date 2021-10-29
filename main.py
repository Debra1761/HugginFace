from transformers import pipeline, AutoTokenizer, TFDistilBertModel

import torch
import torch.nn.functional as F

print(1)
model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
print(2)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', pad_to_max_length=True)
print(3)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
# classifier = pipeline("sentiment-analysis")

# result = classifier( "I am very happy learning huggingface transformers." )

# results = classifier(["I am very happy learning huggingface transformers." , 
#                         "I hate having to explain myself."])

# print(result)

for result in results:
    print(result)