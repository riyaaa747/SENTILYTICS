# # from transformers import BertTokenizer
# # from models.model import Transformer, build_transformer  # Changed TransformerClassifier to Transformer
# # from config import *
# # import torch

# # def predict_sentiment(text, model, tokenizer, device, max_len=128):
# #     # Prepare the text
# #     encoded = tokenizer.encode(
# #         text,
# #         max_length=max_len,
# #         padding="max_length",
# #         truncation=True,
# #         return_tensors="pt"
# #     )
    
# #     # Move to device
# #     encoded = encoded.to(device)
    
# #     # Get prediction
# #     model.eval()
# #     with torch.no_grad():
# #         # Forward pass
# #         encoder_output = model.encode(encoded, None)
# #         cls_output = encoder_output[:, 0, :]
# #         logits = model.project(cls_output)
        
# #         # Get probabilities
# #         probs = torch.softmax(logits, dim=1)
# #         prediction = torch.argmax(logits, dim=1)
        
# #         # Print debug info
# #         print(f"\nLogits: {logits.cpu().numpy()}")
# #         print(f"Probabilities: Negative={probs[0][0]:.3f}, Positive={probs[0][1]:.3f}, Neutral={probs[0.5][0.5]:.3f}")
        
# #     sentiment = {1.0: "Positive", 0.0: "Negative", 0.5: "Neutral"}.get(prediction.item(), "Unknown")
# #     return sentiment

# # # Setup
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # # Create and load model
# # model = build_transformer(
# #     SRC_VOCAB_SIZE,
# #     3,  # Binary classification
# #     SRC_SEQ_LEN,
# #     TGT_SEQ_LEN,
# #     D_MODEL,
# #     N_LAYERS,
# #     H,
# #     DROPOUT,
# #     D_FF
# # )
# # model.load_state_dict(torch.load(MODEL_PATH))
# # model.to(device)

# # # Test some reviews
# # test_reviews = [
# #     "This product is amazing but I am little bit disappointed with the packaging.",
# #     "Very disappointed with the quality.",
# #     "The person must be foolish who invests in this product.",
# #     "ok build quality, but could be better."
# # ]

# # print("\nPredicting sentiments:")
# # for review in test_reviews:
# #     print(f"\nReview: {review}")
# #     sentiment = predict_sentiment(review, model, tokenizer, device)
# #     print(f"Final Sentiment: {sentiment}")


# from transformers import BertTokenizer
# from models.model import build_transformer
# from config import *
# import torch

# def predict_sentiment(text, model, tokenizer, device, max_len=128):
#     # Tokenize
#     encoded = tokenizer.encode(
#         text,
#         max_length=max_len,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt"
#     ).to(device)

#     # Predict
#     model.eval()
#     with torch.no_grad():
#         encoder_output = model.encode(encoded, None)
#         cls_output = encoder_output[:, 0, :]
#         logits = model.project(cls_output)
        
#         probs = torch.softmax(logits, dim=1).squeeze(0)  # Now it's shape [3]

#         prediction = torch.argmax(logits, dim=1).item()

#         # Print detailed probabilities
#         print(f"\nLogits: {logits.cpu().numpy()}")
#         print(f"Probabilities: Negative={probs[0][0]:.3f}, Positive={probs[0][1]:.3f}, Neutral={probs[0][2]:.3f}")

#         # Convert to label
#         sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
#         sentiment = sentiment_map.get(prediction, "Unknown")
        
#         return sentiment, probs

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # Load model
# model = build_transformer(
#     SRC_VOCAB_SIZE,
#     3,  # 3-class classification
#     SRC_SEQ_LEN,
#     TGT_SEQ_LEN,
#     D_MODEL,
#     N_LAYERS,
#     H,
#     DROPOUT,
#     D_FF
# )
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device)

# # Run predictions
# test_reviews = [
#     "This product is amazing but I am little bit disappointed with the packaging.",
#     "This product is okay, neither great nor bad.",
#     "Product is very nice.",
#     "Rishav is very nice."
# ]

# print("\nPredicting sentiments:")
# for review in test_reviews:
#     print(f"\nReview: {review}")
#     sentiment = predict_sentiment(review, model, tokenizer, device)
#     print(f"Final Sentiment: {sentiment}")





from transformers import BertTokenizer
from models.model import build_transformer
from config import *
import torch

def predict_sentiment(text, model, tokenizer, device, max_len=128):
    # Tokenize
    encoded = tokenizer.encode(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(encoded, None)
        cls_output = encoder_output[:, 0, :]
        logits = model.project(cls_output)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1).squeeze(0)  # Now it's shape [3], 1D tensor

        prediction = torch.argmax(probs).item()  # Get the index of the max probability

        # Debug: Print probabilities directly as a 1D tensor
        print(f"Probabilities: Negative={probs[0]:.3f}, Positive={probs[1]:.3f}, Neutral={probs[2]:.3f}")

        # Convert to label
        sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
        sentiment = sentiment_map.get(prediction, "Unknown")
        
        return sentiment, probs

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model
model = build_transformer(
    SRC_VOCAB_SIZE,
    3,  # 3-class classification
    SRC_SEQ_LEN,
    TGT_SEQ_LEN,
    D_MODEL,
    N_LAYERS,
    H,
    DROPOUT,
    D_FF
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
