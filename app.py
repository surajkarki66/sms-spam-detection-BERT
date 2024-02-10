import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Function to load the BERT model and tokenizer
@st.cache_data
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("surajkarki/bert_spam_detection")
    return tokenizer, model

@st.cache_data
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Main function to run the app
def main():
    st.header("SMS Spam Detection :sunglasses:", divider='rainbow')

    # Load the BERT model and tokenizer
    tokenizer, model = get_model()

    # Text area for user input
    user_input = st.text_area('Enter the SMS Text to Detect Spam')

    # Button to trigger the detection
    button = st.button("Detect")

    if user_input and button:
        # Tokenize the input text
        test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')

        # Get prediction from the model
        output = model(**test_sample)
        s_out = softmax(output.logits.detach().numpy())[0]
        y_pred = np.argmax(output.logits.detach().numpy(), axis=1)[0]

        # Display prediction result
        st.subheader("Prediction Result", divider="rainbow")
        if y_pred == 1:
            st.error(f"Spam :scream: : Be careful!, I am {round(s_out[y_pred]*100, 2)}% sure, this message is fake.")
        else:
            st.success(f"Ham :wink: : Chill out!, I am {round(s_out[y_pred]*100, 2)}% sure, this message is actually real.")

# Run the app
if __name__ == "__main__":
    main()
