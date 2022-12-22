from ast import main
from cProfile import label
from cgitb import text
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

def main():
    st.title('Sentiment Analyser App')
    form = st.form(key='sentiment-form')
    user_input = form.text_area('Enter your text')
    submit = form.form_submit_button('Submit')


    # Load model
    model = load_model('best_model(1).h5')
    tokenizer = Tokenizer(num_words=1000, lower=True, split=' ')

    def predict_class(text):

        '''Function to predict sentiment class of the passed text'''
        
        
        #sentiment_classes = ['Negative', 'Neutral', 'Positive']
        max_len=50
        
        # Transforms text to a sequence of integers using a tokenizer object
        tokenizer.fit_on_texts(text)
        xt = tokenizer.texts_to_sequences(text)
        # Pad sequences to the same length
        xt = pad_sequences(xt, padding='post', maxlen=max_len)
        # Do the prediction using the loaded model
        yt = model.predict(xt).argmax(axis=1)
        print(yt)
        # Print the predicted sentiment
        #print('The predicted sentiment is', sentiment_classes[yt[0]])
        if yt == 0:
            return 'Positive'
        elif yt == 1:
            return 'Negative'
        else:
            return 'Neutral'

    print(type(user_input))

    if submit:
        st.write(predict_class([user_input]))

if __name__=='__main__':
    main()

#predict_class(['good'])
#predict_class(['bad'])
#predict_class(["it's ok"])