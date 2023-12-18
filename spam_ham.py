import streamlit as st
import joblib


# Load the pre-trained model and the saved CountVectorizer
model = joblib.load('models/spam_ham.pkl')
featurizer = joblib.load('./models/count_vectorizer_featurizer.pkl')

# Streamlit app
def main():
    # Set the page title
    st.set_page_config(page_title="Spam Not-Spam")

    st.title("Spam or Not Spam Classifier")

    # Sidebar with navigation
    st.sidebar.title("Navigation")

    # Use Markdown to create a bullet-point list
    st.sidebar.markdown("* Home\n* Training Code")

    app_mode = st.sidebar.selectbox("Choose a page", ["Home", "Training Code"])

    if app_mode == "Home":
        show_home()
    elif app_mode == "Training Code":
        show_training_code()


# Function to display the home page
def show_home():
    # User input: text to be classified
    user_input = st.text_area("Enter text to classify:", "")

    # Make prediction when the user clicks the "Predict" button
    if st.button("Predict"):
        if user_input:
            prediction = predict_spam_ham(user_input)
            st.success(f"Prediction: {prediction}")
        else:
            st.warning("Please enter some text for prediction.")

# Function to preprocess and predict spam or ham
def predict_spam_ham(text):
    # Assuming your model expects a list of strings
    input_data = [text]

    # Apply the saved CountVectorizer
    processed_data = featurizer.transform(input_data)

    # Make predictions
    prediction = model.predict(processed_data)

    # Convert prediction to human-readable label
    label = "Spam" if prediction[0] == 1 else "Not Spam"

    return label

# Function to display the training code
def show_training_code():
    st.subheader("requirements.txt")
    st.code("""
scikit-learn
numpy
pandas
matplotlib
seaborn
wordcloud
            """)
    st.code("pip install -r requirements.txt")
    st.subheader("Download the dataset from kaggle")
    st.code("""
            => Link:  https://www.kaggle.com/uciml/sms-spam-collection-dataset
            """)
    st.subheader("Import Libraries")
    st.code("""
                import numpy as np 
    import pandas as pd 
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.naive_bayes import MultinomialNB
    from wordcloud import WordCloud
    """)
    st.subheader("Load Dataset and Data Preprocessing") 
    st.code("""
            df = pd.read_csv('./data/spam.csv', encoding='ISO-8859-1', usecols=['v1', 'v2'])
df.columns = ['labels', 'data']
df['b_labels'] = df['labels'].map({'ham':0, 'spam':1})
Y = df['b_labels'].to_numpy()
df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size=0.33)
featurizer = CountVectorizer(decode_error='ignore')
Xtrain = featurizer.fit_transform(df_train)
Xtest = featurizer.transform(df_test)
            """)
    st.subheader('Training the Model')
    st.code("""
            model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print('Train acc:', model.score(Xtrain, Ytrain))
print('Test acc:', model.score(Xtest, Ytest))
            """)
    st.write("""
Train acc: 0.9930350924189659\n
Test acc: 0.9836867862969005
             """)
    st.subheader('Checking the Results')
    st.code("""
            Ptrain = model.predict(Xtrain)
Ptest = model.predict(Xtest)
print('Train F1:', f1_score(Ytrain, Ptrain))
print('Test F1:', f1_score(Ytest, Ptest))
        """)
    st.write("""
             Train F1: 0.974459724950884\n
Test F1: 0.9339207048458149
             """)
    st.code("""
            Prob_train = model.predict_proba(Xtrain)[:,1]
Prob_test = model.predict_proba(Xtest)[:,1]

print('Train AUC:', roc_auc_score(Ytrain, Prob_train))
print('Test AUC:', roc_auc_score(Ytest, Prob_test))
            """)
    st.write("""
             Train AUC: 0.9936966533433107\n
Test AUC: 0.9777190431791484
             """)
    st.code("""
            cm = confusion_matrix(Ytrain, Ptrain)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
disp.plot();
            """)
    st.image("images/train.png")
    st.code("""
            cm = confusion_matrix(Ytest, Ptest)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
disp.plot();
            """)
    st.image("images/test.png")
    st.subheader("Visualize Important words")
    st.code("""
    def visualize(label):
        words=''
        for msg in df[df['labels'] == label]['data']:
            msg = msg.lower()
            words += msg + ' '
        wordcloud = WordCloud(width=600, height=400).generate(words)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()
            """)
    st.code("visualize('spam')")
    st.image("images/spam.png")
    st.code("visualize('ham')")
    st.image("images/ham.png")
    st.subheader('Save the Model and Transformer')
    st.code("""
            import joblib
model = MultinomialNB()
model.fit(X, Y)
joblib.dump(model, './models/spam_ham.pkl')
joblib.dump(featurizer, './models/count_vectorizer_featurizer.pkl')
            """)
    

if __name__ == "__main__":
    main()
