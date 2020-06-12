import joblib

# load the pre-trained tf.Keras tokenizer
with open('pickles/text_tokenizer.pkl', 'rb') as f:
  tok = joblib.load(f)
  
# load the pre-trained scikit-learn LabelEncoder object
with open('pickles/labelencoder.pkl', 'rb') as f:
  labelencoder = joblib.load(f)
   
def predict_function(image_input, question_input, model, tokenizer, labelencoder):
    """
    This function include the entire pipeline, from taking raw data as input,
    data preprocessing and then making final predictions.

    Inputs:
        image_input    : List of image files
        question_input : List of raw question data
        model          : Keras model object
        tokenizer      : pre-trained tf.Keras tokenizer
        labelencoder   : pre-trained scikit-learn LabelEncoder object

    Returns:
        Predictions on the raw data
    """

    MAX_LEN = 22

    #1 --- Extract Image features
    print('1/4 Extracting Image Features')
    img_feat = image_feature_extractor(image_input)

    #2 --- Clean the questions.
    print('2/4 Cleaning the questions')
    questions_processed = pd.Series(question_input).apply(process_sentence)

    #3 --- Tokenize the question data using a pre-trained tokenizer and pad them
    print('3/4 Tokenizing and Padding the questions data')
    question_data = tok.texts_to_sequences(questions_processed)
    question_data = sequence.pad_sequences(question_data, \
                                           maxlen=MAX_LEN,\
                                           padding='post')


    #4 --- Predict the answers
    print('4/4 Predicting the answers')
    y_predict = predict_answers(img_feat, question_data, model, labelencoder)

    return y_predict