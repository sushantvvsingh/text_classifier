"""text classifier to classify a movie on the basis of preview given. Classifies it into five classes.
The dataset is taken from the Sentiment Analysis on Movie Reviews (Kernels only) task from Kaggle. The dataset consists of syntactic subphrases of the Rotten Tomatoes movie reviews.
"""
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

train_df, test_df = pd.read_csv("train.tsv", sep="\t"),pd.read_csv("test.tsv", sep="\t")
train_df.head()

# Training input.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], num_epochs=None, shuffle=True)

# Prediction on the whole training and testing set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], shuffle=False)
 

predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["Sentiment"], shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase", 
  module_spec="https://tfhub.dev/google/universal-sentence-encoder/2",
    
    trainable=True)

estimator = tf.estimator.DNNClassifier(
    hidden_units=[300, 70],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

estimator.train(input_fn=train_input_fn, steps=10000);

train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))


