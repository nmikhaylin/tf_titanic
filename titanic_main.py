import os
import re
import csv
import random

import numpy as np
import tensorflow as tf
import pandas as pd

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("summaries_dir", "summaries", "Directory for summaries.")
flags.DEFINE_integer("num_iterations", 10, "Number of iteration cycles.")
flags.DEFINE_integer("num_updates_per_iteration", 5,
                     "Number of weight updates per iteration.")
flags.DEFINE_float("validation_set_fraction", .1,
                   "Fraction of test set to leave for validation.")

DATA_FOLDER = "data"


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def ExtractCabinLetter(cabin_str):
  if cabin_str:
    match = re.match("[a-zA-Z]+", cabin_str)
    if match:
      return match.group(0)
  return ""

def ExtractCabinNumber(cabin_str):
  if cabin_str:
    match = re.search("[0-9]+", cabin_str)
    if match:
      return int(match.group(0))
  return -1


def LoadTestingData():
  raw_rows = []
  pclasses = set()
  genders = set()
  embarked = set()
  cabin_letters = set()
  cabin_numbers = set()
  df = pd.DataFrame(
      columns=["PassengerId", "Pclass", "Name", "Sex", "Age",
               "SibSp", "Parch", "Ticket", "Fare", "Cabin_Letter",
               "Cabin_Number", "Embarked"])
  with open(os.path.join(DATA_FOLDER, "titanic_test.csv"), "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
      raw_rows.append(row)
      if not row["Embarked"] in pclasses:
          embarked.add(row["Embarked"])
      if not row["Sex"] in pclasses:
          genders.add(row["Sex"])
      if not row["Pclass"] in pclasses:
          pclasses.add(row["Pclass"])
      cabin_letter = ExtractCabinLetter(row["Cabin"])
      if not cabin_letter in cabin_letters:
          cabin_letters.add(cabin_letter)
      row["Cabin_Letter"] = cabin_letter
      cabin_number = ExtractCabinNumber(row["Cabin"])
      if not cabin_number in cabin_numbers:
          cabin_numbers.add(cabin_number)
      row["Cabin_Number"] = cabin_number
      row["Parch"] = int(row["Parch"])
      row["Fare"] = float(row["Parch"])
      del row["Cabin"]
      df.loc[row["PassengerId"]] = pd.Series(row)
  return df


def LoadTrainingData():
  raw_rows = []
  pclasses = set()
  genders = set()
  embarked = set()
  cabin_letters = set()
  cabin_numbers = set()
  df = pd.DataFrame(
      columns=["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
               "SibSp", "Parch", "Ticket", "Fare", "Cabin_Letter",
               "Cabin_Number", "Embarked"])
  with open(os.path.join(DATA_FOLDER, "titanic_train.csv"), "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
      raw_rows.append(row)
      if not row["Embarked"] in pclasses:
          embarked.add(row["Embarked"])
      if not row["Sex"] in pclasses:
          genders.add(row["Sex"])
      if not row["Pclass"] in pclasses:
          pclasses.add(row["Pclass"])
      cabin_letter = ExtractCabinLetter(row["Cabin"])
      if not cabin_letter in cabin_letters:
          cabin_letters.add(cabin_letter)
      row["Cabin_Letter"] = cabin_letter
      cabin_number = ExtractCabinNumber(row["Cabin"])
      if not cabin_number in cabin_numbers:
          cabin_numbers.add(cabin_number)
      row["Cabin_Number"] = cabin_number
      row["Parch"] = int(row["Parch"])
      row["Fare"] = float(row["Parch"])
      del row["Cabin"]
      df.loc[row["PassengerId"]] = pd.Series(row)
  return df


def ProcessFeatures(raw_df):
  """Returns a dict for the fixed tensor columns."""
  ret_df = raw_df.drop("Name", 1)
  ret_df = ret_df.drop("Ticket", 1)
  ret_df = ret_df.drop("PassengerId", 1)
  remove_unknowns = lambda x:  float(x) if x else 30.0
  is_unknown = lambda x:  False if x else True
  ret_df["Age_Unknown"] = ret_df["Age"].apply(is_unknown).astype(np.float32)
  ret_df["Age"] = ret_df["Age"].apply(remove_unknowns)
  ret_df["Fare_Unknown"] = ret_df["Fare"].apply(is_unknown).astype(np.float32)
  ret_df["Fare"] = ret_df["Fare"].apply(remove_unknowns)
  int_columns = ["Parch", "SibSp"]
  for column in int_columns:
      ret_df[column] = ret_df[column].astype(np.int32)
  categorical_columns = [
      "Pclass", "Sex", "Cabin_Letter", "Embarked"]
  one_hotted = pd.get_dummies(
      ret_df, columns=categorical_columns).astype(np.float32)
  for col in (set([u'Cabin_Letter_',
                   u'Cabin_Letter_A', u'Cabin_Letter_B', u'Cabin_Letter_C',
                   u'Cabin_Letter_D', u'Cabin_Letter_E', u'Cabin_Letter_F',
                   u'Cabin_Letter_G', u'Embarked_C', u'Cabin_Letter_T',
                   u'Embarked_', u'Embarked_Q', u'Embarked_S'])
              - set(one_hotted.columns)):
      one_hotted[col] = 0.0
  one_hotted["Bias"] = 1.0
  one_hotted = one_hotted[list(sorted(one_hotted.columns))]
  # Remove unnecessary columns from the full matrix.
  columns_to_select = ["Sex_male", "Sex_female", "Age", "Age_Unknown",
                       "Fare", "Fare_Unknown", "Cabin_Number", "Pclass_1",
                       "Pclass_2", "Pclass_3", "Bias"]
  return one_hotted[columns_to_select].as_matrix()

def GenerateSampledTrainingAndValidationSets(all_features, all_labels):
  all_indx_set = set(range(all_labels.shape[0]))
  validation_indices = set(random.sample(all_indx_set,
    int(len(all_indx_set) * FLAGS.validation_set_fraction)))
  training_indices = all_indx_set - validation_indices
  validation_features = all_features[list(validation_indices),:]
  training_features = all_features[list(training_indices),:]
  training_labels = all_labels[list(training_indices),:]
  validation_labels = all_labels[list(validation_indices),:]
  return (training_features, training_labels, validation_features,
          validation_labels)




def MakePredictions(training_df, testing_df):
  with tf.Session() as sess:
    training_labels = training_df["Survived"].astype(np.float32)
    all_features = ProcessFeatures(training_df.drop("Survived", 1))
    testing_features = ProcessFeatures(testing_df)
    float_labels = np.expand_dims(training_labels, axis=1)
    print "Num survived: %f" % np.sum(float_labels)
    feature_size = all_features.shape[1]
    print "Number of features: %d" % feature_size
    input_features = tf.placeholder(tf.float32, shape=[None, feature_size])
    output_labels = tf.placeholder(tf.float32, shape=[None, 1])
    # Output is survived or not.
    W = tf.Variable(tf.zeros([feature_size, 1]))
    y = tf.matmul(input_features, W)
    pred = tf.cast(tf.cast(y > 0, tf.bool), tf.float32)
    add_delta = tf.matmul(tf.transpose(input_features), output_labels - pred)
    update_weights = W.assign_add(add_delta)
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, output_labels), tf.float32))
    training_accuracy = tf.summary.scalar("training_accuracy", accuracy)
    predicted_survived = tf.reduce_sum(pred)
    training_survived = tf.summary.scalar("predicted_survived", predicted_survived)
    validation_accuracy = tf.summary.scalar("validation_accuracy", accuracy)

    training_merged_summaries = tf.summary.merge(
        [training_accuracy, training_survived])
    validation_merged_summaries = tf.summary.merge(
        [validation_accuracy])
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/train", sess.graph)
    for i in range(FLAGS.num_iterations):
      (training_features, training_float_labels, validation_features,
          validation_float_labels) = GenerateSampledTrainingAndValidationSets(
              all_features, float_labels)
      train_summary = sess.run(training_merged_summaries, feed_dict={
        input_features: training_features,
        output_labels: training_float_labels})
      validation_summary = sess.run(validation_merged_summaries, feed_dict={
        input_features: validation_features,
        output_labels: validation_float_labels})
      train_writer.add_summary(train_summary, i)
      train_writer.add_summary(validation_summary, i)
      for j in range(FLAGS.num_updates_per_iteration):
        (training_features, training_float_labels, validation_features,
            validation_float_labels) = GenerateSampledTrainingAndValidationSets(
                all_features, float_labels)
        for ex in range(training_float_labels.shape[0]):
          sess.run(update_weights, feed_dict={
            input_features: training_features[ex:ex+1,:],
            output_labels: training_float_labels[ex:ex+1,:]})
    print "Final training accuracy: %f" % accuracy.eval(
        feed_dict={input_features: training_features,
                   output_labels: training_float_labels})
    print "Final validation accuracy: %f" % accuracy.eval(
        feed_dict={input_features: validation_features,
                   output_labels: validation_float_labels})
    ret_df = pd.DataFrame(data={"PassengerId": testing_df["PassengerId"],
                                "Survived": tf.squeeze(pred).eval(feed_dict={
                                  input_features: testing_features}).astype(
                                    np.int32)})

    return ret_df


if __name__ == "__main__":
  training_df = LoadTrainingData()
  testing_df = LoadTestingData()
  predictions = MakePredictions(training_df, testing_df)
  predictions.to_csv("results.csv", index=False)
