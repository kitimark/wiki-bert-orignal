import json
import re
import tensorflow as tf


def load_predict_answers():
  with tf.gfile.Open('gs://bert-th/output-wikibot-3/predictions.json', 'r') as f:
    reader = json.load(f)

  clean_predicts = {}
  for key, value in reader.items():
    clean_predicts[int(key)] = re.sub(' ', '', value)

  return clean_predicts


def load_dev_answers():
  with tf.gfile.Open('gs://bert-th/wiki-dataset/wiki_dev.json', 'r') as f:
    reader = json.load(f)['data']

  answers = {}
  for line in reader:
    id = line['question_id']
    answer = line['answer']
    answers[int(id)] = re.sub(' ', '', answer)

  return answers


def evaluate(predict_answers, dev_answers):
  eval = {
    'correct': 0,
    'almost': 0,
    'dev_is_substring': 0,
    'predict_is_substring': 0,
    'incorrect': 0
  }
  for key in predict_answers:

    predict = predict_answers[key]
    dev = dev_answers[key]

    if predict == dev:
      eval['correct'] += 1
    else:
      if predict.find(dev) != -1 :
        eval['dev_is_substring'] += 1
      if dev.find(predict) != -1:
        eval['predict_is_substring'] += 1
      eval['incorrect'] += 1

  eval['almost'] = eval['dev_is_substring'] + eval['predict_is_substring']

  print("Summary")
  print("Correct ratio: %s" % (eval['correct'] / len(predict_answers)))
  print("Almost correct ratio: %s" % (eval['almost'] / len(predict_answers)))
  print("Dev is substring ratio: %s" % (eval['dev_is_substring'] / len(predict_answers))) 
  print("Predict is substring ratio: %s" % (eval['predict_is_substring'] / len(predict_answers))) 


if __name__ == "__main__":
  predict_answers = load_predict_answers()
  dev_answers = load_dev_answers()

  evaluate(predict_answers, dev_answers)
