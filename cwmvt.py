#Implementation for the CWMVT method
#CWMVT - Confidence Ratio With Majority Voting and sum-of-scores Tiebreaker

#!pip install transformers sentencepiece datasets torch numpy pandas

import torch
import numpy as np
import pandas as pd
import json
import csv
from collections import Counter
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline


device = 0 if torch.cuda.is_available() else -1

#mBERT = "JCIFF/mBERT-qa-squad_v1-finetuned-FFI"
#XLMR = ""JCIFF/XLMR-qa-squad_v1-finetuned-FFI"

mT5 = "JCIFF/mT5-base-qa-squad_v2-finetuned-FFI"
mT0 = "JCIFF/mT0-base-qa-squad_v2-finetuned-FFI"
AfroXLMR = "ToluClassics/extractive_reader_afroxlmr_squad_v2"
mDeBERTa = "Jciff_files/finetuned/mDeberta_base"
RoBERTa = "deepset/roberta-base-squad2"

# Mixture-of-Experts setup.
MoE = [mT5, AfroXLMR, mT0, RoBERTa, mDeBERTa]

models = {}
tokenizers = {}

for model_name in MoE:
  models[model_name] = AutoModelForQuestionAnswering.from_pretrained(model_name)
  tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

# Create QA pipelines for each model
qa_pipelines = {}
for model_name in MoE:
  qa_pipelines[model_name] = pipeline("question-answering", model=models[model_name], tokenizer=tokenizers[model_name], device=device)


def normalize_text(s):
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return str(text).lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Gets predicted answer spans and corresponding confidence scores from all experts in setup
def get_answers(question, context):
  answers = {}
  for model_name in MoE:
    answer = qa_pipelines[model_name](question=question, context=context)
    answers[model_name] = (normalize_text(answer['answer']), answer['score'])
  return answers

# Function combines answers using majority voting
def combine_answers(answers):
  answer_counts = Counter(answer for answer, score in answers.values())
  most_common_answer = answer_counts.most_common(1)[0][0]

  # Calculate confidence scores
  total_votes = len(answers)
  confidence_ratio = answer_counts[most_common_answer] / total_votes

  return most_common_answer


def highest_Confidence_answers(answers):
  max_confidence = 0
  answer_with_highest_confidence = None
  for model_name, (answer, score) in answers.items():
    if score > max_confidence:
        max_confidence = score
        answer_with_highest_confidence = answer

  return answer_with_highest_confidence


def majority_voting(predictions):
    """
    Returns majority voted answer span.

    """
    prediction_counts = {}

    prediction_counts = Counter(predictions for predictions, confidences in answers.values())

    return prediction_counts


def get_best_answer(predictions, confidences):
    """
    Get the best answer based on majority voting and confidence ratio.

    Args:
    - predictions (list): A list of predictions from different models.
    - confidences (list): A list of confidences corresponding to each prediction.

    Returns:
    - The best answer based on the algorithm.
    """

    prediction_counts = Counter(predictions for predictions, confidences in answers.values())

    most_common_answer = prediction_counts.most_common(1)[0][0]

    maj_voting = majority_voting(predictions)

    ans_votes = list(maj_voting.values()).count(max(maj_voting.values()))

    if ans_votes == 1:
      best_answer = most_common_answer 
    elif ans_votes != 1:
      tied_predictions = [pred for pred, count in maj_voting.items() if count == max(maj_voting.values())]
      tied_confidence_sums = {}
      for pred in tied_predictions:
        tied_confidence_sums[pred] = sum(conf for p, conf in zip(predictions, confidences) if p == pred)
      best_answer = max(tied_confidence_sums, key=tied_confidence_sums.get)
      
    return best_answer


def model_predicted(answers):
  ans_k = []
  conf_k = []
  for model_name, (answer, score) in answers.items():
    ans_k.append(answer)
    conf_k.append(score)

  return ans_k, conf_k


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens, f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


data_reader = pd.read_json('./bemba/gold_span_passages.afriqa.bem.en.test.json', lines=True, encoding='UTF-8')


ans_in_eng = data_reader.iloc[:,0]          #ground truth, extracted for prediction accuracy validation
context = data_reader.iloc[:,1]             #context passage
qtn_in_afri_lang = data_reader.iloc[:,3]      


Ground_truth = []
for vals in ans_in_eng:
  ln = vals.get('text')
  ln = str(ln).strip("[']'")
  Ground_truth.append(ln)


MV_f1 = []
MV_em = []
CR_f1 = []
CR_em = []
CWMVA_em =[]
CWMVA_f1 =[]

counter = 0

for qtns, ctxt, Gtruth in zip(qtn_in_afri_lang, context, Ground_truth):
  counter +=1
  print("..........................................")
  print("Answering Question:: ", counter)
  if ctxt is not None:
    answers = get_answers(qtns, ctxt)

    # Compute Majority voting answer
    Maj_vote_ans = combine_answers(answers)

    #compute highest Confidence ratio answers
    high_Conf_Ratio_ans = highest_Confidence_answers(answers)

    #Determine CWMVT answers
    a, b =  model_predicted(answers)
    predict_ans = get_best_answer(a, b)

    MV_em_score = compute_exact_match(Gtruth, Maj_vote_ans)
    MV_f1_score = compute_f1(Gtruth, Maj_vote_ans)

    MV_em.append(MV_em_score)
    MV_f1.append(MV_f1_score)

    CR_em_score = compute_exact_match(Gtruth, high_Conf_Ratio_ans)
    CR_f1_score = compute_f1(Gtruth, high_Conf_Ratio_ans)

    CR_em.append(CR_em_score)
    CR_f1.append(CR_f1_score)

    em_score = compute_exact_match(Gtruth, predict_ans) 
    f1_score = compute_f1(Gtruth, predict_ans) 
    CWMVA_f1.append(f1_score)
    CWMVA_em.append(em_score)

    with open('./bem_prediction.csv', 'a', newline='') as result_file:
      obtained_ans = csv.writer(result_file)
      computed_result = [Maj_vote_ans, high_Conf_Ratio_ans, predict_ans, Gtruth]
      obtained_ans.writerow(computed_result)

  else:
    print("Data skipped")


MV_F1 = 100 * sum(MV_f1)/len(MV_f1)
MV_EM = 100 * sum(MV_em)/len(MV_em)
CR_F1 = 100 * sum(CR_f1)/len(CR_f1)
CR_EM = 100 * sum(CR_em)/len(CR_em)
CWMV_F1 = 100 * sum(CWMVA_f1)/len(CWMVA_f1)
CWMV_EM = 100 * sum(CWMVA_em)/len(CWMVA_em)


print("##############################################################")
print("               Results Saved to file                 ")
print("##############################################################")


print()
print("MV Overall F1 and EM")
print(f"F1: {MV_F1}     \t EM: {MV_EM}")
print()
print("CR Overall F1 and EM")
print(f"F1: {CR_F1}     \t EM: {CR_EM}")
print()
print("CWMVA Overall F1 and EM")
print(f"F1: {CWMV_F1}   \t EM: {CWMV_EM}")