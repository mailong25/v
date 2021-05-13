import settings
import flask
import uuid, time
import json, io, time
import logging
import random
import os, gc
import yaml, numpy
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from sentence_transformers import CrossEncoder
from blender import Blender
import nltk
from sklearn.utils import shuffle
app = flask.Flask(__name__)

# os.environ['TRANSFORMERS_CACHE'] = '/home/ubuntu/disks/cache/'
# mname = 'facebook/blenderbot-1B-distill'
# model_temp = BlenderbotForConditionalGeneration.from_pretrained(mname)
# model = model_temp.to('cuda')
# tokenizer = BlenderbotTokenizer.from_pretrained(mname)
# del model_temp
# gc.collect()

PAR_PATH = "/home/ubuntu/resources/blender/ParlAI"
model = Blender(model_file = PAR_PATH + '/data/models/blender/blender_1Bdistill/model',
            parlai_home = PAR_PATH,include_personas = False)
print("* Model loaded")

model_sts = CrossEncoder('cross-encoder/stsb-roberta-large')
model2 = model_sts.model.half().to('cuda')
del model_sts.model
model_sts.model = model2
gc.collect()

#----------------------

skip_questions = yaml.load(open('skip.yaml'))
skip_questions = skip_questions['skip']

topics = yaml.load(open('topics.yaml'))
topic_questions = []
for topic in topics:
    topic_questions += [(topic,q) for q in topics[topic]]

suggestions = yaml.load(open('suggestions.yaml'))

#persona = 'your persona: I am twenty years old and I have a boyfriend    '
persona = ''

#-----------------

def extract_question(response):
    response = response.lower().replace("?",' ?').replace(",",' ,')
    response = ' '.join(response.split())
    response = response.replace("if you don't mind me asking ?"," ?")
    response = response.replace("if i may ask ?"," ?")
    response = response.replace("if you don't mind me asking ,","")
    response = response.replace("if i may ask ,","")
    response = response.replace("what about you ,","")
    response = response.replace("how about you ,","")
    response = response.replace("what about youself ,","")
    response = response.replace("how about yourself ,","")
    response = response.replace("well ,","")
    response = ' '.join(response.split())
    sents = nltk.sent_tokenize(response)
    questions = [q for q in sents if '?' in q if len(q) >= 3]
    return questions

def extract_non_repetitive_questions(questions):
    questions = [q for q in questions if not any(skip for skip in skip_questions if skip in q)]
    return questions

def extract_question_topic(question):
    question = question.lower().replace('?','')
    scores = list(zip([question] * len(topic_questions) , [q[1] for q in topic_questions]))
    scores = model_sts.predict(scores).tolist()
    idx_max = numpy.argmax(scores)
    if scores[idx_max] > 0.8:
        return topic_questions[idx_max][0]
    else:
        return None

def generate_reply(candidates, existing_questions, exists_topics):
    candidates = [c for c in candidates if '?' in c] + [c for c in candidates if '?' not in c]
    checked_candidates = []

    for candidate in candidates:
        #Check if there are any question
        print(candidate)
        candidate_question = extract_non_repetitive_questions(extract_question(candidate))

        if len(candidate_question) == 0:
            return candidate, None

        candidate_question = candidate_question[0]

        if candidate_question in checked_candidates:
            continue

        candidate_topic = extract_question_topic(candidate_question)

        if len(existing_questions) == 0:
            return candidate, candidate_topic

        #Compare with existing questions:
        pair_candidate_and_existing = zip([candidate_question] * len(existing_questions), existing_questions)
        scores = model_sts.predict(list(pair_candidate_and_existing))
        if max(scores) < 0.7 and candidate_topic not in exists_topics:
            return candidate, candidate_topic

        checked_candidates.append(candidate_question)

    best_possible_response = ''
    for candidate in candidates:
        sents = nltk.sent_tokenize(candidate)
        for i in range(0,len(sents)):
            if '?' in sents[i]:
                break
        possible_response = ' '.join(sents[:i])
        if len(possible_response) > len(best_possible_response):
            best_possible_response = possible_response
    
    if len(best_possible_response.split()) > 6:
        return best_possible_response, None
    
    print("Possible:", best_possible_response)
    # Not found any good candidate, all potential questions are existed, then switch topic
    for topic in suggestions:
        if topic not in exists_topics:
            return best_possible_response + ' ' + suggestions[topic][0], topic

    return "I have nothing more to say", None
    
@app.route("/")
def homepage():
    return "Welcome to the REST API!"

def huggingface_inference(turns):
    turns = turns[-6:]
    
    while True:
        input_text = persona + '    '.join(turns)
        inputs = tokenizer([input_text], return_tensors='pt',padding=True)
        if inputs['input_ids'].shape[1] < 125:
            break
        turns = turns[-(len(turns) -1):]

    inputs['attention_mask'] = inputs['attention_mask'].to('cuda')
    inputs['input_ids'] = inputs['input_ids'].to('cuda')

    reply_ids = model.generate(**inputs,num_beams=10, no_repeat_ngram_size=2,
                                num_return_sequences=10, early_stopping=True)

    candidates = tokenizer.batch_decode(reply_ids,skip_special_tokens = True)
    candidates = [candidates[0]] + shuffle(candidates[1:])
    return candidates[:5]

def parlai_inference(turns):
    turns = turns[-40:]
    turns = '\n'.join(turns)
    res = model.predict(turns)
    candidates = [c[0] for c in res['beam_texts']]
    candidates = [candidates[0]] + shuffle(candidates[1:])
    return candidates[:5]

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    
    dialog = json.loads(flask.request.json['dialog'])
    non_repeat = flask.request.json['non_repeat']
    
    turns     = [t['text'] for t in dialog]
    turns_bot = [t['text'] for t in dialog if t['spk'] == 'bot']
    exists_topics = [t['topic'] for t in dialog if t['topic'] != 'None']
    existing_questions = [extract_non_repetitive_questions(extract_question(t)) for t in turns_bot]
    existing_questions = [j for i in existing_questions for j in i]
    
    candidates = parlai_inference(turns)
    
    if non_repeat == True:
        response_text, response_topic = generate_reply(candidates, existing_questions, exists_topics)
    else:
        response_text = candidates[0]
        response_topic = None
    
    data["success"] = True
    data["text"] = response_text
    data["topic"] = str(response_topic)
    return flask.jsonify(data)

if __name__ == "__main__":
    print("* Starting web service...")
    app.run(host='127.0.0.1', port=5901,debug=False)
