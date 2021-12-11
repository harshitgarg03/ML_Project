#Libraries
import os
import sys
import math
import copy
from operator import *

def train(filepath):
    with open(filepath, "r", encoding="utf8", errors='ignore') as f:
        lines = f.readlines()
        
 
    tokens = set()
    
    emission_count = {}
    
    for line in lines:
        line_split = line.strip().rsplit(" ", 1)

        if len(line_split) == 2:
            token = line_split[0]
            tag = line_split[1]        
            tokens.add(token)
    
            if tag in emission_count:
                nested_tag_dict = emission_count[tag]
            else:
                nested_tag_dict = {}
                
            if token in nested_tag_dict:
                nested_tag_dict[token] = nested_tag_dict[token] + 1
            else:
                nested_tag_dict[token] = 1
            
            emission_count[tag] = nested_tag_dict
    
    return tokens, emission_count
#Laplace smoothing


# def est_emission_param(emission_count, token, tag, k=1):
# 	tag_dict = emission_count[tag]

# 	if token != "#UNK#":
# 		a = tag_dict.get(token, 0)
# 	else:
# 		a = k 
# 	sumz = 0
# 	for z in tag_dict.values():
# 		sumz += z + 1	
# 	# b = sum(tag_dict.values()) + k

# 	return a + 1 / (sumz + k)


# Bruteforce for the best K value 
def est_emission_param(emission_count, token, tag, k = 3):
    tag_dict = emission_count[tag]
    
    tag_dict = emission_count[tag]

    if token != "#UNK#":
        a = tag_dict.get(token, 0)
    else:
        a = k 
    b = sum(tag_dict.values()) + k

    return a / b

def transition(filepath):
  with open(filepath, 'r', encoding="utf8") as f:
    lines = f.readlines()

  start = 'START'
  stop = 'STOP'

  u = start
  v = start 
  
  
  transition_count = {}


  

  for line in lines:
    line_split = line.strip().rsplit(" ", 1)
    
    # case 1: word line
    if len(line_split) == 2:
      token = line_split[0]
      w = line_split[1]

      if u not in transition_count:
        transition_count[u] = {}
      
      if v not in transition_count[u]:
        v_dict = {}
      else:
        v_dict = transition_count[u][v]
          
      if w in v_dict:
        v_dict[w] += 1
      else:
        v_dict[w] = 1

      transition_count[u][v] = v_dict

      u = v
      v = w

    if len(line_split) != 2:

      if u not in transition_count:
        transition_count[u] = {}
      
      if v not in transition_count[u]:
        v_dict = {}
      else:
        v_dict = transition_count[u][v]
      w = stop

      if w in v_dict:
        v_dict[w] += 1
      else:
        v_dict[w] = 1


      transition_count[u][v] = v_dict 

      u = start
      v = start

  return transition_count



def transition_param(transition_count, u, v, w):
  
  if u not in transition_count:
    a = 0
    b = 1
  elif v not in transition_count[u]:
    a = 0
    b = 1
  else:
    v_dict = transition_count[u][v]

    a = v_dict.get(w, 0)

    b = sum(v_dict.values())
    
  return a / b

def viterbi_forward(emission_count, transition_count, tokens, sentence):
  n = len(sentence)
  smallest = -999999
  TOKEN = "#UNK#"

  states = list(transition_count.keys())
  states.remove('START')
  
  scores = {}



  scores[0] = {}

  for w in states:
    trans_frac = transition_param(transition_count, 'START', 'START', w)

    if trans_frac != 0:
      trans = math.log(trans_frac)
    else:
      trans = smallest
    
    if sentence[0] not in tokens:
      token = TOKEN
    else:
      token = sentence[0]

    if ((token in emission_count[w]) or (token == TOKEN)):
      emis_frac = est_emission_param(emission_count, token, w)
      emission = math.log(emis_frac)
    else:
      emission = smallest

    start_score = trans + emission
    scores[0][w] = {}
    scores[0][w]['START'] = {"START": start_score}

  if n == 1:
    scores[n] = {}
    scores[n]["STOP"] = {}

    scores[n]["STOP"]["START"] = {}
    for v in states:
      trans_frac = transition_param(transition_count, "START", v, "STOP")
      if trans_frac != 0:
        trans = math.log(trans_frac)
      else:
        trans = smallest
        
      best_score_v = scores[0][v]["START"]["START"]
      current_stop_score = best_score_v + trans
      scores[n]["STOP"]["START"][v] = current_stop_score
    
    path = ["STOP"]
    stop_lst = []

    for v in scores[n]["STOP"]["START"]:
      stop_lst.append((v, scores[n]["STOP"]["START"][v]))
    
    max_state_v_for_start_state = max(stop_lst, key=itemgetter(1))
    
    path.insert(0, max_state_v_for_start_state[0])
    path.insert(0, "START")

    return path

  else:
    scores[1] = {}

    for w in states:
      scores[1][w] = {}
      scores[1][w]["START"] = {}
      for v in states:
        trans_frac = transition_param(transition_count, 'START', v, w)

        if trans_frac != 0:
          trans = math.log(trans_frac)
        else:
          trans = smallest
        
        if sentence[1] not in tokens:
          token = TOKEN
        else:
          token = sentence[1]

        if ((token in emission_count[w]) or (token == TOKEN)):
          emis_frac = est_emission_param(emission_count, token, w)
          emission = math.log(emis_frac)
        else:
          emission = smallest

        current_score = scores[0][v]['START']['START'] + trans + emission
        scores[1][w]['START'][v] = current_score

  if n == 2:
    scores[2] = {}
    scores[2]["STOP"] = {}

    for u in states:
      scores[n]["STOP"][u] = {}
      for v in states:
        trans_frac = transition_param(transition_count, u, v, 'STOP')
        if trans_frac != 0:
          trans = math.log(trans_frac)
        else:
          trans = smallest

        state_v_arr = []
        
        for old_u in scores[n-1][v]:
          state_v_arr.append(scores[n-1][v][old_u][u])
        
        best_score_v = max(state_v_arr)
        current_stop_score = best_score_v + trans
        scores[n]["STOP"][u][v] = current_stop_score

    # Backtracking path
    path = ["STOP"]
    stop_lst = []

    for u in scores[n]["STOP"]:
      state_u_lst = []
      for v in scores[n]["STOP"][u]:
        state_u_lst.append((v, scores[n]["STOP"][u][v]))
      
        
      max_state_v_for_state_u = max(state_u_lst, key=itemgetter(1))
      
      max_tuple = (u, max_state_v_for_state_u[0], max_state_v_for_state_u[1])
      stop_lst.append(max_tuple)

    max_stop_tuple = max(stop_lst, key=itemgetter(2))
    
    path.insert(0, max_stop_tuple[1])
    path.insert(0, max_stop_tuple[0])

    prev = -2

    for k in range(n-1, 0, -1):
      u = scores[k][path[prev]] 
      state_v_lst = []
      
      for i in u.keys():
        if path[prev-1] in u[i]:
          state_v_lst.append((i, u[i][path[prev-1]]))
      
      max_score = max(state_v_lst, key=itemgetter(1))
      
      prev = prev - 1
      
      path.insert(0, max_score[0])
  
  elif n > 2:
    scores[2] = {}
    for w in states:
      scores[2][w] = {}
      for u in states:
        scores[2][w][u] = {}
        for v in states:
          # Transition Probability
          trans_frac = transition_param(transition_count, u, v, w)

          if trans_frac != 0:
            trans = math.log(trans_frac)
          else:
            trans = smallest
          
          if sentence[2] not in tokens:
            token = TOKEN
          else:
            token = sentence[2]

          if ((token in emission_count[w]) or (token == TOKEN)):
            emis_frac = est_emission_param(emission_count, token, w)
            emission = math.log(emis_frac)
          else:
            emission = smallest

          current_score = scores[1][v]['START'][u] + trans + emission
          scores[2][w][u][v] = current_score


    for i in range(3,n):
      scores[i] = {}
      for w in states:
        scores[i][w] = {}
        for u in states:
          scores[i][w][u] = {}
          for v in states:
            # Transition Probability
            trans_frac = transition_param(transition_count, u, v, w)
            if trans_frac != 0:
              trans = math.log(trans_frac)
            else:
              trans = smallest

            if sentence[i] not in tokens:
              token = TOKEN
            else:
              token = sentence[i]

            # Emission Probability
            if ((token in emission_count[w]) or (token == TOKEN)):
              emis_frac = est_emission_param(emission_count, token, w)
              emission = math.log(emis_frac)
            else:
              emission = smallest

            state_v_arr = []
            for old_u in scores[i-1][v]:
                state_v_arr.append(scores[i-1][v][old_u][u])
            best_score_v = max(state_v_arr)

            current_score = best_score_v + trans + emission
            scores[i][w][u][v] = current_score

    scores[n] = {}
    scores[n]["STOP"] = {}

    for u in states:
      scores[n]["STOP"][u] = {}
      for v in states:
        # Transition Probability
        trans_frac = transition_param(transition_count, u, v, 'STOP')
        if trans_frac != 0:
          trans = math.log(trans_frac)
        else:
          trans = smallest
        
        state_v_arr = []
        for old_u in scores[n-1][v]:
          state_v_arr.append(scores[n-1][v][old_u][u])
        
        best_score_v = max(state_v_arr)
        current_stop_score = best_score_v + trans
        scores[n]["STOP"][u][v] = current_stop_score


    # Backtracking path
    path = ["STOP"]
    stop_lst = []

    for u in scores[n]["STOP"]:
      state_u_lst = []
      for v in scores[n]["STOP"][u]:
        state_u_lst.append((v, scores[n]["STOP"][u][v]))
      
      max_state_v_for_state_u = max(state_u_lst, key=itemgetter(1))
      
      max_tuple = (u, max_state_v_for_state_u[0], max_state_v_for_state_u[1])
      stop_lst.append(max_tuple)
    
    max_stop_tuple = max(stop_lst, key=itemgetter(2))
    
    path.insert(0, max_stop_tuple[1])
    path.insert(0, max_stop_tuple[0])

    prev = -2

    for k in range(n-1, 0, -1):
      u = scores[k][path[prev]]
      state_v_lst = []


      for i in u.keys():
        if path[prev-1] in u[i]:
          state_v_lst.append((i, u[i][path[prev-1]]))
      

      max_score = max(state_v_lst, key=itemgetter(1))
      
      prev = prev - 1
      

      path.insert(0, max_score[0])


  return path



def write_output(filepath, lines, all_pred_tags):
	print("Writing output..")
	with open(filepath, "w", encoding="utf8") as f:
		for j in range(len(lines)):
			word = lines[j].strip()
			if word != "\n":
				# print(word)
				tag = all_pred_tags[j]
				# print(tag)
				if(tag != "\n"):
					f.write(word + " " + tag)
					f.write("\n")
				else:
					f.write("\n")
	print("Output successfully written!")

if __name__ == '__main__':
	root_dir = "./"
	datasets = ["ES", "RU"]

	for dataset in datasets:
		print("For dataset {}:".format(dataset))
		train_path = root_dir + "{}/train".format(dataset)
		evaluation_path = root_dir + "{}/dev.in".format(dataset)

		# Train
		transition_count = transition(train_path)
		tokens, emission_count = train(train_path)

		with open(evaluation_path, "r", encoding="utf8", errors='ignore') as f:
			lines = f.readlines()

		sentence = []
		all_pred_tags = []
		print(dataset)

		for line in lines:    
			if line != "\n":    
				line = line.strip()
				sentence.append(line)
			else:
				# print(emission_count)
				sentence_prediction = viterbi_forward(emission_count, transition_count, tokens, sentence)
				sentence_prediction.remove("START")
				sentence_prediction.remove("STOP")
				# print(sentence_prediction)
				all_pred_tags = all_pred_tags + sentence_prediction
				all_pred_tags = all_pred_tags + ["\n"]
				# print(all_pred_tags)
				sentence = []
		# print("length of lines:", len(lines),"lenth of all tags:", len(all_pred_tags))
		assert len(lines) == len(all_pred_tags)
		print("All words have a tag. Proceeding..")

		output_path = root_dir + "{}/dev.p4.out".format(dataset)
		# print(all_pred_tags)
		write_output(output_path, lines, all_pred_tags)
		