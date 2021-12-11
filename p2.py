#Libraries
import os
import sys
import math
import copy
from operator import *


#Part 2
def train(filepath):
	print("Training..")
	with open(filepath, "r", encoding="utf8") as f:
		lines = f.readlines()
		
	start = "START"
	stop = "STOP"

	# Set of all unique tokens in file
	tokens = []
	
	# Nested dictionary to keep track of emission count
	# {tag: {token: count} }
	emission_count = {} 

	# Iterate through file to update tokens and emission_count
	for line in lines:
		line_split = line.strip().rsplit(" ", 1)
		if len(line_split) == 2:
			token = line_split[0]
			tag = line_split[1]

			if token not in tokens:
				tokens.append(token)

			if tag not in emission_count:
				nested_tag_dict = {}
			else:
				nested_tag_dict = emission_count[tag]
			if token not in nested_tag_dict:
				nested_tag_dict[token] = 1
			else:
				nested_tag_dict[token] += 1
			emission_count[tag] = nested_tag_dict
			
	return tokens, emission_count


def est_emission_param(emission_count, token, tag, k=1):
	tag_dict = emission_count[tag]

	if token != "#UNK#":
		a = tag_dict.get(token, 0)
	else:
		a = k 
	b = sum(tag_dict.values()) + k

	return a / b



def transition(filepath):
	with open(filepath, "r", encoding="utf8") as f:
		lines = f.readlines()
		
	start = "START"
	stop = "STOP"

	# Set of all unique tokens in file
	tokens = []
	# Nested dictionary to keep track of emission count
	# {tag: {token: count} }
	u = start
	transition_count = {} 

	# Iterate through file to update tokens and transition_count
	for line in lines:
		line_split = line.strip().rsplit(" ", 1)
		
		#Case 1
		
		if len(line_split) == 2:
			token = line_split[0]
			v = line_split[1]

			if u not in transition_count:
				u_dict = {}
			else:
				u_dict = transition_count[u]

			if v in u_dict:
				u_dict[v] += 1
			else:
				u_dict[v] = 1

			transition_count[u] = u_dict
			u = v

		#Case 2
		
		else:
			u_dict = transition_count[u]
			v = stop

			if v in u_dict:
				u_dict[v] += 1
			else:
				u_dict[v] = 1

			transition_count[u] = u_dict
			u = start
	return transition_count




	
def transition_param(transition_count, u, v):
	

	if u not in transition_count:
		a = 0
	else:
		u_dict = transition_count[u]
		a = u_dict.get(v,0)
		b = sum(u_dict.values())

	return a / b
	

def viterbi_forward(emissions, transitions, words, sentence):
	n = len(sentence)
	smallest = -6969420
	
	states = list(transitions.keys())
	states.remove("START")

	# initialize score dict
	scores = {}

	scores[0] = {}

	for v in states:
		transition_fraction = transition_param(transitions, "START", v)
		if transition_fraction != 0:
			trans = math.log(transition_fraction)
		else:
			trans = smallest

		if sentence[0] not in words:
			token = "#UNK#"
		else:
			token = sentence[0]

		# Emission Probability
		if ((token in emissions[v]) or (token == "#UNK#")): 
			emmision_fraction = est_emission_param(emissions, token, v)
			emission = math.log(emmision_fraction)
		else:
			emission = smallest

		start = trans + emission
		scores[0][v] = ("START", start)
        
	# State 1 to n
	for i in range(1, n):

		scores[i] = {}
		for v in states:
			findmax = []

			for u in states:
				# Transition Probability
				transition_fraction = transition_param(transitions, u, v)

				if transition_fraction != 0:
					trans = math.log(transition_fraction)
				else:
					trans = smallest
			# if the word does not exist, assign special token
				if sentence[i] not in words:
					token = "#UNK#"
				else:
					token = sentence[i]

				# Emission Probability
				if ((token in emissions[v]) or token == "#UNK#"):
					emission_fraction = est_emission_param(emissions, token, v)
					emission = math.log(emission_fraction)
				else:
					emission = smallest
					
				current = scores[i-1][u][1] + trans + emission
				findmax.append(current)
    

            # ARGMAX
			ans = max(findmax)
			state_ans = states[findmax.index(ans)]
			scores[i][v] = (state_ans, ans)
			
	# STATE N to Stop State
	scores[n] = {}
	stopmax = []
	for u in states:
        # Transition Probability
		transition_fraction = transition_param(transitions, u, "STOP")
		if transition_fraction != 0:
			transition = math.log(transition_fraction)
		else:
			transition = smallest
        
		stopscore = scores[n-1][u][1] + transition
		stopmax.append(stopscore)
		
    
	# print(scores)
    # ARGMAX
	stop = max(stopmax)
	# print(stop)
	state_ans = states[stopmax.index(stop)]
	# print(states)
	# print(state_ans)
	scores[n] = (state_ans, stop)
	# print(scores)
	# Backtracking path
	path = ["STOP"]
	last = scores[n][0]
	path.insert(0, last)
	
	for k in range(n-1, -1, -1):
		last = scores[k][last][0]
		
		path.insert(0, last)
		# print(path)
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

		output_path = root_dir + "{}/dev.p2.out".format(dataset)
		# print(all_pred_tags)
		write_output(output_path, lines, all_pred_tags)
		
