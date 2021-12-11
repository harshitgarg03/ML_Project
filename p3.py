#Libraries
import os
import sys
import math
import copy
from operator import *


def train(filepath):
    with open(filepath, "r", encoding="utf8", errors='ignore') as f:
        lines = f.readlines()
        
    tokens3 = set()
    
    emission_count = {}
    
    for line in lines:
        line_split = line.strip().rsplit(" ", 1)  
        
        if len(line_split) == 2:
            token3 = line_split[0]
            tag = line_split[1]        
            tokens3.add(token3)
    
            
            if tag in emission_count:
               
                nested_tag_dict = emission_count[tag]
            else:
                
                nested_tag_dict = {}
                
            if token3 in nested_tag_dict:
                nested_tag_dict[token3] = nested_tag_dict[token3] + 1
            else:
                nested_tag_dict[token3] = 1
            
            emission_count[tag] = nested_tag_dict
    
    return tokens3, emission_count

def est_emission_param(emission_count, token, tag, k = 1):
    tag_dict = emission_count[tag]
    
    b = sum(tag_dict.values()) + k
    
    if token != "#UNK#":
        a = tag_dict[token]
    else: 
        a = k
    
    return a / b

def get_sentence_tag(sentence, tokens, emission_count, k=1):
	pred_tags = []

	for word in sentence:
		pred_tag = ""
		max_emission = float('-inf')

		for tag in emission_count:
			if word not in tokens:
				word = "#UNK#"

			if word in emission_count[tag] or word == "#UNK#":
				emission = est_emission_param(emission_count, word, tag, k)
				if emission > max_emission:
					pred_tag = tag 
					max_emission = emission

		pred_tags.append(pred_tag)

	return pred_tags
        

def transition(filepath):
    with open(filepath, "r", encoding="utf8", errors='ignore') as f:
        lines = f.readlines()
    
    start = "START"
    stop = "STOP"
    
    u = start
    
    transition_count = {}
    
    for line in lines:
        line_split = line.strip().rsplit(" ", 1)  
        
        # case 1
        if len(line_split) == 2:
            token3 = line_split[0]
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
            
        if len(line_split) != 2:
            u_dict = transition_count[u]
            v = stop
            if v in u_dict:
                u_dict[v] += 1
            else:
                u_dict[v] = 1
            transition_count[u] = u_dict
            
            u = start
            
            
    return transition_count

def transition_para(transition_count, u, v):
    if u not in transition_count:
        a = 0
        
    else:
        u_dict = transition_count[u]
    
        a = u_dict.get(v, 0)
    
        b = sum(u_dict.values())
        
    
    return a/b

def viterbi_forward(N, emissions, transitions, words, sentence):
    n = len(sentence)
    smallest = -9999999

    states = list(transitions.keys())
    states.remove("START")

    scores = {}
    scores[0] = {}

    for v in states:
        # Transition Probability
        transition_fraction = transition_para(transitions, "START", v)
        if transition_fraction != 0:
            trans = math.log(transition_fraction)
        else:
            trans = smallest
        
        if sentence[0] not in words:
            token3 = "#UNK#"
        else:
            token3 = sentence[0]

        # Emission Probability
        if ((token3 in emissions[v]) or (token3 == "#UNK#")): 
            emmision_fraction = est_emission_param(emissions, token3, v)
            emission = math.log(emmision_fraction)
        else:
            emission = smallest
        
        start = trans + emission
        scores[0][v] = ("START", start)

    copyscores = copy.deepcopy(scores)
    
    # State 1 to n
    for i in range(1, n):
        scores[i] = {}
        copyscores[i] = {}
        for v in states:
            findmax = []
            for u in states:
                # Transition Probability
                transition_fraction = transition_para(transitions, u, v)
                if transition_fraction != 0:
                    trans = math.log(transition_fraction)
                else:
                    trans = smallest
                if sentence[i] not in words:
                    v_v = "#UNK#"
                else:
                    v_v = sentence[i]

                # Emission Probability
                if ((v_v in emissions[v]) or (v_v == "#UNK#")): 
                    emmision_fraction = est_emission_param(emissions, v_v, v)
                    emission = math.log(emmision_fraction)
                else:
                    emission = smallest
              
                if i == 1 :
                  currentscore = scores[i-1][u][1] + trans + emission
                  findmax.append(currentscore)
                else:
                    #two nested for loops
                    # current score = [1st, 2nd , 3rd, 4th and 5th]
                  currentscores = [[scores[i-1][u][m][1] for m in range(N)][j] + trans + emission for j in range(N)]
                  for score in currentscores:
                    findmax.append(score)
            ans = [] 
            state_ans = []
            copyfindmax = copy.deepcopy(findmax)
            for m in range(N):
                ans.append(max(copyfindmax))
                # N=1 O or B postive, N=2 O, Positve, B neutral
                state_ans.append(states[findmax.index(ans[m]) // N])
                # print("NUMBER:::::", N)
                # print("THIS IS ANSWER::::::    ", ans[m])
                # print("PRINT INDEX       ", findmax.index(ans[m] // N) )
                copyfindmax[findmax.index(ans[m])] = -999999999.999
            scores[i][v] = tuple((state_ans[m], ans[m]) for m in range(N))
            

    # STOP STATE
    scores[n] = {}
    copyscores[n] = {}
    stopmax = []

    for u in states:
        # Transition Probability
        transition_fraction = transition_para(transitions, u, "STOP")
        if transition_fraction != 0:
            trans = math.log(transition_fraction)
        else:
            trans = smallest

        if(type(scores[n-1][u][0])==tuple):
            stopscore = [[scores[n-1][u][m][1] for m in range(N)][j] + trans + emission for j in range(N)]
        else:
            t=scores[n-1][u]
            stopscore = [t[1]+ trans + emission]    
            
        for score in stopscore:
            stopmax.append(score)
            

    stop = []
    state_ans = []
    copystopmax = copy.deepcopy(stopmax)
    for i in range(N):
        stop.append(max(copystopmax))
        state_ans.append(states[stopmax.index(stop[i]) // N])
        copystopmax[stopmax.index(stop[i])] = -999999999.999
    scores[n][u] = tuple((state_ans[m], stop[m]) for m in range(N))
    
      
    N_bestPaths = []
    lasts = [] 
    for i in range(N):
      path = ["STOP"]
      last = list(scores[n].values())[0][i][0]
      lasts.append(last)
      path.insert(0, last)
      N_bestPaths.append(path)
    
    for i in range(N):
        for k in range(n-1, -1, -1):
            if k == 0:
                last = scores[k][N_bestPaths[i][0]][0] 
            else:
                last = scores[k][N_bestPaths[i][0]][0][0]
            N_bestPaths[i].insert(0, last)
    
    
    return N_bestPaths[N-1]

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

        train_path = root_dir + "{}/train".format(dataset)
        evaluation_path = root_dir + "{}/dev.in".format(dataset)
        
        # training
        transition_count = transition(train_path)

        tokens3, emission_count = train(train_path)
        
        # evaluation
        with open(evaluation_path, "r", encoding='utf8') as f:
            lines = f.readlines()
        sentence = []
        
        all_pred_tags = []
        print(dataset)
        N = 5
        for line in lines:    
            if line != "\n":    
                line = line.strip()
                sentence.append(line)
            else:
                # print(emission_count)
                sentence_prediction = viterbi_forward(N, emission_count, transition_count, tokens3, sentence)
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

        output_path = root_dir + "{}/dev.p3.out".format(dataset)
        # print(all_pred_tags)
        write_output(output_path, lines, all_pred_tags)