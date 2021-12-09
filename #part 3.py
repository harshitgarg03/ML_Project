#part 3
def train(filepath):
    print("Training..")
    with open(filepath, "r", encoding="utf8", errors='ignore') as f:
        lines = f.readlines()

    start = "START"
    stop = "STOP"        
  
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

def est_emission_param(emission_count, token, tag, k = 0.5):
    
    tag_dict = emission_count[tag]
    
    b = sum(tag_dict.values()) + k
    
    
    if token != "#UNK#":
        a = tag_dict.get(token, 0)
    
    else: 
        a = k
    
    return a / b

def tag_producer(emission_count, labels, tokens3):
    tag_output = []
    
    for i in labels:
        predicted_state = ""
        highest_prob = -9999999.0
        
        for y in emission_count:
            if i not in tokens3:
                i = "#UNK#"
                
            if ((i in emission_count[y]) or (i == "#UNK#")):
                emission_prob = est_emission_param(emission_count, i, stayte_y, 1)

                if emission_prob > highest_prob:
                    highest_prob = emission_prob
                    predicted_state = y
                    
        tag_output.append(predicted_state)
    return tag_output


def transition(filepath):
    with open(filepath, "r", encoding="utf8") as f:
        lines = f.readlines()
    
    start = "START"
    stop = "STOP"
    
    
    u = start
    transition_count = {}
    
    for line in lines:
     
        split_line = line.strip()  
        split_line = split_line.rsplit(" ", 1) 
        
        # case 1
        if len(split_line) == 2:
            v = split_line[1]

          
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

def viterbi_forward(N, emissions, transitions, words, labels):
    n = len(labels)
    smallest = -99999999

    states = list(transitions.keys())
    states.remove("START")

    scores = {}

    scores[0] = {}

    for v in states:
    
        transition_fraction = transition_param(transitions, "START", v)
        if transition_fraction != 0:
            trans = math.log(transition_fraction)
        else:
            trans = smallest

        if labels[0] not in words:
            token3 = "#UNK#"
        else:
            token3 = labels[0]

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
                transition_fraction = transition_param(transitions, u, v)
                if transition_fraction != 0:
                    trans = math.log(transition_fraction)
                else:
                    trans = smallest
                
                if labels[i] not in words:
                    token3 = "#UNK#"
                else:
                    token3 = labels[i]

                if ((token3 in emissions[v]) or (token3 == "#UNK#")): 
                    emission_fraction = est_emission_param(emissions, token3, v)
                    emission = math.log(emission_fraction)
                else:
                    emission = smallest
              
                if i == 1 :
                  current = scores[i-1][u][1] + trans + emission
                  findmax.append(current)
                else:
                  current = [[scores[i-1][u][m][1] for m in range(N)][j] + trans + emission for j in range(N)] 
                  for s in current:
                    findmax.append(s)
            ans = [] 
            state_ans = []
            copyfindmax = copy.deepcopy(findmax)
            for m in range(N):
                ans.append(max(copyfindmax))
                state_ans.append(states[findmax.index(ans[m]) // N])
                copyfindmax[findmax.index(ans[m])] = -999999999.999
            
            scores[i][v] = tuple((state_ans[m], ans[m]) for m in range(N))
            

    # STOP STATE
    scores[n] = {}
    copyscores[n] = {}
    stopmax = []

    for u in states:
        
        transition_fraction = transition_param(transitions, u, "STOP")
        if transition_fraction != 0:
            transition = math.log(transition_fraction)
        else:
            transition = smallest

        if(type(scores[n-1][u][0])==tuple):
            stopscore = [[scores[n-1][u][m][1] for m in range(N)][j] + transition + emission for j in range(N)]
        else:
            t=scores[n-1][u]
            stopscore = [t[1]+ transition + emission]    
            
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

if __name__ == '__main__':
    root_dir = "./"
    datasets = ["ES", "RU"]

    for dataset in datasets:

        print("For dataset {}:".format(dataset))
        train_path = root_dir + "{}/train".format(dataset)
        evaluation_path = root_dir + "{}/dev.in".format(dataset)
        
        # training
        transition_count = transition(train_path)

        tokens3, emission_count = train(train_path)
        
        # evaluation
        with open(evaluation_path, "r", encoding='utf8') as f:
            
            lines = f.readlines()
        
        
        labels = []
        
        all_pred_tags = []
        print(dataset)
    
        N = 5
        
        for line in lines:        
            if line != "\n":
                line = line.strip()
                labels.append(line)
            else:
                sentence_prediction = viterbi_forward(N, emission_count, transition_count, tokens3, labels)
                sentence_prediction.remove("START")
                sentence_prediction.remove("STOP")
                all_pred_tags = all_pred_tags + sentence_prediction
                all_pred_tags = all_pred_tags + ["\n"]
                labels = []
        
        assert len(lines) == len(all_pred_tags)
       
        with open(root_dir + "{folder}/dev.p4.out".format(folder = dataset), "w", encoding='utf8') as g:
            for j in range(len(lines)):
                word = lines[j].strip()
                if word != "\n":
                    tag = all_pred_tags[j]
                    if(tag != "\n"):
                        g.write(word + " " + tag)
                        g.write("\n")
                    else:
                        g.write("\n")

    print("done")