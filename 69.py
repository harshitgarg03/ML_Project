root_dir = "./"
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

def est_emission_param(emission_count, token, tag, k = 1):
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

  u = 'START'
  v = 'START' 
  
  
  transition_tracker = {}
  

  for line in lines:
    line_split = line.strip().rsplit(" ", 1)
    
    # case 1: word line
    if len(line_split) == 2:
      token = line_split[0]
      w = line_split[1]

      if u not in transition_tracker:
        transition_tracker[u] = {}
      
      if v not in transition_tracker[u]:
        state_v_dict = {}
      else:
        state_v_dict = transition_tracker[u][v]
          
      if w in state_v_dict:
        state_v_dict[w] += 1
      else:
        state_v_dict[w] = 1

      transition_tracker[u][v] = state_v_dict

      u = v
      v = w

    if len(line_split) != 2:

      if u not in transition_tracker:
        transition_tracker[u] = {}
      
      if v not in transition_tracker[u]:
        state_v_dict = {}
      else:
        state_v_dict = transition_tracker[u][v]
      w = stop

      if w in state_v_dict:
        state_v_dict[w] += 1
      else:
        state_v_dict[w] = 1


      transition_tracker[u][v] = state_v_dict 

      u = start
      v = start

  return transition_tracker

def HMM2_transition_para(transition_tracker, u, v, w):
  
  if u not in transition_tracker:
    fraction = 0
  elif v not in transition_tracker[u]:
    fraction = 0
  else:
    state_v_dict = transition_tracker[u][v]

    numerator = state_v_dict.get(w, 0)

    denominator = sum(state_v_dict.values())
    fraction = numerator / denominator

  return fraction

def HMM2_viterbi(emission_dict, transition_dict, observations, sentence):
  n = len(sentence)
  smallest = -9999
  TOKEN = "#UNK#"

  states = list(transition_dict.keys())
  states.remove('START')
  
  scores = {}



  scores[0] = {}

  for w in states:
    trans_frac = HMM2_transition_para(transition_dict, 'START', 'START', w)

    if trans_frac != 0:
      trans = math.log(trans_frac)
    else:
      trans = smallest
    
    if sentence[0] not in observations:
      token = TOKEN
    else:
      token = sentence[0]

    if ((token in emission_dict[w]) or (token == TOKEN)):
      emis_frac = est_emission_param(emission_dict, token, w)
      emis = math.log(emis_frac)
    else:
      emis = smallest

    start_score = trans + emis
    scores[0][w] = {}
    scores[0][w]['START'] = {"START": start_score}

  if n == 1:
    scores[n] = {}
    scores[n]["STOP"] = {}

    scores[n]["STOP"]["START"] = {}
    for v in states:
      trans_frac = HMM2_transition_para(transition_dict, "START", v, "STOP")
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
        trans_frac = HMM2_transition_para(transition_dict, 'START', v, w)

        if trans_frac != 0:
          trans = math.log(trans_frac)
        else:
          trans = smallest
        
        if sentence[1] not in observations:
          token = TOKEN
        else:
          token = sentence[1]

        if ((token in emission_dict[w]) or (token == TOKEN)):
          emis_frac = est_emission_param(emission_dict, token, w)
          emis = math.log(emis_frac)
        else:
          emis = smallest

        current_score = scores[0][v]['START']['START'] + trans + emis
        scores[1][w]['START'][v] = current_score

  if n == 2:
    scores[2] = {}
    scores[2]["STOP"] = {}

    for u in states:
      scores[n]["STOP"][u] = {}
      for v in states:
        trans_frac = HMM2_transition_para(transition_dict, u, v, 'STOP')
        if trans_frac != 0:
          trans = math.log(trans_frac)
        else:
          trans = smallest

        state_v_arr = []
        
        for old_state_u in scores[n-1][v]:
          state_v_arr.append(scores[n-1][v][old_state_u][u])
        
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
          trans_frac = HMM2_transition_para(transition_dict, u, v, w)

          if trans_frac != 0:
            trans = math.log(trans_frac)
          else:
            trans = smallest
          
          if sentence[2] not in observations:
            token = TOKEN
          else:
            token = sentence[2]

          if ((token in emission_dict[w]) or (token == TOKEN)):
            emis_frac = est_emission_param(emission_dict, token, w)
            emis = math.log(emis_frac)
          else:
            emis = smallest

          current_score = scores[1][v]['START'][u] + trans + emis
          scores[2][w][u][v] = current_score


    for i in range(3,n):
      scores[i] = {}
      for w in states:
        scores[i][w] = {}
        for u in states:
          scores[i][w][u] = {}
          for v in states:
            # Transition Probability
            trans_frac = HMM2_transition_para(transition_dict, u, v, w)
            if trans_frac != 0:
              trans = math.log(trans_frac)
            else:
              trans = smallest

            if sentence[i] not in observations:
              token = TOKEN
            else:
              token = sentence[i]

            # Emission Probability
            if ((token in emission_dict[w]) or (token == TOKEN)):
              emis_frac = est_emission_param(emission_dict, token, w)
              emis = math.log(emis_frac)
            else:
              emis = smallest

            state_v_arr = []
            for old_state_u in scores[i-1][v]:
                state_v_arr.append(scores[i-1][v][old_state_u][u])
            best_score_v = max(state_v_arr)

            current_score = best_score_v + trans + emis
            scores[i][w][u][v] = current_score

    scores[n] = {}
    scores[n]["STOP"] = {}

    for u in states:
      scores[n]["STOP"][u] = {}
      for v in states:
        # Transition Probability
        trans_frac = HMM2_transition_para(transition_dict, u, v, 'STOP')
        if trans_frac != 0:
          trans = math.log(trans_frac)
        else:
          trans = smallest
        
        state_v_arr = []
        for old_state_u in scores[n-1][v]:
          state_v_arr.append(scores[n-1][v][old_state_u][u])
        
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


if __name__ == '__main__':
    design_dataset = ["EN"]

    for i in design_dataset:
        train = root_dir + "{folder}/train".format(folder = i)
        evaluation = root_dir + "{folder}/dev.in".format(folder = i)
        

        transition_tracker = transition(train)

        tokens, emission_count = train(train)
        

        with open(evaluation, "r", encoding="utf8") as f:

            lines = f.readlines()
        

        sentence = []
        

        all_prediction = []
        print(i)
        
        # each line is a word
        for line in lines:        
            if line != "\n":
                line = line.strip()
                sentence.append(line)
            else:
                sentence_prediction = HMM2_viterbi(emission_count, transition_tracker, tokens, sentence)
                sentence_prediction.remove("START")
                sentence_prediction.remove("STOP")
                all_prediction = all_prediction + sentence_prediction
                all_prediction = all_prediction + ["\n"]
                sentence = []
        
        assert len(lines) == len(all_prediction)
        # create output filepath
        with open(root_dir + "{folder}/dev.p5.out".format(folder = i), "w", encoding="utf8") as g:
            for j in range(len(lines)):
                word = lines[j].strip()
                if word != "\n":
                    tag = all_prediction[j]
                    if(tag != "\n"):
                        g.write(word + " " + tag)
                        g.write("\n")
                    else:
                        g.write("\n")
        
    print("done")
