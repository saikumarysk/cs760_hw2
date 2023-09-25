import math

def shannon_entropy(prob):
    ent = 0
    
    for i in range(len(prob)):
        if prob[i] == 0: continue
        ent += prob[i] * math.log(1/prob[i], 2)
    
    return round(ent, 5)