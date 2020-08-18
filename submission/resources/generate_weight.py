#!/usr/bin/env python
import json

def gen_weight(input_file="model/train.txt", output_file="model/weight.json"):
    class_total = {}
    weight      = {}
    token_count = 0
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        tmp = line.strip().split()
        if tmp == []:
            pass
        elif tmp[1] not in class_total.keys():
            class_total[tmp[1]]  = 1
            token_count         += 1
        else:
            class_total[tmp[1]] += 1
            token_count         += 1
    
    for k,v in class_total.items():
        weight[k] = (v/token_count)**(-1)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(weight, f)


if __name__ == "__main__":
    gen_weight()
