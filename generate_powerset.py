#!/bin/python3

import sys
from itertools import chain, combinations

input_sequence = [3,4,5,6,7,8,9,10,11]
print(f"Preparing powerset of following channels: {input_sequence}")

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    tmp_list = list(iterable)
    return chain.from_iterable(combinations(tmp_list, index) for index in range(len(tmp_list)+1))

result = powerset(input_sequence)
result_list = list(result)
result_list_of_lists = [list(partial_result) for partial_result in result_list]

with open('channels.txt', 'w') as f:
    for partial_result_list in result_list_of_lists:
        if not partial_result_list:
            continue
        f.write(f"{str(partial_result_list).strip('[').strip(']')}")
        f.write("\n")
print("Done")
