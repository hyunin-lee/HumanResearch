import numpy as np


def reward_distortion(r_list: list,_lambda:float):
    beta1, beta2 = 0.9, 0.7
    # 0< beta1,beta2 <1
    # _lambda > 0
    u_r_list = []
    for r in r_list :
        if r > 0 :
            u_r = r ** beta1
        else :
            u_r = - _lambda * (-r) ** beta2
        u_r_list.append(u_r)
    return u_r_list

def p_normalization(probs) :
    total_sum = sum(probs)
    return  [i/total_sum for i in probs]
def prob_distortion(probs:list,gamma:float):
    w_p_list = []
    for p in probs:
        w_p = p**gamma / (p**gamma + (1-p)**gamma)**(1/gamma)
        w_p_list.append(w_p)
    w_p_list = p_normalization(w_p_list)
    return w_p_list