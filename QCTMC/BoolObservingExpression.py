import pickle
import random
import re
import time
import sys
from QCTMC import qctmc_rho_t, isolate_solution, conflict_solution

sys.path.append('../')

if __name__ == '__main__':
    start = time.time()
    iso_time = 0
    conflict_time = 0
    instance = 5
    conflict_cache = 0
    iso_cache = 0
    rho_t, P, qctmc_time = qctmc_rho_t()
    count = 0

    con_iso = int(sys.argv[4])
    degree = int(sys.argv[1])
    height_from = int(sys.argv[2])
    height_to = int(sys.argv[3])
    # con_iso=1
    # degree=1
    # height_from=1
    # height_to=10
    Ileft = 1
    Iinter1 = 10
    Jinter1 = 5
    Jleft = 0.5
    # k = 3
    # var = 3
    # clause = 4
    # pickle_dump(k, var, clause, degree, height_from, height_to)
    exp_w = open(f'pickle_files/bool_exp_{degree}_{height_from}_{height_to}.pkl', 'rb')
    exps = pickle.load(exp_w)
    cnf_exps = pickle.load(exp_w)
    cnf_exp_lists = pickle.load(exp_w)
    Iinter_list = pickle.load(exp_w)
    Jinter_list = pickle.load(exp_w)
    boxleft_list = pickle.load(exp_w)
    dialeft_list = pickle.load(exp_w)
    exp_w.close()
    ww = open('./result.txt', 'a+')
    if con_iso:
        while count < instance:
            exp, cnf_exp, cnf_exp_list = exps[count], cnf_exps[count], cnf_exp_lists[count]
            pickle.dumps(exp)
            factor = exp[0].split(' ')
            Iinter = Iinter_list[count]
            Jinter = Jinter_list[count]
            boxleft = boxleft_list[count]
            dialeft = dialeft_list[count]
            I = [boxleft, boxleft + Iinter]
            J = [dialeft, dialeft + Jinter]
            inf_I = I[0]
            sup_I = I[1]
            inf_J = J[0]
            sup_J = J[1]
            B = [inf_I + inf_J, sup_I + sup_J]
            print(f'I: {I}, J: {J}')
            cnf_flag = 1
            ee_time = 0
            ee_cache = 0
            print(f'cnf_exp_list: {cnf_exp_list}')
            for ee in cnf_exp_list:
                if cnf_flag:
                    cnf_flag1 = 0
                    for e in ee:
                        f_e = e.split(' ')
                        if not cnf_flag1:
                            conflict_time1, conflict_cache1, cnf_flag1 = conflict_solution(f_e, rho_t, P, I, J,
                                                                                           qctmc_time)
                            conflict_time += conflict_time1
                            conflict_cache += conflict_cache1
                            ee_time += conflict_time1
                            ee_cache += conflict_cache1
                        else:
                            break
                    cnf_flag = cnf_flag1
                else:
                    print('The formula is not satisfied!')
                    break
            if cnf_flag:
                print('The boolean  formula is satisfied#')
            count += 1
            ww.write(f'{count+height_to} {ee_time:.2f} {ee_cache:.0f} {cnf_flag}\n')
            print(f'conflict__:{count + height_to} {ee_time:.2f} {ee_cache:.0f} {cnf_flag}\n')
            ww.flush()

        print(f'average time｜ isolate: {iso_time / instance}, conflict: {conflict_time / instance}')
        print(f'average space｜ isolate: {iso_cache / instance}, conflict: {conflict_cache / instance}')
    else:
        while count < instance:
            exp, cnf_exp, cnf_exp_list = exps[count], cnf_exps[count], cnf_exp_lists[count]
            pickle.dumps(exp)
            factor = exp[0].split(' ')
            Iinter = Iinter_list[count]
            Jinter = Jinter_list[count]
            boxleft = boxleft_list[count]
            dialeft = dialeft_list[count]
            I = [boxleft, boxleft + Iinter]
            J = [dialeft, dialeft + Jinter]
            inf_I = I[0]
            sup_I = I[1]
            inf_J = J[0]
            sup_J = J[1]
            B = [inf_I + inf_J, sup_I + sup_J]
            print(f'I: {I}, J: {J}')
            iso_flag = 1
            ee_time = 0
            ee_cache = 0
            print(f'cnf_exp_list: {cnf_exp_list}')
            for ee in cnf_exp_list:
                if iso_flag:
                    cnf_flag2 = 0
                    for e in ee:
                        f_e = e.split(' ')
                        if not cnf_flag2:
                            iso_time1, iso_cache1, value, solution = isolate_solution(I,J,f_e, rho_t, P, B, qctmc_time)
                            iso_time += iso_time1
                            iso_cache += iso_cache1
                            cnf_flag2 = value
                            ee_time += iso_time1
                            ee_cache += iso_cache1

                    iso_flag = cnf_flag2
                else:
                    print('The formula is not isolation satisfied!')
                    break
            if iso_flag:
                print('The boolean  formula is isolation satisfied#')
            count += 1
            ww.write(f'{count + height_to} {ee_time:.2f} {ee_cache:.0f} {iso_flag}\n')
            print(f'iso__: {count + height_to} {ee_time:.2f} {ee_cache:.0f} {iso_flag}\n')
            ww.flush()
        print(f'average time｜ isolate: {iso_time / instance}, conflict: {conflict_time / instance}')
        print(f'average space｜ isolate: {iso_cache / instance}, conflict: {conflict_cache / instance}')
    ww.close()
