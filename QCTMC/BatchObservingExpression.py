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
    rho_t, P, qctmc_time = qctmc_rho_t(0)
    count = 0

    con_iso = int(sys.argv[4])
    degree = int(sys.argv[1])
    height_from = int(sys.argv[2])
    height_to = int(sys.argv[3])
    is_bool = int(sys.argv[5])
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
    ww = open('./result.txt', 'a+')
    if is_bool:
        ww.write('**CNF OF MULTIPLE SIGNALS** ')
        exp_w = open(f'./paper_results/pickle_files/bool_exp_{degree}_{height_from}_{height_to}.pkl', 'rb')
    else:
        ww.write('**SINGLE SIGNAL** ')
        exp_w = open(f'./paper_results/pickle_files/exp_{degree}_{height_from}_{height_to}.pkl', 'rb')
    exps = pickle.load(exp_w)
    cnf_exps = pickle.load(exp_w)
    cnf_exp_lists = pickle.load(exp_w)
    Iinter_list = pickle.load(exp_w)
    Jinter_list = pickle.load(exp_w)
    boxleft_list = pickle.load(exp_w)
    dialeft_list = pickle.load(exp_w)
    exp_w.close()
    if con_iso:
        ww.write(f'SAMPLE-DRIVEN: DEGREE:{degree}, HEIGHT:{height_from}-{height_to} \nID TIME SPACE SATISFIED\n')
        while count < instance:
            exp, cnf_exp, cnf_exp_list = exps[count], cnf_exps[count], cnf_exp_lists[count]
            # pickle.dumps(exp)
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
            print(f'ID: {count + height_to+1}')
            print(f'I: {I}, J: {J}')
            cnf_flag = 1
            ee_time = 0
            ee_cache = 0
            print(f'CNF_EXP_LIST: {cnf_exp_list}')
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
            print(f'SAMPLE-DRIVEN: {count + height_to} {ee_time:.2f} {ee_cache:.0f} {cnf_flag}\n')
            ww.flush()
        ww.write(f'AVERAGE TIME: {(conflict_time / instance):.2f}, AVERAGE SPACE: {(conflict_cache / instance):.0f}\n')
        ww.write('******************END**********************\n')
        print(f'AVERAGE TIME: {(conflict_time / instance):.2f}, AVERAGE SPACE: {(conflict_cache / instance):.0f}')
    else:
        ww.write(f'ISOLATION-BASED: DEGREE:{degree}, HEIGHT:{height_from}-{height_to} \nID TIME SPACE SATISFIED\n')
        while count < instance:
            exp, cnf_exp, cnf_exp_list = exps[count], cnf_exps[count], cnf_exp_lists[count]
            # pickle.dumps(exp)
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

            print(f'ID: {count + height_to+1}')
            print(f'I: {I}, J: {J}')
            iso_flag = 1
            ee_time = 0
            ee_cache = 0
            print(f'CNF_EXP_LIST: {cnf_exp_list}')
            for ee in cnf_exp_list:
                if iso_flag:
                    cnf_flag2 = 0
                    for e in ee:
                        f_e = e.split(' ')
                        if not cnf_flag2:
                            iso_time1, iso_cache1, value, solution = isolate_solution(I, J, f_e, rho_t, P, B, qctmc_time)
                            iso_time += iso_time1
                            iso_cache += iso_cache1
                            cnf_flag2 = value
                            ee_time += iso_time1
                            ee_cache += iso_cache1

                    iso_flag = cnf_flag2
                else:
                    print('ISOLATION-BASED: The formula is not satisfied!')
                    break
            if iso_flag:
                print('ISOLATION-BASED: The boolean formula is satisfied#')
            count += 1
            ww.write(f'{count + height_to} {ee_time:.2f} {ee_cache:.0f} {iso_flag}\n')
            print(f'ISOLATION-BASED: {count + height_to} {ee_time:.2f} {ee_cache:.0f} {iso_flag}\n')
            ww.flush()
        ww.write(f'AVERAGE TIME: {(iso_time / instance):.2f}, AVERAGE SPACE: {(iso_cache / instance):.0f}\n')
        ww.write('******************END**********************\n')
        print(f'AVERAGE TIME: {(iso_time / instance):.2f}, AVERAGE SPACE: {(iso_cache / instance):.0f}')

    ww.close()
