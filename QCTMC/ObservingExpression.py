import random
import time
import sys
from QCTMC import qctmc_rho_t, isolate_solution, conflict_solution
from ConflictDrivenSolving import Factor_Polynomial_By_X
from random_pickle import random_observing_expressions
from random_expr import random_expression

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

    degree = int(sys.argv[1])
    height_from = int(sys.argv[2])
    height_to = int(sys.argv[3])
    is_bool=int(sys.argv[4])
    Ileft = 1
    Iinter1 = 10
    Jinter1 = 5
    Jleft = 0.5
    iso_count = 0

    exps = []
    cnf_exps = []
    cnf_exp_lists = []
    Iinter_list = []
    Jinter_list = []
    boxleft_list = []
    dialeft_list = []
    k_list = []
    var_list = []
    clause_list = []

    ww = open('./result.txt', 'a+')
    if is_bool:
        ww.write('**CNF OF MULTIPLE SIGNALS** ')
    else:
        ww.write('**SINGLE SIGNAL** ')
    ww.write(f'DEGREE:{degree}, HEIGHT:{height_from}-{height_to} \nALGO: ID TIME SPACE SATISFIED\n')
    while count < instance:
        if is_bool:
            k = random.randint(3, 4)
            var = random.randint(k, 8)
            clause = random.randint(1, 6)
        else:
            k = 1
            var = 1
            clause =1
        print(f'K:{k} VAR:{var} CLAUSE:{clause}')
        exp, cnf_exp, cnf_exp_list = random_expression(k, var, clause, degree, height_from, height_to,is_bool,1,height_to)

        Iinter = random.randint(1, Iinter1)
        Jinter = random.randint(1, Jinter1)
        boxleft = round(random.uniform(0, Ileft), 1)
        dialeft = round(random.uniform(0, Jleft), 1)
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
        print('-------------------------SAMPLE-DRIVEN-----------------------------')

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
                print('SAMPLE-DRIVEN: The formula is not satisfied!')
                break
        if cnf_flag:
            print('SAMPLE-DRIVEN: The boolean formula is satisfied#')
        ww.write(f'SAMPLE-DRIVEN: {count+height_to+1} {ee_time:.2f} {ee_cache:.0f} {cnf_flag}\n')
        print(f'SAMPLE-DRIVEN:{count + height_to} {ee_time:.2f} {ee_cache:.0f} {cnf_flag}\n')
        ww.flush()
        print('-------------------------ISOLATION-BASED-----------------------------')
        iso_flag = 1
        ee_time = 0
        ee_cache = 0
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
        ww.write(f'ISOLATION-BASED: {count + height_to} {ee_time:.2f} {ee_cache:.0f} {iso_flag}\n')
        print(f'ISOLATION-BASED: {count + height_to} {ee_time:.2f} {ee_cache:.0f} {iso_flag}\n')
        if cnf_flag != iso_flag:
            print('--------error---------')
            ww.write(f'I: {I}, J: {J}')
            ww.write(f'cnf_exp_list: {cnf_exp_list}')
            ww.write('-----------error--------\n')
        ww.flush()
    ww.write(f'SAMPLE-DRIVEN AVERAGE TIME: {(conflict_time / instance):.2f}, AVERAGE SPACE: {(conflict_cache / instance):.0f}\n')
    ww.write(f'ISOLATION-BASED: AVERAGE TIME: {(iso_time / instance):.2f}, AVERAGE SPACE: {(iso_cache / instance):.0f}\n')
    ww.write('******************END**********************\n')
    print(f'SAMPLE-DRIVEN AVERAGE TIME: {(conflict_time / instance):.2f}, AVERAGE SPACE: {(conflict_cache / instance):.0f}\n')
    print(f'ISOLATION-BASED: AVERAGE TIME: {(iso_time / instance):.2f}, AVERAGE SPACE: {(iso_cache / instance):.0f}\n')

    ww.close()