import pickle
import random
import time
import sys
from ConflictDrivenSolving import Factor_Polynomial_By_X
from QCTMC import qctmc_rho_t, isolate_solution, conflict_solution
from random_expr import random_expression

sys.path.append('../')



def pickle_dump(k, var, clause, degree, height_from, height_to):
    exps = []
    cnf_exps = []
    cnf_exp_lists = []
    Iinter_list = []
    Jinter_list = []
    boxleft_list = []
    dialeft_list = []
    for i in range(5):
        exp, cnf_exp, cnf_exp_list = random_expression(k, var, clause, degree, height_from, height_to, 10)

        Iinter = random.randint(1, Iinter1)
        Jinter = random.randint(1, Jinter1)
        boxleft = round(random.uniform(0, Ileft), 1)
        dialeft = round(random.uniform(0, Jleft), 1)
        exps.append(exp)
        cnf_exps.append(cnf_exp)
        cnf_exp_lists.append(cnf_exp_list)
        Iinter_list.append(Iinter)
        Jinter_list.append(Jinter)
        boxleft_list.append(boxleft)
        dialeft_list.append(dialeft)

    exp_w = open(f'pickle_files/exp_{degree}_{height_from}_{height_to}.pkl', 'wb+')
    pickle.dump(exps, exp_w, True)
    pickle.dump(cnf_exps, exp_w, True)
    pickle.dump(cnf_exp_lists, exp_w, True)
    pickle.dump(Iinter_list, exp_w, True)
    pickle.dump(Jinter_list, exp_w, True)
    pickle.dump(boxleft_list, exp_w, True)
    pickle.dump(dialeft_list, exp_w, True)

    exp_w.close()
    print(f'exps: {exps}')
    print(f'cnf_exps: {cnf_exps}')
    print(f'cnf_exp_lists: {cnf_exp_lists}')

    print(f'Iinter_list: {Iinter_list}')
    print(f'Jinter_list: {Jinter_list}')
    print(f'boxleft_list: {boxleft_list}')
    print(f'dialeft_list: {dialeft_list}')


def random_observing_expressions(degree, height_from, height_to, is_bool):
    con_iso = 0
    start = time.time()
    iso_time = 0
    conflict_time = 0
    instance = 5
    conflict_cache = 0
    iso_cache = 0
    rho_t, P, qctmc_time = qctmc_rho_t(0)
    count = 0
    # root_count = random.randint(1, 3)
    root_count = 5
    # root_count = 2
    # print(f'root_count: {root_count}')
    degree = int(sys.argv[1])
    height_from = int(sys.argv[2])
    height_to = int(sys.argv[3])
    is_bool = int(sys.argv[4])
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
    while iso_count < instance:
        print(f'iso_count: {iso_count}, root_count: {root_count}')
        # k = 1
        # var = 1
        # clause =1
        k = random.randint(3, 4)
        var = random.randint(k, 8)
        clause = random.randint(1, 6)
        print(f'k:{k} var:{var} clause:{clause}')
        exp, cnf_exp, cnf_exp_list = random_expression(k, var, clause, degree, height_from, height_to, 10)
        Iinter = random.randint(1, Iinter1)
        Jinter = random.randint(1, Jinter1)
        boxleft = round(random.uniform(0, Ileft), 1)
        dialeft = round(random.uniform(0, Jleft), 1)
        factor = exp[0].split(' ')
        I = [boxleft, boxleft + Iinter]
        J = [dialeft, dialeft + Jinter]
        inf_I = I[0]
        sup_I = I[1]
        inf_J = J[0]
        sup_J = J[1]
        B = [inf_I + inf_J, sup_I + sup_J]
        print(f'I: {I}, J: {J}')
        iso_flag = False
        for ee in cnf_exp_list:
            cnf_flag = False
            for e in ee:
                f_e = e.split(' ')
                iso_time1, iso_cache1, value, solution = isolate_solution(I, J, f_e, rho_t, P, B, qctmc_time)
                if len(solution) > 0 or iso_count >= root_count:
                    cnf_flag = True
                    break
            if cnf_flag:
                iso_flag = True
                break

        if iso_flag:
            exps.append(exp)
            cnf_exps.append(cnf_exp)
            cnf_exp_lists.append(cnf_exp_list)
            Iinter_list.append(Iinter)
            Jinter_list.append(Jinter)
            boxleft_list.append(boxleft)
            dialeft_list.append(dialeft)
            k_list.append(k)
            var_list.append(var)
            clause_list.append(clause)
            iso_count += 1
    if is_bool:
        exp_w = open(f'pickle_files/bool_exp_{degree}_{height_from}_{height_to}.pkl', 'wb')
    else:
        exp_w = open(f'pickle_files/exp_{degree}_{height_from}_{height_to}.pkl', 'wb')
    pickle.dump(exps, exp_w, True)
    pickle.dump(cnf_exps, exp_w, True)
    pickle.dump(cnf_exp_lists, exp_w, True)
    pickle.dump(Iinter_list, exp_w, True)
    pickle.dump(Jinter_list, exp_w, True)
    pickle.dump(boxleft_list, exp_w, True)
    pickle.dump(dialeft_list, exp_w, True)
    # pickle.dump(k_list, exp_w, True)
    # pickle.dump(var_list, exp_w, True)
    # pickle.dump(clause_list, exp_w, True)
    exp_w.close()
    print(f'EXPS: {exps}')
    print(f'CNF_EXPS: {cnf_exps}')
    print(f'CNF_EXP_LISTS: {cnf_exp_lists}')

    print(f'Boxleft_list: {boxleft_list}')
    print(f'Dialeft_list: {dialeft_list}')
    print(f'Boxright_list: {Iinter_list}')
    print(f'Diaright_list: {Jinter_list}')


if __name__ == '__main__':
    con_iso = 0
    start = time.time()
    iso_time = 0
    conflict_time = 0
    instance = 5
    conflict_cache = 0
    iso_cache = 0
    rho_t, P, qctmc_time = qctmc_rho_t(0)
    count = 0
    # root_count = random.randint(1, 3)
    root_count = -1
    # root_count = 2
    # print(f'root_count: {root_count}')
    degree = int(sys.argv[1])
    height_from = int(sys.argv[2])
    height_to = int(sys.argv[3])
    is_bool = int(sys.argv[4])
    coe_from=1
    coe_to=height_to
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
    while iso_count < instance:
        print(f'iso_count: {iso_count}')
        if not is_bool:
            k = 1
            var = 1
            clause = 1
        else:
            k = random.randint(3, 4)
            var = random.randint(k, 5)
            clause = random.randint(1, 3)
        print(f'k:{k} var:{var} clause:{clause}')
        exp, cnf_exp, cnf_exp_list = random_expression(k, var, clause, degree, height_from, height_to,is_bool,coe_from,coe_to)

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
        print(f'I: {I}, J: {J}')
        iso_flag = True
        for ee in cnf_exp_list:
            cnf_flag = True
            for e in ee:
                factor=e.split(' ')
                fac = Factor_Polynomial_By_X(factor[0],P,rho_t)
                if len(fac) == 0:
                    cnf_flag = False
                    break
            if not cnf_flag:
                iso_flag = False
                break

        if iso_flag:
            exps.append(exp)
            cnf_exps.append(cnf_exp)
            cnf_exp_lists.append(cnf_exp_list)
            Iinter_list.append(Iinter)
            Jinter_list.append(Jinter)
            boxleft_list.append(boxleft)
            dialeft_list.append(dialeft)
            k_list.append(k)
            var_list.append(var)
            clause_list.append(clause)
            iso_count += 1
    if is_bool:
        exp_w = open(f'pickle_files/bool_exp_{degree}_{height_from}_{height_to}.pkl', 'wb')
    else:
        exp_w = open(f'pickle_files/exp_{degree}_{height_from}_{height_to}.pkl', 'wb')
    pickle.dump(exps, exp_w, True)
    pickle.dump(cnf_exps, exp_w, True)
    pickle.dump(cnf_exp_lists, exp_w, True)
    pickle.dump(Iinter_list, exp_w, True)
    pickle.dump(Jinter_list, exp_w, True)
    pickle.dump(boxleft_list, exp_w, True)
    pickle.dump(dialeft_list, exp_w, True)
    # pickle.dump(k_list, exp_w, True)
    # pickle.dump(var_list, exp_w, True)
    # pickle.dump(clause_list, exp_w, True)
    exp_w.close()
    print(f'EXPS: {exps}')
    print(f'CNF_EXPS: {cnf_exps}')
    print(f'CNF_EXP_LISTS: {cnf_exp_lists}')

    print(f'Boxleft_list: {boxleft_list}')
    print(f'Dialeft_list: {dialeft_list}')
    print(f'Boxright_list: {Iinter_list}')
    print(f'Diaright_list: {Jinter_list}')
