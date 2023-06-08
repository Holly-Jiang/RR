import os


def SupplementSign(formula):
    fors = formula.split(' ')
    if fors[1] == '>':
        fors[1] = '<='
    elif fors[1] == '>=':
        fors[1] = '<'
    elif fors[1] == '<':
        fors[1] = '>='
    elif fors[1] == '<=':
        fors[1] = '>'
    return f"{fors[0]} {fors[1]} {fors[2]}"


def CNF_Observing(path, exps):
    f = open(path, 'r')
    line = f.readline().strip()
    start_flag = 0
    cnf_exp_list = []
    cnf_exp = ''
    cnf = ''
    while not (line == '' or line == ' ' or line == '\n'):
        if line.startswith('c') or line.startswith('p'):
            line = f.readline().strip()
            continue

        exp_list = []
        if start_flag:
            cnf_exp += 'and ('
            cnf += 'and ('
        else:
            cnf_exp += '('
            cnf += '('
        start_flag = 1
        lilist = line.split(' ')
        or_flag = 0
        for i in lilist:
            if or_flag and i != '0':
                cnf_exp += 'or '
                cnf += 'or '
            or_flag = 1
            if i == '0':
                continue
            elif i.startswith('-'):
                cnf_exp += f'not {exps[int(i[1]) - 1]} '
                cnf += f'not x{int(i[1]) - 1} '
                exp_list.append(f'{SupplementSign(exps[int(i[1]) - 1])}')
            else:
                cnf_exp += f'{exps[int(i[0]) - 1]} '
                cnf += f'x{int(i[0]) - 1} '
                exp_list.append(f'{exps[int(i[0]) - 1]}')
        cnf_exp += ') '
        cnf += ') '
        cnf_exp_list.append(exp_list)
        line = f.readline().strip()
    return cnf, cnf_exp, cnf_exp_list


def CNF_analysis(path):
    f = open(path, 'r')
    line = f.readline().strip()
    cnf = ''
    start_flag = 0
    while not (line == '' or line == ' ' or line == '\n'):
        if line.startswith('c') or line.startswith('p'):
            line = f.readline().strip()
            continue
        if start_flag:
            cnf += 'and ('
        else:
            cnf += '('
        start_flag = 1
        lilist = line.split(' ')
        or_flag = 0
        for i in lilist:
            if or_flag and i != '0':
                cnf += 'or '
            or_flag = 1
            if i == '0':
                continue
            elif i.startswith('-'):
                cnf += f'not x{i[1]} '
            else:
                cnf += f'x{i[0]} '
        cnf += ') '
        line = f.readline().strip()
    return cnf

# if __name__ == '__main__':
#     # sys.exit(main())
#     print(os.system('cnfgen randkcnf 3 3 5 > cnf.txt'))
#     print(CNF_analysis('./cnf.txt'))
