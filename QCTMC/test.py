


single_conflict_time = (2.40+1.77+2.56+1.91+2.09+2.08+2.10+2.39+2.31+2.74+2.18+2.24+2.45+10.70+4.38+3.29) / 16
single_conflict_spce = (114+107+110+108+109+108+110+111+111+110+110+111+113+118+117+115) / 16
bool_conflict_time = (14.50+15.14+15.08+16.20+15.62+17.22+18.86+14.73+24.61+21.09+11.44+16.08+22.57+19.65+20.93+11.07) / 16
bool_conflict_space = (748+946+650+835+899+645+627+680+910+902+616+728+1013+993+922+643) / 16

single_isolate_time = (14.65+53.88+51.33+54.55+25.35+10.09+22.91+32.06+12.66+6.25+6.46+25.28+8.88+10.32+2.93+4.66) / 16
single_isolate_space = (123+136+132+131+130+118+122+132+123+117+116+129+120+123+112+118) / 16
bool_isolate_time = (68.88+109.84+119.77+142.66+151.28+77.65+154.08+137.53+78.30+52.50+43.05+69.93+53.45+38.19+41.94+43.12) / 16
bool_isolate_space = (730+968+1106+1137+863+716+1019+1024+803+671+700+992+907+706+1056+835) / 16

print(f'single_conflict_time: {single_conflict_time}')
print(f'single_conflict_spce: {single_conflict_spce}')
print(f'bool_conflict_time: {bool_conflict_time}')
print(f'bool_conflict_space: {bool_conflict_space}')
print(f'single_isolate_time: {single_isolate_time}')
print(f'single_isolate_space: {single_isolate_space}')
print(f'bool_isolate_time: {bool_isolate_time}')
print(f'bool_isolate_space: {bool_isolate_space}')

print((single_conflict_time - single_isolate_time) / (single_isolate_time))
print((single_conflict_spce - single_isolate_space) / (single_isolate_space))
print((bool_conflict_time - bool_isolate_time) / (bool_isolate_time))
print((bool_conflict_space - bool_isolate_space) / (bool_isolate_space))


def read_file(path, str1, str2, index):
    f = open(path, 'r')
    line = f.readline().strip()
    count = '0'

    flag1 = False

    while line != '' and line != ' ':
        if line.startswith(str1):
            line = f.readline().strip()
            flag1 = True
            continue
        elif flag1 and line.startswith(str2):
            line = line.strip()
            line1 = line.split(' ')
            count += '+'
            if index == 2:
                count += line1[index].split(',')[0]
            elif index == 5:
                count += line1[index]
            flag1 = False
            line = f.readline().strip()
        else:
            line = f.readline().strip()
    print(count)
    return  count

path = './result.txt'
str1 = '**SINGLE SIGNAL** SAMPLE-DRIVEN:'
str2 = 'AVERAGE TIME:'
index=2
sct=read_file(path, str1, str2, index)
index=5
scs=read_file(path, str1, str2, index)

str1 = '**CNF OF MULTIPLE SIGNALS** SAMPLE-DRIVEN:'
index=2
cct=read_file(path, str1, str2, index)
index=5
ccs=read_file(path, str1, str2, index)

path = './result.txt'
str1 = '**SINGLE SIGNAL** ISOLATION-BASED:'
str2 = 'AVERAGE TIME:'
index=2
sit=read_file(path, str1, str2, index)
index=5
sis=read_file(path, str1, str2, index)

str1 = '**CNF OF MULTIPLE SIGNALS** ISOLATION-BASED:'
index=2
cit=read_file(path, str1, str2, index)
index=5
cis=read_file(path, str1, str2, index)


single_conflict_time = (eval(sct)) / 16
single_conflict_spce = (eval(scs)) / 16
bool_conflict_time = (eval(cct)) / 16
bool_conflict_space = (eval(ccs)) / 16

single_isolate_time = (eval(sit)) / 16
single_isolate_space = (eval(sis)) / 16
bool_isolate_time = (eval(cit)) / 16
bool_isolate_space = (eval(cis)) / 16

print(f'single_conflict_time: {single_conflict_time}')
print(f'single_conflict_spce: {single_conflict_spce}')
print(f'bool_conflict_time: {bool_conflict_time}')
print(f'bool_conflict_space: {bool_conflict_space}')
print(f'single_isolate_time: {single_isolate_time}')
print(f'single_isolate_space: {single_isolate_space}')
print(f'bool_isolate_time: {bool_isolate_time}')
print(f'bool_isolate_space: {bool_isolate_space}')

print((single_conflict_time - single_isolate_time) / (single_isolate_time))
print((single_conflict_spce - single_isolate_space) / (single_isolate_space))
print((bool_conflict_time - bool_isolate_time) / (bool_isolate_time))
print((bool_conflict_space - bool_isolate_space) / (bool_isolate_space))


