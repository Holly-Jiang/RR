def read_file(path):
    f = open(path, 'r')
    line = f.readline().strip()
    res = []
    count = 1
    time_iso = 0
    time_sample = 0
    space_iso = 0
    space_sample = 0
    while line != '' and line != ' ':
        if line.startswith('AVERAGE') or line.startswith('**') or line.startswith('ISOLATION') or line.startswith('ID') or line.startswith('SAMPLE'):
            line = f.readline().strip()
            continue
        line1 = line
        l1 = line1.split(' ')
        line2 = f.readline().strip()
        l2 = line2.split(' ')
        line3 = f.readline().strip()
        l3 = line3.split(' ')
        line4 = f.readline().strip()
        l4 = line4.split(' ')
        line5 = f.readline().strip()
        l5 = line5.split(' ')
        f.readline().strip()
        f.readline().strip()
        f.readline().strip()
        f.readline().strip()
        line1a = f.readline().strip()
        l1.extend(line1a.split(' '))
        line2a = f.readline().strip()
        l2.extend(line2a.split(' '))
        line3a = f.readline().strip()
        l3.extend(line3a.split(' '))
        line4a = f.readline().strip()
        l4.extend(line4a.split(' '))
        line5a = f.readline().strip()
        l5.extend(line5a.split(' '))
        iso_flag = 'a' if int(l1[7]) else 'c'
        sample_flag='a' if int(l1[3]) else 'c'
        print(f'{count} {l1[1]} {float(l1[2])} {sample_flag} {float(l1[5]):.2f} {float(l1[6]):.2f} {iso_flag}')
        time_iso += round(float(l1[5]), 2)
        time_sample += round(float(l1[1]), 2)
        space_iso += int(l1[6])
        space_sample += int(l1[2])
        count += 1
        iso_flag = 'a' if int(l2[7]) else 'c'
        sample_flag='a' if int(l2[3]) else 'c'
        print(f'{count} {l2[1]} {float(l2[2])} {sample_flag} {float(l2[5]):.2f} {float(l2[6]):.2f} {iso_flag}')
        count += 1
        time_iso += round(float(l2[5]), 2)
        time_sample += round(float(l2[1]), 2)
        space_iso += int(l2[6])
        space_sample += int(l2[2])
        iso_flag = 'a' if int(l3[7]) else 'c'
        sample_flag='a' if int(l3[3]) else 'c'
        print(f'{count} {l3[1]} {float(l3[2])} {sample_flag} {float(l3[5]):.2f} {float(l3[6]):.2f} {iso_flag}')
        count += 1
        time_iso += round(float(l3[5]), 2)
        time_sample += round(float(l3[1]), 2)
        space_iso += int(l3[6])
        space_sample += int(l3[2])
        iso_flag = 'a' if int(l4[7]) else 'c'
        sample_flag='a' if int(l4[3]) else 'c'
        print(f'{count} {l4[1]} {float(l4[2])} {sample_flag} {float(l4[5]):.2f} {float(l4[6]):.2f} {iso_flag}')
        count += 1
        time_iso += round(float(l4[5]), 2)
        time_sample += round(float(l4[1]), 2)
        space_iso += int(l4[6])
        space_sample += int(l4[2])
        iso_flag = 'a' if int(l5[7]) else 'c'
        sample_flag='a' if int(l5[3]) else 'c'
        print(f'{count} {l5[1]} {float(l5[2])} {sample_flag} {float(l5[5]):.2f} {float(l5[6]):.2f} {iso_flag}')
        count += 1
        time_iso += round(float(l5[5]), 2)
        time_sample += round(float(l5[1]), 2)
        space_iso += int(l5[6])
        space_sample += int(l5[2])
        # aa='a' if int(l1[3]) else 'c'
        # # print(f'{count} {l1[1]} {float(l1[2])} {float(l1[5]):.2f} {float(l1[6]):.2f} {aa}')
        # print(f'\draw [blue2, dotted]({count},{float(l1[2])})--({count},{float(l1[6])});')
        # count += 1
        # aa='a' if int(l2[3]) else 'c'
        # print(f'\draw [blue2, dotted]({count},{float(l2[2])})--({count},{float(l2[6])});')
        # # print(f'{count} {l2[1]} {float(l2[2])} {float(l2[5]):.2f} {float(l2[6]):.2f} {aa}')
        # count += 1
        # aa='a' if int(l3[3]) else 'c'
        # print(f'\draw [blue2, dotted]({count},{float(l3[2])})--({count},{float(l3[6])});')
        # # print(f'{count} {l3[1]} {float(l3[2])} {float(l3[5]):.2f} {float(l3[6]):.2f} {aa}')
        # count += 1
        # aa='a' if int(l4[3]) else 'c'
        # print(f'\draw [blue2, dotted]({count},{float(l4[2])})--({count},{float(l4[6])});')
        # # print(f'{count} {l4[1]} {float(l4[2])} {float(l4[5]):.2f} {float(l4[6]):.2f} {aa}')
        # count += 1
        # aa='a' if int(l5[3]) else 'c'
        # print(f'\draw [blue2, dotted]({count},{float(l5[2])})--({count},{float(l5[6])});')
        # # print(f'{count} {l5[1]} {float(l5[2])} {float(l5[5]):.2f} {float(l5[6]):.2f} {aa}')
        # count += 1
        line = f.readline().strip()

    print(f' time: iso:{time_iso}, sample: {time_sample}')
    print(f'space: iso: {space_iso}, sample: {space_sample}')

def read_file1(path):
    f = open(path, 'r')
    line = f.readline().strip()
    res = []
    count = 1
    while line != '' and line != ' ':
        if line.startswith('AVERAGE') or line.startswith('****') or line.startswith('ISOLATE') or line.startswith(
                'ID') or line.startswith('CONFLICT'):
            line = f.readline().strip()
            continue
        line1 = line
        l1 = line1.split(' ')
        line2 = f.readline().strip()
        l2 = line2.split(' ')
        line3 = f.readline().strip()
        l3 = line3.split(' ')
        line4 = f.readline().strip()
        l4 = line4.split(' ')
        line5 = f.readline().strip()
        l5 = line5.split(' ')
        f.readline().strip()
        f.readline().strip()
        f.readline().strip()
        f.readline().strip()
        line1a = f.readline().strip()
        l1.extend(line1a.split(' '))
        line2a = f.readline().strip()
        l2.extend(line2a.split(' '))
        line3a = f.readline().strip()
        l3.extend(line3a.split(' '))
        line4a = f.readline().strip()
        l4.extend(line4a.split(' '))
        line5a = f.readline().strip()
        l5.extend(line5a.split(' '))

        aa = 'a' if int(l1[3]) else 'c'
        # print(f'{count} {l1[1]} {float(l1[2])} {float(l1[5]):.2f} {float(l1[6]):.2f} {aa}')
        print(f'\draw [blue2, dotted]({count},{float(l1[2])})--({count},{float(l1[6])});')
        count += 1
        aa = 'a' if int(l2[3]) else 'c'
        print(f'\draw [blue2, dotted]({count},{float(l2[2])})--({count},{float(l2[6])});')
        # print(f'{count} {l2[1]} {float(l2[2])} {float(l2[5]):.2f} {float(l2[6]):.2f} {aa}')
        count += 1
        aa = 'a' if int(l3[3]) else 'c'
        print(f'\draw [blue2, dotted]({count},{float(l3[2])})--({count},{float(l3[6])});')
        # print(f'{count} {l3[1]} {float(l3[2])} {float(l3[5]):.2f} {float(l3[6]):.2f} {aa}')
        count += 1
        aa = 'a' if int(l4[3]) else 'c'
        print(f'\draw [blue2, dotted]({count},{float(l4[2])})--({count},{float(l4[6])});')
        # print(f'{count} {l4[1]} {float(l4[2])} {float(l4[5]):.2f} {float(l4[6]):.2f} {aa}')
        count += 1
        aa = 'a' if int(l5[3]) else 'c'
        print(f'\draw [blue2, dotted]({count},{float(l5[2])})--({count},{float(l5[6])});')
        # print(f'{count} {l5[1]} {float(l5[2])} {float(l5[5]):.2f} {float(l5[6]):.2f} {aa}')
        count += 1
        line = f.readline().strip()


if __name__ == '__main__':
    read_file('./test.txt')
    print('------------------------')
    read_file1('./test.txt')
