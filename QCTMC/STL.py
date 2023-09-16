import re
import turtle
from PIL import Image
import io, os

NOT = 1
AND = 2
UNTIL = 3
ATOMIC = 4
MAX_PARSE_NUM = 100

class STL:
    list_ind: int

    def __init__(self, STL_formula):
        '''
        :param STL_formula: string
        '''
        self.list_ind = -1
        self.parse_STL_tree = [None] * MAX_PARSE_NUM
        self.parse_STL_formula(STL_formula)
        self.parse_STL_tree = self.parse_STL_tree[0: self.list_ind + 1]
        self.tree_height = self.list_ind + 2


    def formula_type(self, formula):
        '''
        :param formula: list
        :return: map
        '''
        if formula[0] == '¬':
            return {'type': NOT, 'sub1': formula[1]}
        elif formula[0] == '∧':
            return {'type': AND, 'sub1': formula[1], 'sub2': formula[2]}
        elif re.match('U', formula[0]) is not None:
            return {'type': UNTIL, 'sub1': formula[1], 'sub2': formula[2], 'interval': eval(formula[0][1:])}
        else:
            return {'type': ATOMIC, 'sub1': formula[0]}

        # if re.search('¬', formula) is not None:
        #     return {'type': NOT, 'sub': formula.strip('¬')}
        # elif re.search('∧', formula) is not None:
        #     formula_split = formula.split('∧')
        #     return {'type': AND, 'sub1': formula_split[0], 'sub2': formula_split[1]}
        # elif re.search('U', formula) is not None:
        #     s1 = formula.split('U')
        #     reg = r"\[\d+,\d+\]"
        #     invl = re.findall(reg, s1[1])[0]
        #     interval = eval(invl)
        #     return {'type': UNTIL, 'sub1': s1[0], 'sub2': s1[1].replace(invl,''), 'invl': interval}
        # else:
        #     return {'type': ATOMIC, 'sub': formula}

    def list2string(self, lis):
        returnString = ''
        for item in lis:
            returnString += str(item)
        return returnString

    def parse_STL_formula(self, formula):
        self.list_ind += 1
        print("parsing....")
        print("formula:", formula)
        left_formula = None
        right_formula = None
        connect_symbol = None
        head_of_rightformula = 0

        if formula[0] != '(':
            # four cases:
            # 1. Phi
            # 2. ¬Phi | ¬(phi)
            # 3. Phi1 ∧ Phi2 | Phi1 ∧ (phi)
            # 4. Phi1 U[l,u] Phi2 | Phi1 U[l,u] (phi)
            if re.search(r"\(", formula) is None:
                if re.search('¬', formula) is not None:
                    parse_element = self.formula_type(['¬', self.list_ind + 1])
                    self.parse_STL_tree[self.list_ind] = parse_element
                    right_formula = formula[1:]
                    self.parse_STL_formula(right_formula)

                elif re.search('∧', formula) is not None:
                    parse_element = self.formula_type(['∧', self.list_ind + 1, self.list_ind + 2])
                    self.parse_STL_tree[self.list_ind] = parse_element
                    sym_position = re.search('∧', formula).span()[0]
                    left_formula = formula[0: sym_position]
                    self.parse_STL_formula(left_formula)
                    right_formula = formula[sym_position + 1:]
                    self.parse_STL_formula(right_formula)
                elif re.search(r"U\[\d+,\d+\]", formula) is not None:
                    sym_position = re.search(r"U\[\d+,\d+\]", formula).span()
                    parse_element = self.formula_type([formula[sym_position[0]: sym_position[1]], self.list_ind + 1, self.list_ind + 2])
                    self.parse_STL_tree[self.list_ind] = parse_element

                    left_formula = formula[0: sym_position[0]]
                    self.parse_STL_formula(left_formula)
                    right_formula = formula[sym_position[1]:]
                    self.parse_STL_formula(right_formula)
                else:
                    parse_element = self.formula_type([formula])
                    self.parse_STL_tree[self.list_ind] = parse_element
            else:
                lparant_position = re.search('\(', formula).span()[0]
                right_formula = formula[lparant_position + 1: -1]
                connect_symbol = formula[lparant_position - 1]
                if  connect_symbol == '¬':
                    parse_element = self.formula_type(['¬', self.list_ind + 1])
                    self.parse_STL_tree[self.list_ind] = parse_element
                    self.parse_STL_formula(right_formula)
                elif connect_symbol == '∧':
                    parse_element = self.formula_type(['∧', self.list_ind + 1, self.list_ind + 2])
                    self.parse_STL_tree[self.list_ind] = parse_element
                    left_formula = formula[0: lparant_position - 1]
                    self.parse_STL_formula(left_formula)
                    self.parse_STL_formula(right_formula)
                else:
                    sym_position = re.search(r"U\[\d+,\d+\]", formula).span()
                    parse_element = self.formula_type(
                        [formula[sym_position[0]: sym_position[1]], self.list_ind + 1, self.list_ind + 2])
                    self.parse_STL_tree[self.list_ind] = parse_element

                    left_formula = formula[0: sym_position[0]]
                    self.parse_STL_formula(left_formula)
                    right_formula = formula[sym_position[1] + 1: -1]
                    self.parse_STL_formula(right_formula)
        elif formula[0] == '(':
            match = 0
            right_bracket_position = 0
            for sym in formula:
                if sym == '(':
                    match += 1
                elif sym == ')':
                    match -= 1
                if match == 0:
                    break
                right_bracket_position += 1
            left_formula = formula[1: right_bracket_position]

            reg = r"∧|U\[\d+,\d+\]"
            try:
                connect_symbol = re.match(reg, formula[right_bracket_position + 1:]).group()
            except:
                raise Exception('some syntax errors in your STL formula, please CHECK!')
            connect_symbol_position = re.match(reg, formula[right_bracket_position + 1:]).span()
            head_of_rightformula = right_bracket_position + connect_symbol_position[1] + 1

            cur_formula_num = self.list_ind
            if formula[head_of_rightformula] == '(':

                right_formula = formula[head_of_rightformula + 1: -1]
                # self.list_ind += 1
                self.parse_STL_formula(left_formula)

                parse_element = self.formula_type([connect_symbol, cur_formula_num + 1, self.list_ind + 1])
                self.parse_STL_tree[cur_formula_num] = parse_element

                # self.list_ind += 1
                self.parse_STL_formula(right_formula)
            else:

                right_formula = formula[head_of_rightformula:]
                self.parse_STL_formula(left_formula)

                parse_element = self.formula_type([connect_symbol, cur_formula_num + 1, right_formula])

                self.parse_STL_tree[cur_formula_num] = parse_element
        else:
            raise Exception('some syntax errors in your STL formula, please CHECK!')

    def PostMonitoringPeriod(self, sub_formula) -> int:
        '''
        :param sub_formula: map -> {'type': ,'sub1': ,'sub2': , ...}
        :return: int
        '''
        if sub_formula['type'] == ATOMIC:
            return 0
        elif sub_formula['type'] == NOT:
            mnt = self.PostMonitoringPeriod(self.parse_STL_tree[sub_formula['sub1']])
            return mnt
        elif sub_formula['type'] == AND:
            mnt1 = self.PostMonitoringPeriod(self.parse_STL_tree[sub_formula['sub1']])
            mnt2 = self.PostMonitoringPeriod(self.parse_STL_tree[sub_formula['sub2']])
            return max(mnt1, mnt2)
        elif sub_formula['type'] == UNTIL:
            supInvl = sub_formula['interval'][1]
            mnt1 = self.PostMonitoringPeriod(self.parse_STL_tree[sub_formula['sub1']])
            mnt2 = self.PostMonitoringPeriod(self.parse_STL_tree[sub_formula['sub2']])
            return supInvl + max(mnt1, mnt2)


    def draw_parse_tree(self, file_name) -> None:
        # The function use turtle to draw a parse tree
        def jump_to(x, y):
            t.penup()
            t.goto(x, y)
            t.pendown()

        def draw(node_info, x, y, dx):
            if node_info['type'] == ATOMIC:
                t.goto(x, y)
                jump_to(x, y - 20)
                t.write(node_info['sub1'], align="center")
            elif node_info['type'] == NOT:
                t.goto(x, y)
                jump_to(x, y - 20)
                t.write('¬', align="center")
                draw(self.parse_STL_tree[node_info['sub1']], x, y - 60, dx / 2)

            else:
                t.goto(x, y)
                jump_to(x, y - 20)
                if node_info['type'] == UNTIL:
                    t.write("U" + str(node_info['interval']), align="center")
                elif node_info['type'] == AND:
                    t.write("∧", align="center")
                draw(self.parse_STL_tree[node_info['sub1']], x - dx, y - 60, dx / 2)
                jump_to(x, y - 20)
                draw(self.parse_STL_tree[node_info['sub2']], x + dx, y - 60, dx / 2)
        t = turtle.Turtle()
        t.hideturtle()
        t.speed(0)
        jump_to(0, 10 * self.tree_height)
        draw(self.parse_STL_tree[0], 0, 10 * self.tree_height, 20 * self.tree_height)
        # turtle.mainloop()

        ts = t.getscreen().getcanvas().postscript(file = file_name + ".eps", colormode = 'color')
        im = Image.open(file_name + ".eps")
        im.load(scale = 10)
        TARGET_BOUNDS = (1024, 1024)
        ratio = min(TARGET_BOUNDS[0] / im.size[0], TARGET_BOUNDS[1] / im.size[1])
        new_size = (int(im.size[0] * ratio), int(im.size[1] * ratio))
        im = im.resize(new_size, )
        im.save(file_name + ".png")
        os.remove(file_name + ".eps")
        # im.save(file_name + ".png", format = "PNG")
        # turtle.mainloop()


# stl = STL('((¬(True∧b))U[1,2](¬f))U[1,2](cU[2,3](d∧e))')
# stl = STL('aU[1,2]b')
stl = STL('¬(TrueU[0,5](a∧(¬(TrueU[0,1]b))))')
result = stl.parse_STL_tree
mnt = stl.PostMonitoringPeriod(result[0])
print(mnt)
print(result)
stl.draw_parse_tree("2")