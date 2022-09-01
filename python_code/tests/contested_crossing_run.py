
from python_code.contested_crossing import ContestedCrossing

ccross = ContestedCrossing()
cpoint=(1,3)

#check: right,down_right,down_left,left,up_left,up_right in various combinations gives initial point
assert(right_cell(down_right_cell(down_left_cell(left_cell(up_left_cell(up_right_cell(cpoint))))))==cpoint)
assert(left_cell(down_left_cell(down_right_cell(right_cell(up_right_cell(up_left_cell(cpoint))))))==cpoint)
assert(right_cell(up_right_cell(up_left_cell(left_cell(down_left_cell(down_right_cell(cpoint))))))==cpoint)
assert(left_cell(up_left_cell(up_right_cell(right_cell(down_right_cell(down_left_cell(cpoint))))))==cpoint)

#check: direction matches
assert(ccross._direction(cpoint,left_cell(cpoint))==W)
assert(ccross._direction(cpoint,up_left_cell(cpoint))==NW)
assert(ccross._direction(cpoint,up_right_cell(cpoint))==NE)
assert(ccross._direction(cpoint,right_cell(cpoint))==E)
assert(ccross._direction(cpoint,down_right_cell(cpoint))==NW)
assert(ccross._direction(cpoint,down_left_cell(cpoint))==NE)


