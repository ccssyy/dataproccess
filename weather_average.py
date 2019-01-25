import numpy as np
import math
import xlrd


def read_excel_file(inputfile):
    weather = xlrd.open_workbook(inputfile)
    sheet1_name = weather.sheet_names()[1]
    sheet_1 = weather.sheet_by_index(0)
    print(sheet_1.row_values(0)[1::])
    data = []
    for i in range(1, sheet_1.nrows - 1):
        data.append(sheet_1.row_values(i)[1::])
    # print(data)
    return data


def week_average(data):
    averange = []
    for i in range(0, len(data), 7):
        week_data = data[i:i + 7]
        averange.append(np.mean(week_data, 0))
        # print(week_data)
        # print('averange:{0}'.format(averange))
        # exit(0)
    np.savetxt('average_per_week.txt', averange, fmt='%.1f', newline='\n', encoding='utf-8')


if __name__ == '__main__':
    data = read_excel_file('weather.xlsx')
    week_average(data)
    # ff = xlrd.open_workbook('test.xlsx')
    # sheet_1 = ff.sheet_by_index(0)
    # print(sheet_1.row_values(0))
