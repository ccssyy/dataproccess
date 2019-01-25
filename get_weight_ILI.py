import csv
import numpy as np

file_name = 'ILINet.csv'

def processCsv(file):
    weight_ILI = []
    output_flie = open('weight_ILI.txt', 'w+', encoding='utf-8')
    with open(file) as f:
        reader = csv.reader(f)
        head_row = next(reader)
        title_row = next(reader)
        # print(list(reader))
        count = 1
        weight_per_year = []
        for row in reader:
            weight_per_year.append(row[4])
            # print(row[4])
            if count % 10 == 0:
                weight_ILI.append(weight_per_year)
                output_flie.writelines(str(weight_per_year).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')+'\n')
                weight_per_year = []
            count += 1
    print(weight_ILI)
    output_flie.close()
    # np.savetxt('weight_ILI.txt', np.array(weight_ILI, dtype=np.uint8), encoding='utf-8')

if __name__ == '__main__':
    processCsv(file_name)