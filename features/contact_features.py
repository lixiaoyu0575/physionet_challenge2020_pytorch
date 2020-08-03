import csv
def get_data():
    filename_A = 'Out_Feature_Emilia_A.csv'
    filename_E = 'Out_Feature_Emilia_E.csv'
    filename_H = 'Out_Feature_Emilia_H.csv'
    filename_S = 'Out_Feature_Emilia_S.csv'
    filename_I = 'Out_Feature_Emilia_I.csv'
    filename_Q = 'Out_Feature_Emilia_Q.csv'
    fileAll = "Out_Feature_Emilia.csv"   #合并输出文件
    df_A = open(filename_A,'r',encoding='utf-8').read()
    df_E = open(filename_E, 'r', encoding='utf-8').read()
    df_H = open(filename_H, 'r', encoding='utf-8').read()
    df_S = open(filename_S, 'r', encoding='utf-8').read()
    df_I = open(filename_I, 'r', encoding='utf-8').read()
    df_Q = open(filename_Q, 'r', encoding='utf-8').read()

    with open(fileAll,'a',encoding='utf-8') as f:
        f.write(df_A)
        f.write(df_E)
        f.write(df_H)
        f.write(df_S)
        f.write(df_I)
        f.write(df_Q)

    f.close()

get_data()