from run import batchNtest
import threading
import pandas as pd
import yaml
th = threading.Semaphore(24)

def load_config(config_path='ievd/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
# [0,3,4,6,7,8,9]
if __name__ == '__main__':
    config = load_config('ievd/config2.yaml')
    all_data = []
    for mode in [8]:
        print('\n mode=',mode)

        df = batchNtest(th,config=config,mode=mode)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # 导出到 Excel
    output_path = 'results.xlsx'
    combined_df.to_excel(output_path, index=False)
    print(f'\n 结果已导出到: {output_path}')

    print('\n 汇总统计:')
    summary = combined_df.groupby(['N', 'mode']).agg({
        'tIEVD_time': 'mean',
        'tEIG_time': 'mean',
        'residual': 'mean',
        'orthogonality': 'mean',
        'eigenvalue_error': 'mean'
    }).reset_index()
    print(summary)

    # config = load_config('ievd/config1.yaml')
    # all_data = []
    # for mode in [0,3,4,6,7,8,9]:
    #     print('\n mode=',mode)

    #     df = batchNtest(th,config=config,mode=mode)
    #     all_data.append(df)

    # combined_df = pd.concat(all_data, ignore_index=True)

    # # 导出到 Excel
    # output_path = 'results1.xlsx'
    # combined_df.to_excel(output_path, index=False)
    # print(f'\n 结果已导出到: {output_path}')

    # print('\n 汇总统计:')
    # summary = combined_df.groupby(['N', 'mode']).agg({
    #     'tIEVD_time': 'mean',
    #     'tEIG_time': 'mean',
    #     'residual': 'mean',
    #     'orthogonality': 'mean',
    #     'eigenvalue_error': 'mean'
    # }).reset_index()
    # print(summary)

    # config = load_config('ievd/config2.yaml')
    # all_data = []
    # for mode in [0,3,4,6,7,8,9]:
    #     print('\n mode=',mode)

    #     df = batchNtest(th,config=config,mode=mode)
    #     all_data.append(df)

    # combined_df = pd.concat(all_data, ignore_index=True)

    # # 导出到 Excel
    # output_path = 'results2.xlsx'
    # combined_df.to_excel(output_path, index=False)
    # print(f'\n 结果已导出到: {output_path}')

    # print('\n 汇总统计:')
    # summary = combined_df.groupby(['N', 'mode']).agg({
    #     'tIEVD_time': 'mean',
    #     'tEIG_time': 'mean',
    #     'residual': 'mean',
    #     'orthogonality': 'mean',
    #     'eigenvalue_error': 'mean'
    # }).reset_index()
    # print(summary)

    # config = load_config('ievd/config3.yaml')
    # all_data = []
    # for mode in [0,3,4,6,7,8,9]:
    #     print('\n mode=',mode)

    #     df = batchNtest(th,config=config,mode=mode)
    #     all_data.append(df)

    # combined_df = pd.concat(all_data, ignore_index=True)

    # # 导出到 Excel
    # output_path = 'results3.xlsx'
    # combined_df.to_excel(output_path, index=False)
    # print(f'\n 结果已导出到: {output_path}')

    # print('\n 汇总统计:')
    # summary = combined_df.groupby(['N', 'mode']).agg({
    #     'tIEVD_time': 'mean',
    #     'tEIG_time': 'mean',
    #     'residual': 'mean',
    #     'orthogonality': 'mean',
    #     'eigenvalue_error': 'mean'
    # }).reset_index()
    # print(summary)