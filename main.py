from run import batchNtest
import threading
import pandas as pd
import yaml
th = threading.Semaphore(24)

def load_config(config_path='ievd/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    config = load_config('config.yaml')
    all_data = []
    for mode in range(6):
        print('\n mode=',mode)

        df = batchNtest(th,config=config,mode=mode)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    output_path = 'results.xlsx'
    combined_df.to_excel(output_path, index=False)
    print(f'\n result to: {output_path}')

    print('\n analysis:')
    summary = combined_df.groupby(['N', 'mode']).agg({
        'tIEVD_time': 'mean',
        'tEIG_time': 'mean',
        'residual': 'mean',
        'orthogonality': 'mean',
        'eigenvalue_error': 'mean'
    }).reset_index()
    print(summary)
