"""Execute one dataset notebook headlessly; preserve its saved empty outputs."""
import argparse
import json
import os
from pathlib import Path

from build_dataset_notebooks import CATALOG, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=[spec[1] for spec in CATALOG])
    parser.add_argument('--mode', choices=['benchmark', 'smoke'], default=os.environ.get('SDM_RUN_MODE', 'benchmark'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--offline', action='store_true', help='Require cached data and pretrained weights; do not download.')
    args = parser.parse_args()
    os.environ['SDM_RUN_MODE'] = args.mode
    if args.output_root: os.environ['SDM_RUN_DIR'] = str(args.output_root.expanduser().resolve())
    if args.data_root: os.environ['SDM_DATA_DIR'] = str(args.data_root.expanduser().resolve())
    if args.offline: os.environ['SDM_DOWNLOAD'] = '0'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.show = lambda: plt.close('all')
    number = next(number for number, key, *_ in CATALOG if key == args.dataset)
    path = ROOT / f'notebooks/{number}_{args.dataset}_sdm_calibration_comparison.ipynb'
    notebook = json.loads(path.read_text())
    namespace = {'display': lambda *args: None, '__name__': '__main__'}
    os.chdir(ROOT)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            print(f'Running {path.name}: {cell["id"]}', flush=True)
            exec(compile(''.join(cell['source']), f'{path.name}:{cell["id"]}', 'exec'), namespace)
    print(namespace['temperature_table'].to_string(index=False))
    print(namespace['metrics_table'].to_string())
    print(f'Completed {args.mode} run: {namespace["RUN_DIR"]}')
    if args.mode == 'smoke': print('Pipeline check only; these are not benchmark results.')


if __name__ == '__main__':
    main()
