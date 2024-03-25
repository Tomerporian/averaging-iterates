import yaml
import os
import shutil
import itertools
import argparse
import subprocess
import datetime
import pandas as pd
import numpy as np
from copy import deepcopy
from time import sleep
import tqdm

def format_key(k):
    return '_'.join([kk[:3] for kk in k.split('_')])


def format_value(v):
    return str(v).replace(' ', '').replace('/', '-')


def unwind_grid(grid_dict):
    list_keys = [k for k in grid_dict.keys() if k.startswith('LIST')]
    grid = []
    for k in list_keys:
        subgrid_list = grid_dict.pop(k)
        subgrid = []
        for subgrid_dict in subgrid_list:
            subgrid.extend([tuple(zip(*el)) for el in unwind_grid(subgrid_dict)])
        grid.append(tuple(set(subgrid)))
    for k, v in grid_dict.items():
        if isinstance(v, str):
            v = list(eval(v))
        if not isinstance(v, list) or isinstance(v, tuple):
            v = [v]
        grid.append(tuple((k, vv) for vv in v))
    return list(itertools.product(*grid))


def grid_to_str(list_of_dicts):
    nunique = pd.DataFrame(list_of_dicts).nunique()
    nunique = nunique[nunique > 1]
    # print(nunique)
    # print(list_of_dicts)
    assert nunique.prod() == len(list_of_dicts), ValueError('nunique.prod() is {}, but len(list_of_dicts) is {}'.format(
        nunique.prod(), len(list_of_dicts)))
    return ' x '.join(f'{n} {k}' for k, n in nunique.items())


def expand_tuple_keys(dict_with_tuples):
    tuple_keys = [k for k in dict_with_tuples if isinstance(k, tuple)]
    if len(tuple_keys) == 0:
        return dict_with_tuples
    else:
        for kk in tuple_keys:
            vv = dict_with_tuples.pop(kk)
            dict_with_tuples.update(dict(zip(kk, vv)))
        return expand_tuple_keys(dict_with_tuples)

def get_slurm_script(num_nodes, num_gpus):
    if num_nodes not in [1, 2, 4, 8]:
        raise ValueError('Only 1, 2, 4, or 8 nodes are supported for now.')
    if num_nodes > 1:
        sbatch_file = f"scripts/jsc_script_{num_nodes}_nodes.sh"
        print(f"Using {sbatch_file} for {num_nodes} nodes")
        return sbatch_file
        # for example if num_nodes = 2, then the script is "scripts/jsc_script_2_nodes.sh"
    else:
        return f"scripts/jsc_script_{num_gpus}_gpus.sh" if num_gpus > 1 else "scripts/jsc_script_1_gpu.sh"
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jobfile', type=str,
                        help='path to YAML file containing job configuration')
    parser.add_argument('-s', '--script',default="", type=str, help='name of script to provide to sbatch commands')
    parser.add_argument('-y', '--yes', action='store_true', help='confirm submission without viewing grid details first')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing results directory')
    parser.add_argument('-d', '--dry_run', action='store_true',
                        help='prints the sbatch commands instead of executing them, and does not create an output folder')
    parser.add_argument('-r', '--rerun', action='store_true',
                        help='re-submits all jobs not marked as done; when used the jobfile argument should be the path '
                             'to the results folder of a previous execution of the script.')
    parser.add_argument('-c', '--chunk_size', type=int, default=1, help='number of jobs to submit at once')
    parser.add_argument('-n', '--num_configs',default=0, type=int, help='running limited number of configorations, for dealing with overwrite mistakes')
    parser.add_argument('-a', '--autorestart',action='store_true', help='use autorestart.py script on jsc')

    args = parser.parse_args()

    run_str = 'RUN' if not args.rerun else 'RE-RUN'
    if args.dry_run:
        print(f'===== THIS IS A ***DRY*** {run_str} =====')
    else:
        print(f'===== THIS IS A ***REAL*** {run_str} =====')
    
    if not args.rerun:
        with open(args.jobfile, 'r') as f:
            job_description = yaml.safe_load(f)

        job_details = job_description['job_details']
        output_dir = job_details['output_dir']
        if 'num_gpus' in job_details or 'num_nodes' in job_details:
            num_gpus = job_details['num_gpus']
            num_nodes = job_details['num_nodes']
            # args.script = get_slurm_script(num_nodes, num_gpus)
            args.script = f"scripts/jsc_script_general.sh {num_nodes} {num_gpus}"
        batch_name = datetime.datetime.now().strftime('%y-%m-%d') + '-' + job_details['name']
        batch_dir = os.path.join(output_dir, batch_name)
    else:
        jobfile = os.path.join(args.jobfile, 'job.yaml')
        with open(jobfile, 'r') as f:
            job_description = yaml.safe_load(f)
        batch_dir = args.jobfile
        batch_name = os.path.split(batch_dir)[-1]

    if os.path.exists(batch_dir) and os.listdir(batch_dir):
        if not (args.overwrite or args.rerun):
            raise FileExistsError('Directory exists and overwrite flag is not set')
        if args.overwrite:
            print('Removing existing output directory')
            shutil.rmtree(batch_dir)

    print(f'Writing experiment result to directory {batch_dir}')

    # --------------------- HANDLE CONFIG ENUMERATION------------------------
    configs = [dict(c) for c in unwind_grid(deepcopy(job_description['parameters']))]
    configs_summary_str = grid_to_str(configs)
    configs = list(map(expand_tuple_keys, configs))
    total_configs = len(configs)
    nunique_keys = pd.DataFrame(configs).nunique()
    varying_keys = nunique_keys[nunique_keys > 1].index.values

    print(f'Ready to run {total_configs} configurations: {configs_summary_str}')
    if args.num_configs:
        len_configs = len(configs)
        configs = configs[:args.num_configs]
        print(f'Running limited number of configurations: {len_configs - args.num_configs}')
    if not args.yes:
        input("Press Enter to continue...")
    # --------------------- HANDLE CONFIG ENUMERATION------------------------
    def grid_generator():
        chunk_spec_filename = []
        for i, kvs in enumerate(configs):
            spec_name = '+'.join([f'{format_key(k)}={format_value(kvs[k])}'
                                 for k in varying_keys])
            spec_name = f'{i:03d}_{batch_name}+{spec_name}'
            out_dir = os.path.join(batch_dir, spec_name)
            if os.path.exists(os.path.join(out_dir, 'done')):
                continue
            spec = dict(output_dir=out_dir, **kvs)
            spec_filename = os.path.join(out_dir, 'spec.yaml')
            spec['output'] = out_dir
            # spec['experiment'] = out_dir
            # if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)
            with open(spec_filename, 'w') as f:
                yaml.dump(spec, f, sort_keys=True)
            chunk_spec_filename.append(spec_filename)
            if len(chunk_spec_filename) == args.chunk_size:
                yield spec_name, chunk_spec_filename, out_dir
                chunk_spec_filename = []
        if chunk_spec_filename:
            yield spec_name, chunk_spec_filename, out_dir
    
    def grid_generator_rerun():
        chunk_spec_filename = []
        for spec_name in os.listdir(batch_dir):
            out_dir = os.path.join(batch_dir, spec_name)
            if not os.path.isdir(out_dir) or spec_name.startswith('.'):
                continue
            if os.path.exists(os.path.join(out_dir, 'done')):
                continue
            spec_filename = os.path.join(out_dir, 'spec.yaml')
            chunk_spec_filename.append(spec_filename)
            if len(chunk_spec_filename) == args.chunk_size:
                yield spec_name, chunk_spec_filename, out_dir
                chunk_spec_filename = []
        if chunk_spec_filename:
            yield spec_name, chunk_spec_filename, out_dir

    # --------------------- HANDLE OUTPUT FOLDER -------------------------------
    if not (args.dry_run or args.rerun):
        os.makedirs(batch_dir, exist_ok=True)
        job_outfile = os.path.join(batch_dir, 'job.yaml')
        if os.path.exists(job_outfile):
            c = 1
            while os.path.exists(os.path.join(batch_dir, f'job.old.{c}.yaml')):
                c += 1
            os.rename(job_outfile,
                      os.path.join(batch_dir, f'job.old.{c}.yaml'))
            print(f'Renaming old job file to job.old.{c}.yaml')

        with open(job_outfile, 'w') as f:
            yaml.dump(job_description, f, default_flow_style=False)

    count = 0
    gen = grid_generator if not args.rerun else grid_generator_rerun
        
    for spec_name, spec_filename, out_dir in gen():
        # spec_filename = ['--config ' + s for s in spec_filename]

        specs = " ".join(spec_filename) if args.chunk_size > 1 else spec_filename[0]

        out_path = os.path.join(out_dir, '%j.out')
        # print("Running command: ", f'sbatch --job-name={spec_name} --output={out_path} --error={out_path} {args.script} {specs}')
        cmd = f'sbatch --job-name={spec_name} --output={out_path} --error={out_path} {args.script} {specs} {spec_name} {batch_dir}'
        # if args.autorestart:
        #     template_str = '{job_id}.out'
        #     cmd = f'python autorestart.py "sbatch --job-name={spec_name} --output={out_path} --error={out_path} {args.script} {specs}" --path_to_folder={out_path} --output-file-template="{template_str}" --check-interval-secs=900 --verbose'
        #     cmd = f'screen -dm bash -c "{cmd}"'
        if args.dry_run:
            # print(f'Would now run "{cmd}"')
            if args.rerun:
                with open(os.path.join(out_dir, 'config.yaml'), 'r') as f:
                    prev_config = yaml.safe_load(f)
                print(f'Host of previous failed run: {prev_config["host_name"]}')
        else:
            while True:
                try:
                    sbatch_output = subprocess.run(cmd, shell=True, check=True)
                    break
                except subprocess.CalledProcessError:
                    print('Encountered called process error while submitting, waiting and trying again')
                    sleep(4 + float(np.random.rand(1) * 5))
        count += 1
        if count % 500 == 0 and not args.dry_run and not count==7500:
            print(f'\nStarted {count} jobs\n')
            sleep(120 + float(np.random.rand(1) * 5))
        # if count==7500:
        #     break
    print(f'\nStarted {count} jobs in total')




