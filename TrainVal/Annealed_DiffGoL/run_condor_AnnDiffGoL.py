import json
from pathlib import Path
from shutil import rmtree
import numpy as np
import htcondor2 as htcondor
from sacred import Experiment
from hashlib import md5
from memory_profiler import profile

ex = Experiment('Train_GoL_AnnDiffGoL')

@ex.config
def config():
    
    seeds = [8421, 221, 502, 700, 1204, 3340, 4501, 6054, 6621, 15678, 19302, 38475, 77293, 91827, 99100]  
    deltas = [1] # ,2,3,4
    shapes = [(15,15)] # , (20,20), (25,25), (30,30)
    path_datos = './Resultados/Resultados_AnnDiffGoL'
    epochs = 1_000
    hold = True
    epochs2hold = 50
    dict_hps = {
            'kernel_size':3,  # La vencidad es un cuadrado 3x3
            'n_hidden_convs':6,
            'n_hidden_filters':48,
            'threshold': 0.5,
            'learning_rate': 0.0001,
            'lambda_phys': 0,
            'lambda_bin': 0,
            'gamma': 2,
            'alpha': 0.75,
             
            }
    batch_size = 2**8
    
    tasks_config_dir = f'./task_config_GoL_AnnDiffGoL'
    condor_output_dir= f'./condor_output_GoL_AnnDiffGoL'
    priority = 0

@ex.main
def main(seeds, deltas, shapes, path_datos, epochs, hold, epochs2hold, dict_hps, batch_size, tasks_config_dir, condor_output_dir, priority):
    
    jobs_data = {}

    for shape in shapes:
        shape_str = str(shape).replace(', ', 'x').replace('[', '').replace(']', '')
        jobs_data[shape_str] = {}
        for delta in deltas:
            jobs_data[shape_str][delta] = []
            for seed in seeds:
                jobs_data[shape_str][delta].append(
                            {
                                "task_config": {
                                    "cpus": 1,
                                    "memory": str(int(6 * (shape[0] / 15)**2)) + 'G',
                                    "gpus": 1,
                                    "gpumemory": 24000,
                                    "experiment_config": "",
                                    # "requirements": "(CUDACapability >= 7.0)",
                                    # The next elements are included just for using them
                                    # in the submit file to determine output file names
                                    "shape": str(shape_str),
                                    "delta": str(delta),
                                    "shard": str(seed)
                                    
                                },
                                "experiment_config": {
                                    "seed":seed, 
                                    "delta": delta,
                                    "shape": shape,
                                    "path_datos": path_datos,
                                    "epochs": epochs,
                                    "hold": hold,
                                    "epochs2hold": epochs2hold,
                                    "dict_hps": dict_hps,
                                    "batch_size": batch_size
                                },
                            }
                        )
    print(jobs_data)
    _launch_condor_jobs(
        jobs_data, 'Train_GoL_AnnDiffGoL', tasks_config_dir, condor_output_dir, priority
    )


def _launch_condor_jobs(jobs_data, name, tasks_config_dir, condor_output_dir, priority):
    tasks_config_dir = Path(tasks_config_dir)
    rmtree(tasks_config_dir, ignore_errors=True)

    total_jobs_count = sum(
        sum(len(jobs) for jobs in datasets.values()) for datasets in jobs_data.values()
    )
    job_count = 0

    print("Creating experiment config json files: ")
    for shape, deltas in jobs_data.items():
        for delta, job_list in deltas.items():
            # Create a json config directory for each estimator and dataset
            est_dat_dir = tasks_config_dir / str(shape) / str(delta)
            est_dat_dir.mkdir(parents=True, exist_ok=True)

            for job_data in job_list:
                job_count += 1
                # Compute de hash of the experiment config to identify each config
                job_hash = job_data["task_config"]["job_hash"] = md5(
                    json.dumps(job_data["experiment_config"], sort_keys=True).encode(
                        "utf-8"
                    )
                ).hexdigest()

                # Save the path of the json file in the task_config dictionary
                job_data["task_config"]["experiment_config"] = str(
                    est_dat_dir / f"s{job_data['task_config']['shard']}"
                    f"_{job_hash}.json"
                )

                # Save the experiment config into a json file
                with open(job_data["task_config"]["experiment_config"], "w") as f:
                    json.dump(job_data["experiment_config"], f)

                # Convert all items from task_config to string to avoid error with condor
                for key, value in job_data["task_config"].items():
                    job_data["task_config"][key] = str(value)

                print(f"{job_count}/{total_jobs_count}", end="\r")

        print("")

    condor_output_dir = Path(condor_output_dir)
    rmtree(condor_output_dir, ignore_errors=True)
    condor_output_dir.mkdir(parents=True, exist_ok=True)

    for shape, deltas in jobs_data.items():
        for delta, job_list in deltas.items():
            # Create condor output directories
            (condor_output_dir / str(shape) / str(delta)).mkdir(parents=True, exist_ok=True)

            schedd = htcondor.Schedd()
            job = htcondor.Submit(
                {
                    "executable": "run_experiment_AnnDiffGoL.sh",
                    "arguments": "with $(experiment_config)",
                    "getenv": "True",
                    "output": f"{str(condor_output_dir)}/$(estimator_name)/$(shape)/$(delta)/output_s_$(job_hash).out",
                    "error": f"{str(condor_output_dir)}/$(estimator_name)/$(shape)/$(delta)/error_s_$(job_hash).err",
                    "log": f"{str(condor_output_dir)}/$(estimator_name)/$(shape)/$(delta)/log_s_$(job_hash).log",
                    "should_transfer_files": "NO",
                    "request_GPUs": "$(gpus)",
                    # "require_GPUs": "GlobalMemoryMb >= $(gpumemory)",
                    "request_CPUs": "$(cpus)",
                    "request_memory": "$(memory)",
                    "batch_name": f"{str(shape).replace(', ', 'x').replace('[', '').replace(']', '')}_{str(delta)}_{name}",
                    "priority": priority,
                    "on_exit_hold": "ExitBySignal == True || ExitCode != 0",
                }
            )

            submit_result = schedd.submit(
                job, itemdata=iter(map(lambda j: j["task_config"], job_list))
            )
            print(
                f"{submit_result.num_procs()} jobs submitted"
                f" as cluster {submit_result.cluster()}"
                f" for {shape} and {delta}."
            )

if __name__ =='__main__':
    ex.run_commandline()