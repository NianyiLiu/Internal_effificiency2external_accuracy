# Internal_effificiency2external_accuracy

## Environment Setup

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

2. Install additional dependencies:

Clone and install the `conn2res` package:

```bash
git clone -b v0.0.1 https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
git clone https://github.com/netneurolab/conn2res.git
cd conn2res
pip install .
git clone https://github.com/MICA-MNI/ENIGMA.git
cd ENIGMA
python setup.py install
```

## Network Generation

After setting up the environment, generate the network files by running:

```bash
python network_generation.py 1000
```
Replace `1000` with desired number of networks (typically 1000).

## Task Training

To train tasks, use the following command:

```bash
python rc_train.py --task_type neurogym -c 2 --noise 0.01 --save_w_in --exponential --start_number 0 --stop_number 1000
```

- The `-c` parameter specifies the network to use (replace with desired network index).
- The `--task_type` parameter specifies the task type (e.g., `neurogym`, can be replaced as needed).
- Adjust other parameters as required for the experiment.

## Analysis

After training, analysis results can be generated using the provided analysis scripts. For example:

```bash
python analysis_rc.py --start 0 --stop 1000 -c 2
```

- The `--func` parameter specifies the analysis function to run (e.g., `fc_tasks`, `fc`, `task_variance_only`, etc.).
- The `--start` and `--stop` parameters specify the range of data to analyze.
- The `-c` parameter specifies the network to use.
- Adjust parameters as needed.

