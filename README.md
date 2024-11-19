# A Reinforcement Learning Approach to Safe and Quiet UAM Traffic Management

This is the github repository for the noise-aware agent developed for the paper: A Reinforcement Learning Approach to Safe and Quiet UAM Traffic Management.

# Installation Ubunutu

## 1. Install project dependencies

1. Navigate to the UAM_noise_aware_agent directory
    ```bash
    cd UAM_noise_aware_agent
    ```
2. Install bluesky
    ```python
    pip install -e .
    ```

For more information on the BlueSky Simulator, please see: https://github.com/TUDelft-CNS-ATM/bluesky

# Config Parameters:

1. The simulation and model parameters are located in conf/config.gin and conf/config_test.gin.
2. If you would like to change the config file used when running main.py, modify the argument in line 383 of main.py:
    ```bash
    gin.parse_config_file("conf/config_test.gin")
    ```

# Running Project

1. Navigate to the UAM_noise_aware_agent directory
    ```bash
    cd UAM_noise_aware_agent
    ```
2. Try running main script
    ```python
    python main.py
    ````


# Visualization

1. Follow steps 1-2 above (Section: Running Project) in a single terminal (Terminal 1). Open a **second** terminal (Terminal 2) and follow the steps below

2. Navigate to the UAM_noise_aware_agent directory
    ```bash
    cd UAM_noise_aware_agent
    ```
3. Start BlueSky
    ```bash
    python BlueSky.py
    ```
4. The GUI should open up. After the GUI has started, in Terminal 1, run step 2 of **Running Project** to start the simulation.
5. In the BlueSky GUI, select the **Nodes** tab on the lower-right side. Select a different simulation node to see the Austin Environment sim.

# Acknowledgements:

This project builds on the D2MAV-A model proposed by Brittain et. al.: https://arxiv.org/pdf/2003.08353