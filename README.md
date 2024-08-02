# ILASMS_func3a

This is the git repository for Function 3a of the NASA System-Wide Safety program.


# Installation Mac M1/2

## Install Miniforge (only if using a Mac M1/M2)


* `wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh`

* `bash Miniforge3-MacOSX-arm64.sh`

* `rm https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh`


# Install project dependencies

1. Close and open a new terminal
2. Navigate to ILSMS_FUNC3A directory
    ```bash
    cd to/ILASMS_FUNC3A
    ```
3. Navigate to the setup directory 
    ```bash
    cd setup
    ```
4. Install dependencies with conda
    ```bash
    conda env create -f environment_mac_silicon.yml (if using a Mac M1/M2)
    ```

    ```bash
    conda env create -f environment.yaml
    ```

    Note that `environment.yaml` lists only major conda installed packages and you may be prompted to install additional pip packages if there are some missing. You will find out if so when running `python main.py` (see the Running Project section for more info).

5. Close and open a new terminal
6. Activate virtual environment
    ```bash
    conda activate sws
    ```

7. Install the Ray package
    ```bash
    pip install "ray[default]"
    ```
8. Naviagate to project directory (see step 1)
9. Naviagate to BlueSky directory
    ```bash
    cd bluesky
    ```
10. Install bluesky
    ```python
    pip install -e .
    ```

# Running Project

1. Open terminal and activate virtual environment
    ```bash
    conda activate sws
    ```
2. Navigate to ILSMS_FUNC3A directory
    ```bash
    cd to/ILASMS_FUNC3A
    ```
3. Try running main script
    ```python
    python main.py
    ````


# Visualization

1. Follow steps 1-2 above (Section: Running Project) in a single terminal (Terminal 1). Open a **second** terminal (Terminal 2) and follow the steps below
2. Activate virtual environment
    ```bash
    conda activate sws
    ```
2. Navigate to ILSMS_FUNC3A directory
    ```bash
    cd to/ILASMS_FUNC3A
    ```
3. Naviagate to BlueSky directory
    ```bash
    cd bluesky
    ```
4. Start BlueSky
    ```bash
    python BlueSky.py
    ```
5. The GUI should open up. After the GUI has started, in Terminal 1, run step 3 of **Running Project** to start the simulation.
6. In the BlueSky GUI, select the **Nodes** tab on the lower-right side. Select a different simulation node to see the DFW sim.


# Issues

## 1. grpcio error during `python main.py`

If this error is encountered, uninstall grpcio and reinstall with conda

```bash
pip uninstall grpcio; conda install grpcio=1.43.0
```


# Generate multiple scenarios

Run `scripts/multiple_scn_script.py`. The generated scenarios are stored in `scripts/generated_scenarios`. If you want to modify how each scenario file is generated, `multiple_scn_script.py` calls `scripts/demo-multiple.py` under the hood to perform the generation. The other files in this directory (e.g., demo-version\<N>) are old development scripts.

# Running the Final Demo

There is a folder named "final_demo" that contains the finalized models for running function 3a and function 3 respectively. In the sub directory, `models`, `no_tm` represents function 3a where it is only the separation assurance logic without the traffic manager (tm, function 3b). The `tm_25_capacity` folder represents the enture Function 3 algorithm with both the traffic manager and the separation assurance logic with a capacity of 25 aircraft in each corridor.

There are two scenario files listed: `IC1.scn` and `IC2.scn` which are two corridor incursion scenarios. In both files, the corridor incursion event occurs at 1:28:00 into the simulation where an aircraft called `GA<N>` will spawn in and blunder through the corridor network. These would be two good scenarios to show in the final demo.

To run either of these scenario files for eval, you will need to modify the config.gin file prior to running `python main.py`. Currently the configuration file is set to run the `IC1.scn` scenario and can be modified to run `IC2.scn` by simply replacing the scenario name.

To change the background map in BlueSky, first uncomment line 10 of `settings.cfg` (changes the color palette for a white background). Then, when the GUI is active, run `VIS MAP TILEDMAP` in the BlueSky comamnd line. This will switch the background to an openstreetmap style view.


# Running training results

I have included all of my training models and results where the results can be visualized in the `Visualization Results.ipynb` notebook. In addition, there is a notebook called `Process_Results_Heatmap.ipynb` that details how to generate LOS heatmaps. Although, I have also included the heatmaps in the `heatmaps` directory.
