## Description

This project showcases our work in the BIO project, which is identification of a person using finger vein images. Authors of this project are: Vaclav Korvas (xkorva03) and Adam Zvara (xzvara01).

## Installation

Before installing the project, make sure that the OpenCV library for C++ is installed on your system. You can install it by running the following command.
The installation process can be found at the [official OpenCV website](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html). Or alternatively, if you use `apt` package manager, you can run the following command:

```bash
sudo apt-get install -y libopencv-dev
```

There are two main ways to install this project.

Since some files require the `bob` library in order to run (e.g. `bob_example_methods.py`), you first need to install
the library. We recommend following the installation instructions at the [official bob website](https://www.idiap.ch/software/bob/docs/bob/docs/stable/install.html).
After you install the library and activate the conda environment (`conda activate bob_env1`), you can install the rest of the requirements by running the setup script.

You can choose not to install the `bob` library and run our own implementation of the vein extraction and matching algorithms. However, you will not be able to run
the `bob_example_methods.py` file and you won't be able to import extraction and matching algorithms for evaluation. In this case, we recommend creating a virtual environment and
running the setup script:

```bash
python -m venv bvein_venv
source bvein_venv/bin/activate
chmod +x setup.sh
./setup.sh
```

! Please note that you need to run the setup script either way, as it also compiles the C++ RLT implementation which is used by other files !

After the installation of packages is complete, you need to specify database path, as the database is not included in the repository. You can do this by setting the `DB_PATH` environment variable to the path of the database.

```bash
export DB_PATH=/path/to/your/database
```

## Examples

There are 3 simple example files which require no arguments and serve as an example of preprocessing and vein extraction methods:
- `rlt.py` - Runs the RLT algorithm on randomly selected images from database and plots the results.
- `gabor.py` - Runs the Gabor filter algorithm on randomly selected images from database and plots the results.
- `bob_example_methods.py` - Runs the Bob library methods on randomly selected images from database and plots the results (this requires the `bob` library to be installed).

You can run the evaluation script to evaluate the performance of the implemented algorithms. By default the script uses modified RLT algorithm with 50 tests and 30 images per test and 800 iterations.
Since this can take a while to run, you can specify the number of tests, images per test and iterations by passing the arguments to the script. Here is an example of how to run the evaluation script with 5 tests, 5 images per test and 800 iterations:

```bash
python evaluate.py -n 5 -b 5
```

This stores the model in `models/RepeatedLineTracking_5_5_800.pkl` and also stores the results in `results/RepeatedLineTracking_5_5_800` - by default the proposed matcher is used.
If you wish to evaluate the same model but by using MiuraMatch, unfortunately you need to uncomment sections where the matcher is defined in the `evaluate.py` file. But the you can
run it on the same model with:

```bash
python evaluate.py -n 5 -b 5 -f models/RepeatedLineTracking_5_5_800.pkl
```

Which will overwrite the previous results file. You can plot the ROC and DET curves of this file by running:

```bash
python plot.py results/RepeatedLineTracking_5_5_800
```

Finally we have also included some of the results in the `results` folder, which you can plot by running the `plot.py` script.
- to plot ROC of different matching methods: `python plot.py results/MatchingMethodsComparison`
- to plot ROC of Bob RLT vs our implementation: `python plot.py results/BobRLT_50_30_800 results/RepeatedLineTracking_50_30_800 results/RepeatedLineTracking_50_30_1000_modified`
- to plot ROC of different iteration counts in RLT: `python plot.py results/RepeatedLineTracking_50_30_800 results/RepeatedLineTracking_50_30_1000`