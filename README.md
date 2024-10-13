## Description

## How to run

## Examples

## Installation

### bio-bob reference examples
This project also contains examples of vein identification from [bob toolbox](https://www.idiap.ch/software/bob/), specifically the [bob.bio.vein](https://github.com/bioidiap/bob.bio.vein/tree/master). These examples are located in the `bob_bio` directory. In order to run them you must first install the `bob` library. The installation instructions can be found [here](https://www.idiap.ch/software/bob/docs/bob/docs/stable/install.html).

- **compare_methods.py**
    - compares the performance of different vein preprocessing and extraction methods on 3 example images stored in
`bob_bio/vein_imgs` directory. The extraction algorithms present are RepeatedLineTracking, MinimumCurvature, WideLine and PrincipalCurvature. You can optionally specify preprocessing steps to be applied before the extraction, or to show the intermediate results of the preprocessing.

- **match_test.py**
    - uses the publicly available [finger vein dataset](https://www.kaggle.com/datasets/ryeltsin/finger-vein), to test how well the vein extraction algorithms work with vein matching. Preprocessing consists of calculating mask (LeeMask algorithm) and histogram equalization. Extraction method is MinimumCurvature and the matching method is MiuraMatch.
    - how does it work? The script takes a random image of a single finger (which is set as a target) and compares them to:
        - different pictures of the same finger (should match)
        - randomly selected pictures of other fingers (should not match)
    - the results are numbers from 0 to 0.5 for each pair of target image and tested image - where 0.5 means that the images are identical and 0 means that they are completely different
    - please make sure you set the **`DB_PATH` variable** to your local path to the dataset
