# Fixed point ML model training demonstration

## Authors

Keyur Joshi (keyurpjoshi@gmail.com)

## Abstract

This artifact presents a library for training fixed point ML models.
Fixed point datatypes are used for all computations that are affected by the training data.
Floating point computations are used:

* For calculations involving only "publicly available" information such as the ADAM optimizer settings, and
* For calculating accuracy and loss as these do not actually affect the training regime.

It should be possible to replace these floating point computations with fixed point ones.
The gradient of loss, (which *is* relevant to training) is calculated using fixed point datatypes.

Alongside the library, this artifact contains an example script for training an LSTM cell on the USPS 10 dataset.
Three critical functions need to be approximated for this to be possible: sigmoid, tanh and rsqrt (reciprocal of square root).
These approximations are created using Multi-Element Generalized Polynomial Chaos (ME-GPC).

## Prerequisites and Setup

A system with a modern version of Python 3 or equivalent is required.

To run the training regime with the provided approximations, you need numpy and sklearn, installed via:

`python3 -m pip install numpy scikit-learn`

The example model is trained on the USPS 10 dataset. To set it up, run

`setup_usps10.sh`

from the `data` directory.

To create additional or alternative approximations, you also need [chaospy](https://chaospy.readthedocs.io/), installed via:

`python3 -m pip install chaospy`

## Structure

* The `data` directory contains the dataset and the script for preprocessing it.
* The `models` directory will contain the trained model as well as the training log.
* The `EzPC` directory contains code to run the EzPC version of this artifact.
* `fxpTensor.py` contains the fixed point tensor manipulation library. It is modeled after pytorch's own Tensor datatype.
* `lstmUsps10.py` trains a (optionally low rank) LSTM cell on the USPS 10 dataset using the `fxpTensor` library.
* `myTensor.py` and `lstmUsps10Float.py` are floating point equivalents of the above files, used for comparison.
* `iterativeapprox.py` generates approximations for sigmoid, tanh, rsqrt, etc. functions using ME-GPC.
* The `old` directory contains some older code. Most of its functionality is replaced by the other code.

## Quick Start

After completing the setup steps above, simply run `./lstmUsps10.py` from the artifact root directory to train a fixed point model.
Training should not take more than 30 minutes on a modern machine.
You can compare the training statistics (mainly accuracy) against the float version, which is run in a similar fashion.

## EzPC Code Generation and Usage

See `README.md` file located in the EzPC directory for details on using this artifact with EzPC.

## Creating Approximations

Edit and run `iterativeapprox.py` to approximate necessary functions.
You can adjust the range of approximation, order of approximation (higher is more precise but costlier) and target error in terms of actual error and ULPs.
For sigmoid and tanh, I strongly recommend that the range is chosen so that the approximation of the first segment (furthest from 0) is a constant.
To check that the first segment approximation is a constant, ensure that the first column of the output `segmentCoeffs` is all 0, except the last row.
Not doing this may cause the approximation to fail for inputs with an unexpectedly large absolute value.
Alternatively, you can manually edit the first segment approximation to satisfy the condition given above.

## Extensions

Additional datasets can be used by editing the input and training parameters at the top of `./lstmUsps10.py`.
Different LSTM configurations or even completely different RNN architectures can be used by more extensive editing of `./lstmUsps10.py`.

## Note on Loss

The library does not support softmax cross entropy loss.
Using that loss function continuously pushes the correct and incorrect classes activations apart, even if they are already quite far apart.
This leads to overflows with increasing frequency as training progresses.
Using the log-sum-exp trick *does not* prevent these overflows.
Using mean squared error loss instead sets static targets for the correct and incorrect classes, that we can ensure are well within the representable range.

