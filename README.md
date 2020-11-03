# FE model updating in Python

Finite element (FE) model updating in Python by an example. The framework of sensitivity-based FE model updating is implemented in Python
and validated through a simple numerical case study utilizing the FE model program Abaqus.

Please cite both the original research paper which the code was developed for, and the code, if used for research purposes.

## Installation
These files can be used directly by downloading or cloning the repository. Note that the FEM software Abaqus is required to run the files.

## Content

![](test_fig.svg)


FE model updating is performed by considering a simply supported beam. Examples of results from the postprocessing is shown below (included in the
example files).


The files include:
- Implementation of the *theoretical framework* of sensitivity-based FE model updating by perturbation analysis.
- Demonstration of an *analysis framework*, or workflow setup, utilizing the numerical FE model program Abaqus.
- Description of a numerical case study including results for validation.

Experience with Abaqus and Abaqus scripting is not needed, but preferable, for an increased understanding of the numerical case study implementation.
The workflow setup is general and the theoretical framework of the model updating can be utilized with other FEM programs by simple modifications to
the provided files.

The easiest way to get into the use and understanding is to download and run the example files. For additional details of theory,
implementation and use, please see the reference below.


## Support

Please [open an issue](https://github.com/bjorntsv/pyfemu/issues/new) for support.

## References
[1]
