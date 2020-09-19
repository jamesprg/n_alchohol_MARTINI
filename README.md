# "Effect of alcohol on the phase separation in model membranes"
# James Ludwig and Lutz Maibaum

With an goal of making our scientific work both reproducible and transparent, this repository is home to supplemental files relating to the paper listed above. In this repo, you will find both analysis code and examples/files used to conduct or research. Futhremore, a brief outline of the methods/steps will be housed here. Along with the scripts present in this repo, all tools used to conduct this work are open source. We hope this repo supports those interested in the study of model membranes by way of molecular dynamics, paticularly using the MARTINI forcefield.

### Required Software

As described in the Methods section (3) of the paper, we use a combination of open source software. In order to recreate our results and conduct analysis, the following open source softward was used:

- [GROMACS](http://www.gromacs.org/) (2016.3)
- [PLUMED](https://www.plumed.org/) (2.4)
- [python](https://www.python.org/) (3.6)
- [MARTINI](http://cgmartini.nl/) (2.2)
- [WHAM](http://membrane.urmc.rochester.edu/?page_id=126) (2.0+)

Various MARTINI related scripts can be found on the tool section of the site (http://cgmartini.nl/index.php/tools2/proteins-and-bilayers) such as the default insane script. The methods described here are heavily based in the GROMACS framework and likely work best remaining as such if hoping to conduct a similar project using any of these files.

We hope that many of the files and tools provided here can be applied to other MARTINI based model membrane projects, with little to no adaptions needed. 

### Guide

- __Step 1__ - Use our methods section and/or MARTINI tutorials to generate phase separated membranes of desired lipid composition in aqueous solution. If studying alcohols, create additional systems with various alcohol additions. All membrane systems  can be quickly generated using MATRINI's tools or our version of `insnane.py` (it should be noted that MARTINI is capable of handling alcohol's now, but was not at the time this study was started).

- __Step 2__ - Generate density profiles from the lipid membrane with `gmx density` in order to determine the compositions of each phase.

- __Step 3__ - Follow our methods section in order to equilibrate and run single phase systems (again n-alcohol addition is completely optional).

- __Step 4__ - Use a combination of metadynamics and umbrella sampling (we do so with [PLUMED]) in order to remove a desired cholesterol molecule and measure the free energy of this transition in various conditions. Associated scrips for this process are found in the scripts/umbrella_scripts/ directory. The WHAM method is then used to stitch each window together generating the compltete free energy surface of the process. The WHAM command is included in `pmfcalculation.sh` bash script.

- __Step 5__ -  Calculation of various membrane properties can be conducted on the resulting data using various tools in the scripts/ directory. 

  - __Step 5.1__ - Use `bilayerthickness.py` in order to calculate the width of each model membrane.
  
  - __Step 5.2__ - Use `do-order-multiedit.py` (minor tweaks to the one provided by MARTINI) in     order to calculate the lipid tail order parameter described in section 3.2 of the paper.
  - __Step 5.3__ - Use `flipflop.py` in order to calculate number of bilayer crossing events for alcohol and cholesterol in a model MARTINI membranes (built for single phase membrane systems).
  - __Step 5.4__ - Use `density.py` quickly generate the densities for components in the direction normal to the membrane (built for single phase membrane systems). 

### Citation

When using our code (or any of the referenced software), please cite the respective creators or our repository and paper. The paper can be found here.
