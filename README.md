# Breathing Earth System Simulator (BESS) Model Python Implementation

Gregory H. Halverson (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer<br>
NASA Jet Propulsion Laboratory 329G

Youngryel Ryu (he/him)<br>
[yryu@snu.ac.kr](mailto:yryu@snu.ac.kr)<br>
BESS algorithm inventor<br>
Seoul National University

Hideki Kobayashi (he/him)<br>
[hkoba@jamstec.go.jp](mailto:hkoba@jamstec.go.jp)<br>
FLiES algorithm inventor<br>
Japan Agency for Marine-Earth Science and Technology

Robert Freepartner (he/him)<br>
[robert.freepartner@jpl.nasa.gov](robert.freepartner@jpl.nasa.gov)<br>
MATLAB to python translation<br>
Raytheon

Joshua Fisher (he/him)<br>
[jbfisher@chapman.edu](mailto:jbfisher@chapman.edu)<br>
Concept development and project management<br>
Chapman University

Kerry Cawse-Nicholson (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
Project management<br>
NASA Jet Propulsion Laboratory 329G

Zoe Pierrat (she/her)<br>
[zoe.a.pierrat@jpl.nasa.gov](mailto:zoe.a.pierrat@jpl.nasa.gov)<br>
Algorithm maintenance<br>
NASA Jet Propulsion Laboratory 329G

Claire Villanueva-Weeks (she/her)<br>
[claire.s.villanueva-weeks@jpl.nasa.gov](mailto:claire.s.villanueva-weeks@jpl.nasa.gov)<br>
Code maintenance<br>
NASA Jet Propulsion Laboratory 329G

### Abstract

This software package is a Python implementation of the Breathing Earth System Simulator (BESS) model. It was re-implemented in Python by Gregory Halverson at Jet Propulsion Laboratory based on [MATLAB code](https://www.environment.snu.ac.kr/bess-flux) produced by Youngryel Ryu at Seoul University. The BESS model was designed to quantify global gross primary productivity (GPP) and evapotranspiration (ET) using MODIS with a spatial resolution of 1–5 km and a temporal resolution of 8 days. It couples atmospheric and canopy radiative transfer processes with photosynthesis, stomatal conductance, and transpiration models on sunlit and shaded portions of vegetation and soil. An artificial neural network emulator of Hideki Kobayashi's Forest Light Environmental Simulator (FLiES) radiative transfer model is used to estimate incoming solar radiation. This implementation of BESS was designed to process GPP at fine spatial resolution using surface temperature from the ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) mission and normalized difference vegetation index (NDVI) and albedo from the [Spatial Timeseries for Automated high-Resolution multi-Sensor (STARS) data fusion system](https://github.com/STARS-Data-Fusion). The software was developed as part of a research grant by the NASA Research Opportunities in Space and Earth Sciences (ROSES) program. It was designed for use by the ECOSTRESS mission as a precursor for the Surface Biology and Geology (SBG) mission. However, it may also be useful for general remote sensing and GIS projects in Python. This package can be utilized for remote sensing research in Jupyter notebooks and deployed for operations in data processing pipelines. This software is being released according to the SPD-41 open-science requirements of NASA-funded ROSES projects.

### This software accomplishes the following:

This software package is the python implementation of the Breathing Earth System Simulator (BESS) model of gross primary production (GPP) and evapotranspiration (ET). 

### What are the unique features of the software?

- fine spatial scale estimatation of GPP and ET

### What improvements have been made over existing similar software application?

This adaptation of the BESS remote sensing model was designed to process fine spatial scale images.

### What problems are you trying to solve in the software?

This software makes the BESS model accessible for remote sensing researchers as part of the evapotranspiration modeling capabilities being developed for the ECOSTRESS and SBG missions.

### Does your work relate to current or future NASA (include reimbursable) work that has value to the conduct of aeronautical and space activities?  If so, please explain:

This software package was developed as part of a research grant by the NASA Research Opportunities in Space and Earth Sciences (ROSES) program. This software was designed for use by the ECOSTRESS mission as a precursor for the SBG mission, but it may be useful generally for remote sensing and GIS projects in python.

### What advantages does this software have over existing software?

This software can be utilized for remote sensing research in Jupyter notebooks and deployed for operations in data processing pipelines.

### Are there any known commercial applications? What are they? What else is currently on the market that is similar?

This software is useful for both remote sensing data analysis and building remote sensing data pipelines.

### Is anyone interested in the software? Who? Please list organization names and contact information.

- NASA ROSES
- ECOSTRESS
- SBG

### What are the current hardware and operating system requirements to run the software? (Platform, RAM requirement, special equipment, etc.) 

This software is written entirely in python and intended to be distributed using the pip package manager.

### How has the software performed in tests? Describe further testing if planned. 

This software has been deployed for ECOSTRESS and ET-Toolbox.

### Please identify the customer(s) and sponsors(s) outside of your section that requested and are using your software. 

This package is being released according to the SPD-41 open-science requirements of NASA-funded ROSES projects.

## Installation

Use the pip package manager to install this package

```
pip install .
```
