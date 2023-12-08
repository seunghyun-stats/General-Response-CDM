# General-Response-CDM

This repository contains MATLAB codes for the paper "New Paradigm of Identifiable General-response Cognitive Diagnostic Models: Beyond Categorical Data"

The ExpDINA folder contains the code for Algorithm 1 in Section 4 (and Algorithm S.1 in the Supplementary Material), as well as the script used for the simulation studies in Section 5. For each parametric family "parfam", 
- The "generate_X_DINA_parfam" function generates random samples.
- The "get_EM_DINA_parfam" function estimates the model parameters via Algorithm 1.
- The "DINA_main_simulation" script contains a runnable script, where one can implement Algorithm 1 to estimate the ExpDINA parameters. This code was used to generate the results in Section 5.

The ExpACDM
