# General-Response-CDM
-
This repository contains MATLAB codes for the paper "New Paradigm of Identifiable General-response Cognitive Diagnostic Models: Beyond Categorical Data"
-

The ExpDINA folder contains the code for Algorithm 1 in Section 4 (and Algorithm S.1 in the Supplementary Material), as well as the script used for the ExpDINA simulation studies in Section 5. For each parametric family "parfam", 
- The "generate_X_DINA_parfam" function generates random samples.
- The "get_EM_DINA_parfam" function estimates the model parameters via Algorithm 1.
- The "DINA_main_simulation" script contains a runnable script, where one can implement Algorithm 1 to estimate the ExpDINA parameters. This code was used to generate the results in Section 5.

The ExpACDM folder contains the code for Algorithm 2 in Section 4 (and Algorithm S.2 in the Supplementary Material), as well as the script used for the ExpACDM simulation studies in Section 5. For each parametric family "parfam", 
- The "generate_X_ACDM_parfam" function generates random samples.
- The "get_EM_ACDM" function estimates the model parameters via Algorithm 2. Note that this function is universal and can be applied to all ExpACDMs. The function is default set to Normal-based-ACDMs, but one can modify the commented-out blocks in the function to estimate the Poisson-ACDM. For implementation, one has to specify the exponential family distribution by defining its components as functions (ftn_h, ftn_T, ftn_A), which can be found in the "functions" sub-folder. 
- The "ACDM_main_simulation" script contains a runnable script, where one can implement Algorithm 2 to estimate the ExpACDM parameters. This code was used to generate the results in Section 5.

The TIMSS-2019-Data folder contains the TIMSS 2019 response time dataset and the script used to produce the results in Section 6. TBA
