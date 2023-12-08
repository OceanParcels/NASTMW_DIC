# Disentangling Carbon Concentration Changes Along Pathways of North Atlantic Subtropical Mode Water
Code accompanying the manuscript “Disentangling Carbon Concentration Changes Along Pathways of North Atlantic Subtropical Mode Water” (authors: Daan Reijnders, Dorothee C.E. Bakker, Erik van Sebille).

Code in this repository is maintained by Daan Reijnders (b.j.h.r.reijnders@uu.nl).

## Directory structure
 - `analysis` provides the scripts and notebooks for processing Lagrangian trajectories. It includes scripts to identify enrichment and depletion regimes (dubbed 'events' here), compute total $\Delta$DIC, and create the plots from the manuscript. It also includes an analysis of the Eulerian DIC changes and surface biases. Note that NASTMW is sometimes referred to here as EDW (Eighteen Degree Water). 
- `environments` contains the python environments that were used.
- `jobs` contains bash scripts that execute relevant code to the HPC used for this study. 
- `observations` contains a notebook to plot the tracks from observations
- `preperation` contains scripts and notebooks used in preparation for the simulations. It includes model-data comparison notebooks, which were used for selecting NASTMW criteria. It also includes a NASTMW/EDW mask creation script, a climatology computation script, and a script to precompute the vertical mixing fluxes.
- `simulations` contain the Lagragian simulations script. Simulations are initiated from the `submit_job_EDW_sim.sh` script in the `jobs` directory.
- `tools` creates tools for idenitfying enrichment and depletion regimes ('events'), plotting, and preprocessing $\Delta$DIC timeseries.
