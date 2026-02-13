# Gravothermal Simulations of Dark Matter Halos

This repository contains a C++ implementation of gravothermal fluid simulations used to study the evolution of self-interacting and dissipative dark matter halos.

The code evolves spherically symmetric halos by solving the coupled equations of mass conservation, hydrostatic equilibrium, and energy transport, and tracks the time evolution of density, internal energy, and luminosity profiles starting from supplied initial conditions.

This code was developed as part of my Master's thesis at IISER Pune under the supervision of Dr. Susmita Adhikari.

---

## Repository Structure

src/        C++ source code for the gravothermal solver  
data/       Example input files containing halo initial conditions  
examples/   Example output files from a sample run  

---

## Requirements

- C++17 or later  
- Eigen library  
- OpenMP  

---

## Compilation

Example using g++:

```bash
g++ src/gravothermal.cpp -O2 -fopenmp -I /path/to/eigen -o gravothermal
```

Replace `/path/to/eigen` with the location of your Eigen installation.

---

## Running the Code

The program takes a halo index as input:

```bash
./gravothermal 0
```

The code reads halo profiles from:

```
data/sample_input.csv
```

and writes outputs to the directory specified in the code.

---

## Input Format

The input CSV file contains radial profiles for halos. Each row corresponds to a radial bin.

Columns:

- halo_ID — Halo identifier  
- M — Halo mass  
- c — Concentration  
- rho0 — Scale density  
- rs — Scale radius  
- radius — Radial coordinate  
- density — Density profile  
- enclosed_mass — Mass within radius  
- velocity_dispersion_sq — Velocity dispersion squared  
- luminosity — Conductive luminosity  

---

## Output

The simulation outputs time evolution of:

- Radius  
- Density  
- Internal energy  
- Luminosity  

An example output file is provided in the `examples/` directory.

---

## Description of Method

The simulation treats dark matter halos as self-gravitating fluids and evolves their structure using a finite-difference scheme in radius and time. Hydrostatic adjustment is performed using a tridiagonal matrix solver, and the timestep is chosen adaptively based on local energy evolution rates.

---

## Author

Dhruv Hukkeri  
BS-MS Physics, IISER Pune  

