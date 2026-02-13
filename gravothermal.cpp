// ------------------------------------------------------------
// Gravothermal Fluid Simulation of Dark Matter Halos
//
// Author: Dhruv Hukkeri
// Developed as part of Master's thesis at IISER Pune
//
// Description:
// This program evolves dark matter halos treated as 
// self-gravitating fluids by solving the coupled equations of:
//
// 1. Hydrostatic equilibrium
// 2. Energy transport (luminosity conduction)
// 3. Internal energy evolution
//
// Input:
//  - Halo profiles from CSV file
//
// Output:
//  - Time evolution of radius, density, internal energy, luminosity
//
// Dependencies:
//  - Eigen (linear algebra)
//  - OpenMP (parallel support)
// ------------------------------------------------------------

#define EIGEN_DONT_PARALLELIZE
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <map>
#include <omp.h>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
// Structure to store radial halo profiles and properties
struct HaloProfile {
    int halo_index;          // Unique halo identifier
    double M, c;             // Halo mass and concentration
    double rho0, rs;         // Scale density and scale radius

    // Radial profiles
    std::vector<double> r;        // Radius bins
    std::vector<double> rho_r;    // Density profile
    std::vector<double> m_r;      // Enclosed mass
    std::vector<double> v2_r;     // Velocity dispersion squared
    std::vector<double> L_r;      // Luminosity profile
};
// Reads a CSV row and extracts halo profile values
// Returns true if parsing is successful
bool parseRow(const std::string& line, int& halo_index, double& M, double& c, double& rho0, double& rs,
    double& r, double& rho, double& m, double& v2, double& L) {
std::stringstream ss(line);
std::string item;
std::vector<std::string> tokens;

while (std::getline(ss, item, ',')) {
tokens.push_back(item);
}
if (tokens.size() < 10) return false;

halo_index = std::stoi(tokens[0]);
M = std::stod(tokens[1]);
c = std::stod(tokens[2]);
rho0 = std::stod(tokens[3]);
rs = std::stod(tokens[4]);
r = std::stod(tokens[5]);
rho = std::stod(tokens[6]);
m = std::stod(tokens[7]);
v2 = std::stod(tokens[8]);
L = std::stod(tokens[9]);

return true;
}


bool contains_nan(const Eigen::VectorXd& vec) {
    for (int i = 0; i < vec.size(); ++i) {
        if (std::isnan(vec(i))) return true;
    }
    return false;
}
// Tridiagonal Matrix Algorithm (Thomas Algorithm)
// Solves Ax = d where A is tridiagonal
// Used to solve discretized hydrostatic equilibrium equations
Eigen::VectorXd TDMAsolver(const Eigen::VectorXd& a, const Eigen::VectorXd& b, 
    const Eigen::VectorXd& c, const Eigen::VectorXd& d) {
int n = d.size();
Eigen::VectorXd b_mod = b, d_mod = d, x(n);

// Forward Elimination
for (int i = 1; i < n; ++i) {
double m = a(i-1) / b_mod(i-1);
b_mod(i) -= m * c(i-1);
d_mod(i) -= m * d_mod(i-1);
}

// Backward Substitution
x(n-1) = d_mod(n-1) / b_mod(n-1);
for (int i = n-2; i >= 0; --i) {
x(i) = (d_mod(i) - c(i) * x(i+1)) / b_mod(i);
}

return x;
}
// Bulk cooling term for dissipative dark matter
// Returns cooling rate per unit density
double cbulkoverrho (double myrho, double myv, double myvlow, double myvhigh, double mya, double mysigmain) //Bulk cooling 
{
    if (myv <= myvhigh)
    return mya * myrho * mysigmain * pow(myvlow,2.0) * myv *(1.0 + pow(myvlow,2.0)/pow(myv,2.0)) * exp(-pow(myvlow,2.0)/pow(myv,2.0));
    else 
        return 0.0;
}

// ------------------------------------------------------------
// Main gravothermal evolution routine
//
// Steps performed:
// 1. Initialize halo profiles
// 2. Compute energy evolution and adaptive timestep
// 3. Solve hydrostatic adjustment using TDMA solver
// 4. Update thermodynamic quantities
// 5. Compute luminosity and cooling
// 6. Save output periodically
// ------------------------------------------------------------

void run_gravothermal_simulation(const HaloProfile& halo) {
    int N = halo.r.size();
    Eigen::VectorXd r(N), rho(N), v2(N), L(N), v(N), u(N), m(N);

    for (int i = 0; i < N; ++i) {
        r(i) = halo.r[i];
        rho(i) = halo.rho_r[i];
        v2(i) = halo.v2_r[i];
        L(i) = halo.L_r[i];
        v(i) = std::sqrt(v2(i));
        u(i) = 1.5 * v2(i);
        m(i) = halo.m_r[i];
    }

    std::cout << "Running halo " << halo.halo_index << " with rs=" << halo.rs << ", rho0=" << halo.rho0 << std::endl;
    std::cout << "[INFO] Starting simulation for halo " << halo.halo_index << " with N=" << N << std::endl;

    int totalstep = 20000000;
    int savestep = 1000;
    double totaltime = 0;
    double rho0 = halo.rho0; 
    double rs = halo.rs;

    std::string filename = "s1.0_z0.7_vl600_sig5.6_C0.75_M14_18/halo" + std::to_string(halo.halo_index) + ".csv";
    std::ofstream outfile(filename);
    outfile << "tstep,totaltime,r,rho,u,L\n";

    double sigma = 5.26*1e-2;
    double a1 = std::sqrt(16/M_PI);
    double b1 = 1.38;
    double C = 0.75;
    double r_min = 0.01;
    double r_max = 100;
    double vlow = 600/rs*pow(5.4*pow(10,-5)*rho0,0.5);
    double vhigh = 100;
    Eigen::VectorXd r2(N), p(N), dr(N), drho(N), a(N), b(N), c(N), d(N), du(N), dp(N), rhon(N), vn(N), Ln(N), alist(N);


    for (int tstep = 1; tstep <= totalstep; ++tstep) {
        Eigen::VectorXd du_co(N);
        Eigen::VectorXd t_r(N);
        du_co(0) = -(L(0)/m(0) + cbulkoverrho(rho(0), v(0), vlow, vhigh, a1, sigma)) /u(0);
        for (int i = 1; i < N; i++) {
            du_co(i) = -((L(i)-L(i-1))/(m(i) - m(i-1)) + cbulkoverrho(rho(i), v(i), vlow, vhigh, a1, sigma)) /u(i);
        }
        double dt; 
        if (tstep <= 100000000) { // Includes the case tstep == 100000
            dt = 1e-3 / du_co.array().abs().maxCoeff();          // Adaptive timestep based on maximum cooling rate // Ensures numerical stability
        }
        totaltime += dt;
        
        for (int i = 0; i < N; i++) {
            du(i) = du_co(i)*u(i)*dt;
        }

        for (int i = 0; i < N; i++) {
            u(i) = u(i) + du(i);
            v(i) = std::sqrt(2.0/3.0*u(i));
            p(i) = 2.0/3.0*rho(i)*u(i);
            // alist(i) = 2.0/3.0*std::pow(rho(i), -2.0/3.0)*u(i);
        }

        for (int hyt = 0; hyt < 5; hyt++) {
            for (int i = 1; i < N - 1; ++i) {
                a(i) = (-m(i) * (rho(i+1) + rho(i)) - (20 * p(i) * std::pow(r(i), 2) * std::pow(r(i-1), 2)) / (std::pow(r(i), 3) - std::pow(r(i-1), 3)) + 3 * m(i) * rho(i) * std::pow(r(i-1), 2) * (r(i+1) - r(i-1))/ (std::pow(r(i), 3) - std::pow(r(i-1), 3)));
                b(i) = 8 * (p(i+1) - p(i)) * r(i) + 20 * std::pow(r(i), 4) * (p(i+1) / (std::pow(r(i+1), 3) - std::pow(r(i), 3)) + p(i) / (std::pow(r(i), 3) - std::pow(r(i-1), 3))) + 3 * m(i) * (r(i+1) - r(i-1)) * std::pow(r(i),2) * (rho(i+1) / (std::pow(r(i+1), 3) - std::pow(r(i), 3)) - rho(i) / (std::pow(r(i), 3) - std::pow(r(i-1), 3)));
                c(i) = m(i) * (rho(i+1) + rho(i)) - (20 * std::pow(r(i), 2) * p(i+1) * std::pow(r(i+1), 2)) / (std::pow(r(i+1), 3) - std::pow(r(i), 3)) - 3 * rho(i+1) * m(i) * (r(i+1) - r(i-1)) * std::pow(r(i+1), 2)/(std::pow(r(i+1), 3) - std::pow(r(i), 3));
                d(i) = -4 * (p(i+1) - p(i)) * std::pow(r(i), 2) - m(i) * (r(i+1) - r(i-1)) * (rho(i+1) + rho(i));
            }
            
            b(0) = 8.0 * r(0) * (p(1) - p(0)) + 20.0 * std::pow(r(0), 4) * ((p(0) / std::pow(r(0), 3) + p(1) / (std::pow(r(1), 3) - std::pow(r(0), 3)))) + 3.0 * m(0) * r(1) * pow(r(0), 2) * (-rho(0) / pow(r(0), 3) + rho(1) / (pow(r(1), 3) - pow(r(0), 3)));
            c(0) = m(0) * (rho(0) + rho(1)) - 20.0 * std::pow(r(0), 2) * p(1) * std::pow(r(1), 2) / (std::pow(r(1), 3) - std::pow(r(0), 3)) - 3.0 * m(0) * rho(1) * pow(r(1), 3) / (pow(r(1), 3) - pow(r(0), 3));
            d(0) = -4 * (p(1) - p(0)) * std::pow(r(0), 2) - m(0) * r(1) * (rho(1) + rho(0));
            
            a(N-2) = (-m(N-2) * (rho(N-1) + rho(N-2)) - (20 * p(N-2) * std::pow(r(N-2), 2) * std::pow(r(N-3), 2)) / (std::pow(r(N-2), 3) - std::pow(r(N-3), 3)) + 3 * m(N-2) * rho(N-2) * std::pow(r(N-3), 2) * (r(N-1) - r(N-3))/ (std::pow(r(N-2), 3) - std::pow(r(N-3), 3)));
            b(N-2) = 8 * (p(N-1) - p(N-2)) * r(N-2) + 20 * std::pow(r(N-2), 4) * (p(N-1) / (std::pow(r(N-1), 3) - std::pow(r(N-2), 3)) + p(N-2) / (std::pow(r(N-2), 3) - std::pow(r(N-3), 3))) + 3 * m(N-2) * (r(N-1) - r(N-3)) * std::pow(r(N-2), 2) * (rho(N-1) / (std::pow(r(N-1), 3) - std::pow(r(N-2), 3)) -rho(N-2)/(std::pow(r(N-2), 3) - std::pow(r(N-3), 3)));
            d(N-2) = -4 * (p(N-1) - p(N-2)) * std::pow(r(N-2), 2) - m(N-2) * (r(N-1) - r(N-3)) * (rho(N-1) + rho(N-2));
            // Iterative hydrostatic adjustment:
            // Solve tridiagonal system to update radial structure
            dr.segment(0, N-2) = TDMAsolver(a.segment(1, N-2), b.segment(0, N-2), c.segment(0, N-3), d.segment(0, N-2));

            drho(0) = -3.0 * rho(0) * std::pow(r(0), 2) * dr(0) / std::pow(r(0), 3);
            dp(0) = -5.0 * p(0) * std::pow(r(0), 2) * dr(0) / std::pow(r(0), 3);

            for (int i = 1; i < N - 1; i++) {
                drho(i) = -3.0 * rho(i) * (std::pow(r(i), 2) * dr(i) - std::pow(r(i-1), 2) * dr(i-1)) / (std::pow(r(i), 3) - std::pow(r(i-1), 3));
                dp(i) = -5.0 * p(i) * (std::pow(r(i), 2) * dr(i) - std::pow(r(i-1), 2) * dr(i-1)) / (std::pow(r(i), 3) - std::pow(r(i-1), 3));
            }
            
            for (int i = 0; i < N - 1; i++) {
                rho(i) += drho(i);
                p(i) += dp(i);
                r(i) += dr(i);
            }
        }
        for (int i = 0; i < N; i++) {
            u(i) = 1.5*p(i)/rho(i);  // Compute rate of change of internal energy due to conduction and cooling
            v(i) = std::sqrt(2.0/3.0*u(i));
        }
        

        
        for (int i = 1; i < N-1; i++) {
            L(i) = -std::pow(r(i), 2)*(2 * a1 * b1 * C * sigma* (rho(i) + rho(i+1)) * (std::pow(v(i), 3) + std::pow(v(i+1), 3))) / (a1 * C * std::pow(sigma, 2) * (rho(i) + rho(i+1)) * (std::pow(v(i), 2) + std::pow(v(i+1), 2)) + 4 * b1)*(u(i+1) - u(i))/(r(i+1) - r(i-1));
        }
        L(0) = -std::pow(r(0), 2) * (2 * a1 * b1 * C * sigma * (rho(0) + rho(1)) * (std::pow(v(0), 3) + std::pow(v(1), 3))) / (a1 * C * std::pow(sigma, 2) * (rho(0) + rho(1)) * (std::pow(v(0), 2) + std::pow(v(1), 2)) + 4 * b1) *(u(1) - u(0)) / (r(1));

        if (contains_nan(u) || contains_nan(rho) || contains_nan(r)) {
            std::cerr << "NaN detected at step " << tstep << " for halo " << halo.halo_index << ". Stopping simulation." << std::endl;
            
            outfile.flush();
            outfile.close();  
            return;           
        }

        if (tstep % savestep == 0) {
            outfile << tstep << "," << totaltime << ",";

            for (int i = 0; i < N; i++) {
                outfile << r(i);
                if (i < N - 1) outfile << ";";
            }
            outfile << ",";

            for (int i = 0; i < N; i++) {
                outfile << rho(i);
                if (i < N - 1) outfile << ";";
            }
            outfile << ",";

            for (int i = 0; i < N; i++) {
                outfile << u(i);
                if (i < N - 1) outfile << ";";
            }
            outfile << ",";

            for (int i = 0; i < N; i++) {
                outfile << L(i);
                if (i < N - 1) outfile << ";";
            }
            outfile << "\n";
        }
    }

    outfile.close();
}

// ------------------------------------------------------------
// Main program
//
// Usage:
// ./gravothermal <halo_index>
//
// Loads halo profile from CSV file and runs simulation
// ------------------------------------------------------------


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <halo_index>" << std::endl;
        return 1;
    }

    int target_index = std::stoi(argv[1]);

    std::ifstream file("sampled_redshift_0.7006802721088436.csv");
    std::string line;
    std::getline(file, line); // skip header

    HaloProfile halo;
    bool found = false;

    while (std::getline(file, line)) {
        int index;
        double M, c, rho0, rs, r, rho, v2, L, m;
        if (!parseRow(line, index, M, c, rho0, rs, r, rho, m, v2, L)) continue;

        if (index == target_index) {
            if (!found) {
                halo = HaloProfile{index, M, c, rho0, rs, {}, {}, {}, {}, {}};
                found = true;
            }
            halo.r.push_back(r);
            halo.rho_r.push_back(rho);
            halo.m_r.push_back(m);
            halo.v2_r.push_back(v2);
            halo.L_r.push_back(L);
        }
    }

    if (!found) {
        std::cerr << "Halo index " << target_index << " not found in input file." << std::endl;
        return 2;
    }

    run_gravothermal_simulation(halo);
    return 0;
}
