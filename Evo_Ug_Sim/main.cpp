#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <filesystem>
#include "EvoUG.h"
using namespace std;
namespace fs = std::filesystem;

static WillingnessUpdateMode parse_willingness_mode(int argc, char* argv[]) {
    if (argc < 2) {
        return WillingnessUpdateMode::BASELINE;
    }
    string a = argv[1];
    if (a == "baseline" || a == "BASELINE") {
        return WillingnessUpdateMode::BASELINE;
    }
    if (a == "previous_role_payoff" || a == "PREVIOUS_ROLE_PAYOFF") {
        return WillingnessUpdateMode::PREVIOUS_ROLE_PAYOFF;
    }
    cerr << "Unknown willingness mode: " << a << "\n";
    cerr << "Usage: Evo_Ug_Sim.exe [baseline | previous_role_payoff]\n";
    exit(1);
}

static const char* mode_name(WillingnessUpdateMode mode) {
    return (mode == WillingnessUpdateMode::PREVIOUS_ROLE_PAYOFF)
        ? "PREVIOUS_ROLE_PAYOFF"
        : "BASELINE";
}

static string mode_output_root(WillingnessUpdateMode mode) {
    if (mode == WillingnessUpdateMode::PREVIOUS_ROLE_PAYOFF) {
        return "results_previous_role_payoff";
    }
    return "results_cpp";
}

int main(int argc, char* argv[]) {
    WillingnessUpdateMode mode = parse_willingness_mode(argc, argv);

    int L, T, repeats;
    double c, rho, gamma, alpha;

    cout << "Input parameters:\n";
    cout << "willingness mode = " << mode_name(mode) << "\n";
    cout << "L = "; cin >> L;
    cout << "T = "; cin >> T;
    cout << "c = "; cin >> c;
    cout << "rho = "; cin >> rho;
    cout << "gamma = "; cin >> gamma;
    cout << "alpha = "; cin >> alpha;
    cout << "repeats = "; cin >> repeats;

    // BASELINE -> results_cpp/ (original path; existing baseline data is not used by robustness)
    // PREVIOUS_ROLE_PAYOFF -> results_previous_role_payoff/ (separate directory)
    string base_dir = mode_output_root(mode);
    fs::create_directories(base_dir);

    string param_folder_name = "L" + to_string(L) +
        "_T" + to_string(T) +
        "_c" + to_string(c).substr(0, 4) +
        "_rho" + to_string(rho).substr(0, 4) +
        "_gamma" + to_string(gamma).substr(0, 4) +
        "_alpha" + to_string(alpha).substr(0, 4);

    string outdir = base_dir + "/" + param_folder_name;
    fs::create_directories(outdir);

    cout << "Output directory: " << outdir << endl;

    for (int r = 0; r < repeats; r++) {
        int seed = random_device{}();
        string runid = "repeat_" + to_string(r);
        EvoUG sim(L, T, c, rho, 0.1, gamma, alpha, 0.005, seed, outdir, runid, true, mode);
        sim.run();
    }

    cout << "Finished. Results saved to " << outdir << "\n";
    return 0;
}
