#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <algorithm>

class EvoUG {
public:
    int L;
    int N;
    int T;
    double c, rho, K, gamma, alpha, copy_error;
    std::vector<double> p, q, w, payoffs;
    std::vector<int> prop_counts, resp_counts;  // 记录作为提议者和回应者的次数
    std::vector<double> prop_payoffs, resp_payoffs;  // 记录作为提议者和回应者的收益
    std::vector<std::vector<int>> neighs;
    std::vector<std::pair<int, int>> edges;
    std::mt19937_64 rng;
    std::string save_dir, run_id;
    bool save_snapshots;
    std::string param_tag;
    double last_mp, last_mq, last_mw, last_mpayoff, last_msr;

    EvoUG(int L_, int T_, double c_, double rho_, double K_, double gamma_, double alpha_, double copy_error_, int seed_, std::string outdir, std::string runid, bool snapshots);

    void reset_state();
    //double phi_prob_i(double wi, double wj);
    void run(int record_interval = 1);
    void save_summary(int t, double mp, double mq, double mw, double mpayoff, double msr);

private:
    double U01();
    double mean(const std::vector<double>& v);
    double noise();
    void save_vector(std::string name, int t, const std::vector<double>& v);
    int coord_to_idx(int x, int y);
    std::pair<int, int> idx_to_coord(int idx);
    //std::mt19937 rng;  
    int rand_int(int a, int b);
    double phi_prob_i(double wi, double wj);
};
