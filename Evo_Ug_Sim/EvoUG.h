#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <algorithm>

enum class WillingnessUpdateMode {
    BASELINE,               // count==0 -> role average = 0
    PREVIOUS_ROLE_PAYOFF    // count==0 -> last period's average payoff for that role
};

class EvoUG {
public:
    int L;
    int N;
    int T;
    double c, rho, K, gamma, alpha, copy_error;
    std::vector<double> p, q, w, payoffs;
    std::vector<int> prop_counts, resp_counts;
    std::vector<double> prop_payoffs, resp_payoffs;
    std::vector<double> prev_avg_prop, prev_avg_resp;
    WillingnessUpdateMode w_mode;
    std::vector<std::vector<int>> neighs;
    std::vector<std::pair<int, int>> edges;
    std::mt19937_64 rng;
    std::string save_dir, run_id;
    bool save_snapshots;
    bool verbose;
    std::string param_tag;
    double last_mp, last_mq, last_mw, last_mpayoff, last_msr;

    EvoUG(int L_, int T_, double c_, double rho_, double K_, double gamma_, double alpha_, double copy_error_, int seed_, std::string outdir, std::string runid, bool snapshots, WillingnessUpdateMode w_mode_ = WillingnessUpdateMode::BASELINE, bool verbose_ = true);

    void reset_state();
    void run(int record_interval = 1);
    void save_summary(int t, double mp, double mq, double mw, double mpayoff, double msr,
                      double mean_proposer_payoff, double mean_responder_payoff, double mean_R);

private:
    double U01();
    double mean(const std::vector<double>& v);
    double noise();
    void save_vector(std::string name, int t, const std::vector<double>& v);
    int coord_to_idx(int x, int y);
    std::pair<int, int> idx_to_coord(int idx);
    int rand_int(int a, int b);
    double phi_prob_i(double wi, double wj);
};
