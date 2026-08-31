#include <iostream>
#include <random>
#include <map>
#include <fstream>
#include <sstream>
#include <mutex>
#include "EvoUG.h"
#include <nlohmann/json.hpp>
using nlohmann::json;
namespace fs = std::filesystem;
using namespace std;

static std::mutex g_summary_mutex;

EvoUG::EvoUG(int L_, int T_, double c_, double rho_, double K_, double gamma_, double alpha_, double copy_error_, int seed_, string outdir, string runid, bool snapshots, WillingnessUpdateMode w_mode_, bool verbose_)
    : L(L_), T(T_), c(c_), rho(rho_), K(K_), gamma(gamma_), alpha(alpha_), copy_error(copy_error_), w_mode(w_mode_), save_dir(outdir), run_id(runid), save_snapshots(snapshots), verbose(verbose_)
{ 
    N = L * L;
    rng.seed(seed_);

    param_tag = "c=" + to_string(c) +
        "_gamma=" + to_string(gamma) +
        "_rho=" + to_string(rho) +
        "_alpha=" + to_string(alpha);

    p.resize(N);
    q.resize(N);
    w.resize(N);
    payoffs.resize(N);
    prop_counts.resize(N);
    resp_counts.resize(N);
    prop_payoffs.resize(N);
    resp_payoffs.resize(N);
    prev_avg_prop.resize(N, 0.0);
    prev_avg_resp.resize(N, 0.0);

    uniform_real_distribution<double> U(0, 1);
    for (int i = 0; i < N; i++) {
        p[i] = U(rng);
        q[i] = U(rng);
        w[i] = U(rng);
    }

    neighs.resize(N);
    for (int i = 0; i < N; i++) {
        auto [x, y] = idx_to_coord(i);
        for (auto [dx, dy] : vector<pair<int, int>>{ {-1,0},{1,0},{0,-1},{0,1} }) {
            int nx = (x + dx + L) % L;
            int ny = (y + dy + L) % L;
            neighs[i].push_back(coord_to_idx(nx, ny));
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j : neighs[i]) {
            if (i < j) edges.push_back({ i, j });
        }
    }

    // ?????????????
    if (!fs::exists(save_dir)) {
        fs::create_directories(save_dir);
    }

    {
        lock_guard<mutex> lock(g_summary_mutex);
        string summary_file = save_dir + "/summary_" + param_tag + ".csv";
        if (!fs::exists(summary_file)) {
            ofstream summary_f(summary_file);
            if (summary_f.is_open()) {
                summary_f << "repeat_id,gen,mean_p,mean_q,mean_w,mean_payoff,sample_success_rate,mean_proposer_payoff,mean_responder_payoff,mean_R\n";
                summary_f.close();
            }
        }
    }
}

void EvoUG::reset_state() {
    uniform_real_distribution<double> U(0, 1);
    for (int i = 0; i < N; i++) {
        p[i] = U(rng);
        q[i] = U(rng);
        w[i] = U(rng);
    }
    fill(payoffs.begin(), payoffs.end(), 0.0);
    fill(prop_counts.begin(), prop_counts.end(), 0);
    fill(resp_counts.begin(), resp_counts.end(), 0);
    fill(prop_payoffs.begin(), prop_payoffs.end(), 0.0);
    fill(resp_payoffs.begin(), resp_payoffs.end(), 0.0);
    fill(prev_avg_prop.begin(), prev_avg_prop.end(), 0.0);
    fill(prev_avg_resp.begin(), prev_avg_resp.end(), 0.0);
}

double EvoUG::phi_prob_i(double wi, double wj) {
    double exp_i = exp(alpha * wi);
    double exp_j = exp(alpha * wj);
    return exp_i / (exp_i + exp_j);
}
void EvoUG::save_summary(int t, double mp, double mq, double mw, double mpayoff, double msr,
                         double mean_proposer_payoff, double mean_responder_payoff, double mean_R) {
    lock_guard<mutex> lock(g_summary_mutex);
    string summary_file = save_dir + "/summary_" + param_tag + ".csv";
    ofstream summary_f(summary_file, ios::app);
    if (summary_f.is_open()) {
        summary_f << run_id << ","
            << t << ","
            << mp << ","
            << mq << ","
            << mw << ","
            << mpayoff << ","
            << msr << ","
            << mean_proposer_payoff << ","
            << mean_responder_payoff << ","
            << mean_R << "\n";
        summary_f.close();
    }
}

pair<int, int> EvoUG::idx_to_coord(int idx) {
    return { idx / L, idx % L };
}

int EvoUG::coord_to_idx(int x, int y) {
    return ((x % L + L) % L) * L + ((y % L + L) % L);
}

int EvoUG::rand_int(int a, int b) {
    std::uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

double EvoUG::U01() {
    uniform_real_distribution<double> U(0, 1);
    return U(rng);
}

double EvoUG::mean(const vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}

double EvoUG::noise() {
    uniform_real_distribution<double> U(-copy_error, copy_error);
    return U(rng);
}

void EvoUG::save_vector(string name, int t, const vector<double>& v) {
    string tag = "c=" + to_string(c) +
        "_gamma=" + to_string(gamma) +
        "_rho=" + to_string(rho) +
        "_alpha=" + to_string(alpha) +
        "_" + run_id;
    ofstream f(save_dir + "/snapshot_" + name + "_" + tag + "_iter=" + to_string(t) + ".csv");
    for (double x : v) f << x << "\n";
    f.close();
}

void EvoUG::run(int record_interval) {
    vector<map<string, double>> records;
    size_t M = edges.size();
    // t=0 has no previous-period role averages
    fill(prev_avg_prop.begin(), prev_avg_prop.end(), 0.0);
    fill(prev_avg_resp.begin(), prev_avg_resp.end(), 0.0);
    for (int t = 0; t < T; t++) {
        fill(payoffs.begin(), payoffs.end(), 0.0);
        fill(prop_counts.begin(), prop_counts.end(), 0);
        fill(resp_counts.begin(), resp_counts.end(), 0);
        fill(prop_payoffs.begin(), prop_payoffs.end(), 0.0);
        fill(resp_payoffs.begin(), resp_payoffs.end(), 0.0);
        int success = 0;

        // -------- ?????? --------
        for (auto& e : edges) {
            int i = e.first;
            int j = e.second;
            // ????????
            double prob_i = phi_prob_i(w[i], w[j]);
            int proposer, responder;
            if (U01() < prob_i) {
                proposer = i;
                responder = j;
            }
            else {
                proposer = j;
                responder = i;
            }

            double offer = p[proposer];
            double thresh = q[responder];

            if (offer >= thresh) {
                // ??????
                double payoff_prop = 1.0 - offer - c + rho * c;
                double payoff_resp = offer;
                success++;

                payoffs[proposer] += payoff_prop;
                payoffs[responder] += payoff_resp;

                // ?????????????????
                prop_payoffs[proposer] += payoff_prop;
                prop_counts[proposer] += 1;

                // ????????????????
                resp_payoffs[responder] += payoff_resp;
                resp_counts[responder] += 1;
            }
            else {
                // ???????
                double payoff_prop = -c;
                double payoff_resp = 0.0;

                payoffs[proposer] += payoff_prop;
                payoffs[responder] += payoff_resp;

                // ?????????????????
                prop_payoffs[proposer] += payoff_prop;
                prop_counts[proposer] += 1;

                // ????????????????
                resp_payoffs[responder] += payoff_resp;
                resp_counts[responder] += 1;
            }
        }

        // -------- ??????? (??????) --------
        int updates_per_gen = N;  // ???????????????????N??????

        for (int update_count = 0; update_count < updates_per_gen; update_count++) {
            // 1. ?????????????i
            int i = rand_int(0, N - 1);

            // 2. ??i???????????????????j
            int num_neighbors = neighs[i].size();
            int random_neighbor_idx = rand_int(0, num_neighbors - 1);
            int j = neighs[i][random_neighbor_idx];

            // 3. ???Fermi????????????????j
            double fi = payoffs[i];
            double fj = payoffs[j];
            double fermi = 1.0 / (1.0 + exp((fi - fj) / K));

            // 4. ????????????????
            if (U01() < fermi) {
                // ??????j
                p[i] = clamp(p[j] + noise(), 0.0, 1.0);
                q[i] = clamp(q[j] + noise(), 0.0, 1.0);
            }
            else {
                // ????????????????
                p[i] = clamp(p[i] + noise(), 0.0, 1.0);
                q[i] = clamp(q[i] + noise(), 0.0, 1.0);
            }
        }
      

        // -------- ??????? --------
        vector<double> avg_prop(N, 0.0);
        vector<double> avg_resp(N, 0.0);

        for (int i = 0; i < N; i++) {
            if (prop_counts[i] > 0) {
                avg_prop[i] = prop_payoffs[i] / prop_counts[i];
            } else if (w_mode == WillingnessUpdateMode::PREVIOUS_ROLE_PAYOFF) {
                avg_prop[i] = prev_avg_prop[i];
            }

            if (resp_counts[i] > 0) {
                avg_resp[i] = resp_payoffs[i] / resp_counts[i];
            } else if (w_mode == WillingnessUpdateMode::PREVIOUS_ROLE_PAYOFF) {
                avg_resp[i] = prev_avg_resp[i];
            }
        }

        double sum_R = 0.0;
        for (int i = 0; i < N; i++) {
            double R = avg_prop[i] - avg_resp[i];
            sum_R += R;
            w[i] = clamp(w[i] + gamma * R, 0.0, 1.0);
        }

        if (w_mode == WillingnessUpdateMode::PREVIOUS_ROLE_PAYOFF) {
            prev_avg_prop = avg_prop;
            prev_avg_resp = avg_resp;
        }

        double total_prop_payoff = 0.0;
        double total_resp_payoff = 0.0;
        for (int i = 0; i < N; i++) {
            total_prop_payoff += prop_payoffs[i];
            total_resp_payoff += resp_payoffs[i];
        }
        double mean_proposer_payoff = total_prop_payoff / static_cast<double>(M);
        double mean_responder_payoff = total_resp_payoff / static_cast<double>(M);
        double mean_R = sum_R / static_cast<double>(N);

        double mp = mean(p);
        double mq = mean(q);
        double mw = mean(w);
        double msr = (double)success / M;
        double mpayoff = mean(payoffs);

        if (t == T - 1) {
            last_mp = mp;
            last_mq = mq;
            last_mw = mw;
            last_mpayoff = mpayoff;
            last_msr = msr;

            // ????????????
            save_summary(t, mp, mq, mw, mpayoff, msr,
                         mean_proposer_payoff, mean_responder_payoff, mean_R);
        }


        records.push_back({
            {"gen", (double)t},
            {"mean_p", mp},
            {"mean_q", mq},
            {"mean_w", mw},
            {"mean_payoff", mpayoff},
            {"sample_success_rate", msr},
            {"mean_proposer_payoff", mean_proposer_payoff},
            {"mean_responder_payoff", mean_responder_payoff},
            {"mean_R", mean_R}
        });

        if (verbose && t % record_interval == 0) {
            cout << "[Gen " << t << "] mean_p=" << mp
                << " mean_q=" << mq
                << " mean_w=" << mw
                << " success_rate=" << msr
                << " mean_payoff=" << mpayoff << endl;
        }

        if (save_snapshots && (t == 0 || t == 9 || t == 99 || t == 999 || t == 9999 || t == 99999)) {
            save_vector("p", t, p);
            save_vector("q", t, q);
            save_vector("payoff", t, payoffs);
            save_vector("w", t, w);
        }
    }

    if (verbose) {
        cout << "Repeat " << run_id << " finished. Last generation: mean_p=" << last_mp
            << ", mean_q=" << last_mq << ", mean_w=" << last_mw
            << ", success_rate=" << last_msr << endl;
    }

    string tag = "c=" + to_string(c) +
        "_gamma=" + to_string(gamma) +
        "_rho=" + to_string(rho) +
        "_alpha=" + to_string(alpha) +
        "_" + run_id;

    json j;
    j["L"] = L;
    j["T"] = T;
    j["c"] = c;
    j["rho"] = rho;
    j["gamma"] = gamma;
    j["alpha"] = alpha;
    j["copy_error"] = copy_error;
    j["repeat"] = run_id;
    j["willingness_update_mode"] = (w_mode == WillingnessUpdateMode::PREVIOUS_ROLE_PAYOFF)
        ? "PREVIOUS_ROLE_PAYOFF" : "BASELINE";

    ofstream param_file(save_dir + "/params_" + tag + ".json");
    param_file << j.dump(2);
    param_file.close();

    string ts_name = save_dir + "/c=" + to_string(c) +
        "_gamma=" + to_string(gamma) +
        "_rho=" + to_string(rho) +
        "_alpha=" + to_string(alpha) +
        "_" + run_id + "_timeseries.csv";

    ofstream ts_file(ts_name);
    ts_file << "gen,mean_p,mean_q,mean_w,mean_payoff,sample_success_rate,mean_proposer_payoff,mean_responder_payoff,mean_R\n";

    for (auto& rec : records) {
        ts_file << rec["gen"] << ","
            << rec["mean_p"] << ","
            << rec["mean_q"] << ","
            << rec["mean_w"] << ","
            << rec["mean_payoff"] << ","
            << rec["sample_success_rate"] << ","
            << rec["mean_proposer_payoff"] << ","
            << rec["mean_responder_payoff"] << ","
            << rec["mean_R"] << "\n";
    }

    ts_file.close();
}
