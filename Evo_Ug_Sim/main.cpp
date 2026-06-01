#include <iostream>
#include <string>
#include <random>
#include <filesystem>
#include "EvoUG.h"
using namespace std;
namespace fs = std::filesystem;

int main() {
    int L, T, repeats;
    double c, rho, gamma, alpha;

    cout << "输入参数开始实验：\n";
    cout << "L（格子大小）="; cin >> L;
    cout << "T（迭代代数）="; cin >> T;
    cout << "c（成本参数）="; cin >> c;
    cout << "rho（返还比例）="; cin >> rho;
    cout << "gamma（学习率）="; cin >> gamma;
    cout << "alpha（选择强度）="; cin >> alpha;
    cout << "实验重复次数 repeats="; cin >> repeats;

    // 创建基础结果目录
    string base_dir = "results_cpp";
    fs::create_directories(base_dir);

    // 创建参数文件夹名称
    string param_folder_name = "L" + to_string(L) +
        "_T" + to_string(T) +
        "_c" + to_string(c).substr(0, 4) +
        "_rho" + to_string(rho).substr(0, 4) +
        "_gamma" + to_string(gamma).substr(0, 4) +
        "_alpha" + to_string(alpha).substr(0, 4);

    // 完整的输出目录
    string outdir = base_dir + "/" + param_folder_name;
    fs::create_directories(outdir);  // 确保参数文件夹存在

    cout << "输出目录: " << outdir << endl;

    for (int r = 0; r < repeats; r++) {
        int seed = random_device{}();
        string runid = "repeat_" + to_string(r);
        // 参数说明: L, T, c, rho, K, gamma, alpha, copy_error, seed, outdir, runid, snapshots
        EvoUG sim(L, T, c, rho, 0.1, gamma, alpha, 0.005, seed, outdir, runid, true);
        sim.run();
    }

    cout << "实验完成，结果已保存到 " << outdir << " 目录\n";
    return 0;

}
