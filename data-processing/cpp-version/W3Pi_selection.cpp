#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include "ROOT/RVec.hxx"
#include "Math/Vector4D.h"

using namespace ROOT;

static const float MINDR2_ANGSEP = 0.5 * 0.5;
static const float MINDR2 = 0.01 * 0.01;
static const float MAXDR2 = 0.25 * 0.25;
static const float MAXISO = 0.5;
static const float PT_CONV = 0.25;
static const float F_CONV = M_PI / 720;
static const float F_CONV2 = (M_PI / 720) * (M_PI / 720);
static const float MIN_PT = 7;
static const float MED_PT = 12;
static const float HIG_PT = 15;
static const float MIN_MASS = 60;
static const float MAX_MASS = 100;

template <typename T>
bool compute_isolation(int16_t idx, T &pts, T &etas, T &phis, int16_t n_cands)
{
    bool is_iso = false;
    float p_sum = 0;
    float eta = etas[idx];
    float phi = phis[idx];

    for (int i=0; i<n_cands; i++)
    {
        if (idx == i) continue;
        float d_eta = eta - etas[i];
        float d_phi = VecOps::DeltaPhi<float>(phi, phis[i]);
        float dr2 = d_eta * d_eta + d_phi * d_phi;

        if ((dr2 >= MINDR2) & (dr2 <= MAXDR2))
        {
            p_sum += pts[i];
        }
    }

    if (p_sum <= (MAXISO * pts[idx])) is_iso = true;
    return is_iso;
}

template <typename T>
bool isolation(int16_t idx, T &pts, T &etas, T &phis, int16_t n_cands, int16_t &cache)
{
    if (cache == 0) cache = compute_isolation(idx, pts, etas, phis, n_cands) ? 1 : 2;

    return (cache == 1);
}

bool deltaR(float eta1, float eta2, float phi1, float phi2)
{
    bool passed = true;    
    float d_eta = eta1 - eta2;
    float d_phi = d_phi = ROOT::VecOps::DeltaPhi<float>(phi1, phi2);
    float dr2 = d_eta * d_eta + d_phi * d_phi;

    if (dr2 < MINDR2_ANGSEP) passed = false;
    return passed;
}

template <typename T>
float tripletmass(const std::array<int, 3> &t, T &pts, T &etas, T &phis)
{
    Math::PtEtaPhiMVector p1 (pts[t[0]], etas[t[0]], phis[t[0]], (float) 0.1396);
    Math::PtEtaPhiMVector p2 (pts[t[1]], etas[t[1]], phis[t[1]], (float) 0.1396);
    Math::PtEtaPhiMVector p3 (pts[t[2]], etas[t[2]], phis[t[2]], (float) 0.1396);

    float mass = (p1 + p2 + p3).M();
    return mass;
}

int main()
{
    const char* filename = "./PuppiSignal_224.dump";

    std::ifstream infile(filename, std::ios_base::binary);

    if (!infile.is_open())
    {
        std::cerr << "Error while opening the file " << filename << std::endl;
        return 1;
    }

    const int32_t n_events = 50000;
    const int32_t n_cands = 224;

    std::cout << "CMD, D, TLAST, TKEEP, TIME_NS" << std::endl;

    for (int i_evt=0; i_evt<n_events; i_evt++)
    {
        int32_t min_pt_count = 0;
        int32_t med_pt_count = 0;
        int32_t hig_pt_count = 0;

        int64_t data;
        VecOps::RVec<float> pts (n_cands);
        VecOps::RVec<float> etas (n_cands);
        VecOps::RVec<float> phis (n_cands);
        VecOps::RVec<int16_t> pdg_ids (n_cands);
        VecOps::RVec<float> charge (n_cands);
        VecOps::RVec<uint16_t> is_filter;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i=0; i<n_cands; i++)
        {
            infile.read(reinterpret_cast<char*>(&data), sizeof(int64_t));

            pts[i] = (((1 << 14) - 1) & data) * PT_CONV;
            etas[i] = ((data << 38) >> 52) * F_CONV;
            phis[i] = ((data << 27) >> 53) * F_CONV;
            pdg_ids[i] = ((1 << 3) - 1) & (data >> 37);
            charge[i] = (pdg_ids[i] >= 4) ? ((pdg_ids[i] == 4) ? -1 : 1) : ((pdg_ids[i] == 2) ? -1 : 1);
        }

        for (int i=0; i<n_cands; i++)
        {
            if ((pdg_ids[i] == 2) || (pdg_ids[i] == 3) || (pdg_ids[i] == 4) || (pdg_ids[i] == 5))
            {
                if (pts[i] >= MIN_PT) 
                {
                    is_filter.push_back(i);

                    if (pts[i] >= MED_PT)
                    {
                        med_pt_count++;

                        if (pts[i] >= HIG_PT)
                        {
                            hig_pt_count++;
                        }
                    }
                }
            }
        }

        min_pt_count = is_filter.size();

        float best_triplet_score = 0;
        float best_triplet_mass = 0;
        std::array<int, 3> best_triplet = { 0 }; 
        VecOps::RVec<int16_t> iso (min_pt_count, 0);

        if ((min_pt_count < 3) || (med_pt_count < 2) || (hig_pt_count < 1))
        {
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

            std::cout << "DATA," << best_triplet[0] << "," << "0," << "-1," << "0" << std::endl;
            std::cout << "DATA," << best_triplet[1] << "," << "0," << "-1," << "0" << std::endl;
            std::cout << "DATA," << best_triplet[2] << "," << "0," << "-1," << "0" << std::endl;
            std::cout << "DATA," << best_triplet_mass << "," << "0," << "-1," << duration.count() << std::endl;
            continue;   
        }

        for (int i1=0; i1<min_pt_count; i1++)
        {
            if (pts[is_filter[i1]] < HIG_PT) continue;
            if (isolation(is_filter[i1], pts, etas, phis, n_cands, iso[i1]) == 0) continue;

            for (int i2=0; i2<min_pt_count; i2++)
            {
                if (i2 == i1 | pts[is_filter[i2]] < MED_PT) continue;
                if (pts[is_filter[i2]] > pts[is_filter[i1]] || (pts[is_filter[i2]] == pts[is_filter[i1]] && i2 < i1)) continue;

                if (!deltaR(etas[is_filter[i1]], etas[is_filter[i2]], phis[is_filter[i1]], phis[is_filter[i2]])) continue;

                for (int i3=0; i3<min_pt_count; i3++)
                {
                    if ((i3 == i1) | (i3 == i2)) continue;
                    if (pts[is_filter[i3]] < MIN_PT) continue;
                    if (pts[is_filter[i3]] > pts[is_filter[i1]] || (pts[is_filter[i3]] == pts[is_filter[i1]] && i3 < i1)) continue;
                    if (pts[is_filter[i3]] > pts[is_filter[i2]] || (pts[is_filter[i3]] == pts[is_filter[i2]] && i3 < i2)) continue;

                    std::array<int, 3> triplet {is_filter[i1], is_filter[i2], is_filter[i3]};

                    if (std::abs(charge[i1] + charge[i2] + charge[i3]) == 1)
                    {
                        auto mass = tripletmass(triplet, pts, etas, phis);

                        if (mass >= MIN_MASS & mass <= MAX_MASS)
                        {
                            if (deltaR(etas[is_filter[i1]], etas[is_filter[i3]], phis[is_filter[i1]], phis[is_filter[i3]]) &
                                deltaR(etas[is_filter[i2]], etas[is_filter[i3]], phis[is_filter[i2]], phis[is_filter[i3]]))
                            {
                                bool isop = isolation(is_filter[i2], pts, etas, phis, n_cands, iso[i2]) && isolation(is_filter[i3], pts, etas, phis, n_cands, iso[i3]);

                                if (isop)
                                {
                                    float pt_sum = pts[is_filter[i1]] + pts[is_filter[i2]] + pts[is_filter[i3]];

                                    if (pt_sum > best_triplet_score)
                                    {
                                        std::copy_n(triplet.begin(), 3, best_triplet.begin());
                                        best_triplet_score = pt_sum;
                                        best_triplet_mass = mass;              
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

        std::cout << "DATA," << best_triplet[0] << "," << "0," << "-1," << "0" << std::endl;
        std::cout << "DATA," << best_triplet[1] << "," << "0," << "-1," << "0" << std::endl;
        std::cout << "DATA," << best_triplet[2] << "," << "0," << "-1," << "0" << std::endl;
        std::cout << "DATA," << best_triplet_mass << "," << "0," << "-1," << duration.count() << std::endl;
    }

    infile.close();
    return 0;
}