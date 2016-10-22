//
// Created by antoinewdg on 10/19/16.
//

#ifndef COPY_MOVE_DETECTOR_PATCH_DISTANCES_H
#define COPY_MOVE_DETECTOR_PATCH_DISTANCES_H

#include <limits>

#include "utils.h"

template<int P>
inline int euclidian_distance(const Mat_<Vec3b> &p, const Mat_<Vec3b> &q, int max_d) {
    int d = 0;
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < P; j++) {
            int a = int(p(i, j)[0]) - q(i, j)[0];
            int b = int(p(i, j)[1]) - q(i, j)[1];
            int c = int(p(i, j)[2]) - q(i, j)[2];
            d += a * a + b * b + c * c;
            if (d > max_d) {
                return d;
            }
        }

    }

    return d;
}

template<int P>
inline int l1_distance(const Mat_<Vec3b> &p, const Mat_<Vec3b> &q, int max_d) {
    int d = 0;
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < P; j++) {
            d += std::abs<int>(int(p(i, j)[0]) - q(i, j)[0]);
            d += std::abs<int>(int(p(i, j)[1]) - q(i, j)[1]);
            d += std::abs<int>(int(p(i, j)[2]) - q(i, j)[2]);
            if (d > max_d) {
                return d;
            }
        }

    }

    return d;
}



template<int P>
Rect get_patch_rect(const Vec2i &p) {
    return Rect(p[1] - (P/2), p[0] - (P/2), P, P);
}


template<int P>
class EuclidianPatchDistanceRGB {
public:
    static constexpr int patch_size = P;

    void initialize(const Mat_<Vec3b> &s, const Mat_<Vec3b> &t) {
        m_s = s;
        m_t = t;
    }

    inline int operator()(const Vec2i &p, const Vec2i &q,
                          int max_d = std::numeric_limits<int>::max()) {
        return euclidian_distance<P>(m_s(get_patch_rect<P>(p)),
                                     m_t(get_patch_rect<P>(q)),
                                     max_d);
    }


private:
    Mat_<Vec3b> m_s, m_t;
};

template<int P>
class EuclidianPatchDistanceLab {
public:
    static constexpr int patch_size = P;

    void initialize(const Mat_<Vec3b> &s, const Mat_<Vec3b> &t) {
        cv::cvtColor(s, m_s, cv::COLOR_BGR2Lab);
        cv::cvtColor(t, m_t, cv::COLOR_BGR2Lab);
    }

    inline int operator()(const Vec2i &p, const Vec2i &q,
                          int max_d = std::numeric_limits<int>::max()) {
        return euclidian_distance<P>(m_s(get_patch_rect<P>(p)),
                                     m_t(get_patch_rect<P>(q)),
                                     max_d);
    }


private:
    Mat_<Vec3b> m_s, m_t;
};


#endif //COPY_MOVE_DETECTOR_PATCH_DISTANCES_H
