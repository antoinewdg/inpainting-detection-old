//
// Created by antoinewdg on 10/16/16.
//

#ifndef COPY_MOVE_DETECTOR_PATCH_MATCHER_H
#define COPY_MOVE_DETECTOR_PATCH_MATCHER_H

#define PI 3.14159265

#include <ctime>
#include <random>
#include <limits>
#include "utils.h"

/**
 * Implementation of the patch match algorithm.
 */
template<class DistanceFunction>
class PatchMatcher {
public:
    /**
     * Initialize the matcher
     * @param s source image
     * @param s_mask mask describing what pixels in s should be matched
     * @param t target image
     * @param t_mask mask describing what pixels in t should me targeted
     * @return
     */
    PatchMatcher(Mat_<Vec3b> s, Mat_<float> s_mask,
                 Mat_<Vec3b> t, Mat_<float> t_mask,
                 DistanceFunction distance_function);


    /**
     * Intitialize the nnf with random values.
     */
    void initialize_nnf();

    /**
     * Perform random search for a source patch.
     * @param p position of the source patch
     */
    void random_search(const Vec2i &p);

    /**
     * Perform propagation for a source patch.
     *
     * @param p position of the source patch
     * @param neighbors positions of the patches the source patch
     * should be compared to
     */
    void propagate(const Vec2i &p, const array<Vec2i, 2> &neighbors);

    /**
     * Perform an iteration with propagation right and down.
     */
    void iterate_rd();

    /**
     * Perform an iteration with propagation left and up.
     */
    void iterate_lu();

    /**
     * Perform n iterations.
     *
     * The direction of propagation changes at each iteration.
     *
     * @param n
     */
    void iterate_n_times(int n);

    /**
     * Represent the nnf with an images.
     *
     * The image is built with HSV values. Hue represents
     * the direction of the nnf, saturation and value
     * he intensity.
     * @return
     */
    Mat_<Vec3b> nnf_to_image() const;

    /**
     * Represents the patch distance for each source patch.
     *
     * At the first call, the highest distance is stored and
     * mapped to uchar 255. Subsequent calls use the stored
     * value.
     * @return
     */
    Mat_<uchar> distance_map();

    /**
     * First attempt at building the images from the patch matches
     *
     * Does not work, and should be moved.
     *
     * @return
     */
    Mat_<Vec3b> build_s_with_patches_from_t() const;

    /**
     * Get nnf value at position p.
     * @param p
     * @return
     */
    inline Vec2i &nnf(const Vec2i &p) {
        return m_nnf(p[0], p[1]);
    }

    /**
     * Get nnf value at position p.
     * @param p
     * @return
     */
    inline const Vec2i &nnf(const Vec2i &p) const {
        return m_nnf(p[0], p[1]);
    }

    /**
     * Get the PxP rectangle representing a patch.
     * @param p position of the patch
     * @return
     */
    inline Rect get_patch_rect(const Vec2i &p) const {
        return Rect(p[1], p[0], P, P);
    }

    int _propagate_distance(const Vec2i &p, const Vec2i &neighbor, int max_d);

    static constexpr int P = DistanceFunction::patch_size;

    Mat_<Vec3b> s_rgb, t_rgb;
    Mat_<float> s_mask, t_mask;
    Mat_<float> s_patches_mask, t_patches_mask;
    Mat_<Vec2i> m_nnf;
    vector<Vec2i> s_pixels, t_pixels, s_patches, t_patches;
    std::default_random_engine generator;
    int current_max_distance;
    DistanceFunction patch_distance;

};

vector<Vec2i> _enumerate_pixels(Mat_<float> mask) {
    vector<Vec2i> pixels;
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask(i, j) != 0) {
                pixels.push_back(Vec2i(i, j));
            }
        }
    }
    return pixels;
}


template<int P>
Mat_<float> _get_patches_mask(Mat_<float> mask) {
    Mat_<float> p_mask(mask.rows - P + 1, mask.cols - P + 1, 0.f);
    for (int i = 0; i < mask.rows - P + 1; i++) {
        for (int j = 0; j < mask.cols - P + 1; j++) {
            if (cv::countNonZero(mask(Rect(j, i, P, P))) > 0) {
                p_mask(i, j) = 1.f;
            }
        }
    }
    return p_mask;
}

template<class DistanceFunction>
PatchMatcher<DistanceFunction>::PatchMatcher(Mat_<Vec3b> s, Mat_<float> s_mask,
                                             Mat_<Vec3b> t, Mat_<float> t_mask,
                                             DistanceFunction distance_function) :
        s_rgb(s), t_rgb(t), s_mask(s_mask), t_mask(t_mask),
        generator(2),
//        generator(time(0)),
        current_max_distance(-1),
        patch_distance(distance_function) {

    patch_distance.initialize(s_rgb, t_rgb);
    s_pixels = _enumerate_pixels(s_mask);
    t_pixels = _enumerate_pixels(t_mask);

    s_patches_mask = _get_patches_mask<P>(s_mask);
    t_patches_mask = _get_patches_mask<P>(t_mask);
    s_patches = _enumerate_pixels(s_patches_mask);
    t_patches = _enumerate_pixels(t_patches_mask);

    initialize_nnf();
}

template<class DistanceFunction>
void PatchMatcher<DistanceFunction>::initialize_nnf() {
    m_nnf = Mat_<Vec2i>(s_rgb.rows + 1 - P, s_rgb.cols + 1 - P, Vec2i(0, 0));
    std::uniform_int_distribution<int> dist(0, t_patches.size() - 1);

    for (const Vec2i &p : s_patches) {
        int index = dist(generator);
        Vec2i q = t_patches[index];
        nnf(p) = q - p;
    }

}

template<class DistanceFunction>
void PatchMatcher<DistanceFunction>::random_search(const Vec2i &p) {
    Vec2i q = nnf(p) + p;
    int min_d = patch_distance(p, q);
    std::uniform_int_distribution<int> dist(0, t_patches.size() - 1);

    for (int n = 0; n < 20; n++) {
        int index = dist(generator);
        q = t_patches[index];
        int d = patch_distance(p, q, min_d);
        if (d <= min_d) {
            min_d = d;
            nnf(p) = q - p;
        }
    }
}

template<class DistanceFunction>
int PatchMatcher<DistanceFunction>::_propagate_distance(const Vec2i &p, const Vec2i &n, int max_d) {

    // Check that neighbor is not out of bounds of nnf
    if (n[0] < 0 || n[1] < 0 || n[0] >= m_nnf.rows || n[1] >= m_nnf.cols) {
        return std::numeric_limits<int>::max();
    }

    Vec2i q = nnf(n) + p;
    if (q[0] < 0 || q[1] < 0
        || q[0] >= t_patches_mask.rows
        || q[1] >= t_patches_mask.cols) {
        return std::numeric_limits<int>::max();
    }

    return patch_distance(p, q, max_d);
}

template<class DistanceFunction>
void PatchMatcher<DistanceFunction>::propagate(const Vec2i &p, const array<Vec2i, 2> &neighbors) {
    Vec2i q = nnf(p) + p;
    int min_d = patch_distance(p, q);
    for (int n = 0; n < neighbors.size(); n++) {
        int d = _propagate_distance(p, neighbors[n], min_d);
        if (d < min_d) {
            min_d = d;
            m_nnf(p) = m_nnf(neighbors[n]);
        }
    }

}

template<class DistanceFunction>
void PatchMatcher<DistanceFunction>::iterate_rd() {
    for (const Vec2i &p : s_patches) {
        propagate(p, {p - Vec2i(1, 0), p - Vec2i(0, 1)});
        random_search(p);
    }
}

template<class DistanceFunction>
void PatchMatcher<DistanceFunction>::iterate_lu() {
    for (auto it = s_patches.rbegin(); it != s_patches.rend(); it++) {
        const Vec2i &p = *it;
        propagate(p, {p + Vec2i(1, 0), p + Vec2i(0, 1)});
        random_search(p);
    }
}

template<class DistanceFunction>
void PatchMatcher<DistanceFunction>::iterate_n_times(int n) {
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) {
            iterate_rd();
        } else {
            iterate_lu();
        }
    }
}

template<class DistanceFunction>
Mat_<Vec3b> PatchMatcher<DistanceFunction>::nnf_to_image() const {
    Mat_<Vec3b> hsv_values(m_nnf.rows, m_nnf.cols, Vec3b(0, 0, 125));
    double max_norm = cv::norm(Vec2i(m_nnf.rows, m_nnf.cols), cv::NORM_L2);
    for (const Vec2i &p : s_patches) {
        int di = nnf(p)[0];
        int dj = nnf(p)[1];
        if (di == 0 && dj == 0) {
            hsv_values(p[0], p[1]) = Vec3b(0, 0, 0);
            continue;
        }
        uchar hue = uchar((std::atan2(di, dj) + PI) * 90 / PI);
        uchar value = uchar(255.0 * cv::norm(nnf(p), cv::NORM_L2) / max_norm);
        hsv_values(p[0], p[1]) = Vec3b(hue, 255, value);
    }
    cv::cvtColor(hsv_values, hsv_values, CV_HSV2BGR);

    return hsv_values;
}

template<class DistanceFunction>
Mat_<uchar> PatchMatcher<DistanceFunction>::distance_map() {
    Mat_<int> distances(m_nnf.size(), 0);
    for (const Vec2i &p : s_patches) {
        Vec2i q = p + nnf(p);
        distances(p[0], p[1]) = patch_distance(p, q);
    }

    Mat_<uchar> result(m_nnf.size());
    if (current_max_distance == -1) {
        double d;
        cv::minMaxLoc(distances, nullptr, &d);
        current_max_distance = int(d);
    }
    distances *= 255.f / current_max_distance;
    distances.convertTo(result, CV_8U);

    for (const Vec2i &p : s_patches) {
        if (distances(p[0], p[1]) == 0) {
            result(p[0], p[1]) = 255;
        }
    }
    return result;
}

template<class DistanceFunction>
Mat_<Vec3b> PatchMatcher<DistanceFunction>::build_s_with_patches_from_t() const {
    Mat_<Vec3f> accumulated(s_rgb.size(), Vec3f(0, 0, 0));
    Mat_<float> counts(s_rgb.size(), 0);
    Mat_<int> ones(P, P, 1);

    Mat_<float> w;
    Mat_<uchar> m;
    s_mask.convertTo(m, CV_8U);
    cv::distanceTransform(m, w, cv::DIST_L2, 5);

    for (int i = 0; i < w.rows; i++) {
        for (int j = 0; j < w.cols; j++) {
            w(i, j) = std::pow(1.3f, -w(i, j));
        }
    }


    for (const Vec2i &p : s_patches) {
        Vec2i q = nnf(p) + p;
        Rect r = get_patch_rect(p);
        int d = patch_distance(p, q);
        float sigma2 = 360000.f;
        float e = std::exp((-2.f * (d * d)) / (2 * sigma2));

        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {

                accumulated(r)(y, x) += t_rgb(get_patch_rect(q))(y, x) * w(r)(y, x) * e;
                counts(r)(y, x) += (w(r)(y, x) * e);
            }
        }

    }


    for (int i = 0; i < accumulated.rows; i++) {
        for (int j = 0; j < accumulated.cols; j++) {
            if (accumulated(i, j) != Vec3f(0, 0, 0)) {
                accumulated(i, j) /= counts(i, j);
            } else {
                accumulated(i, j) = s_rgb(i, j);
            }
        }
    }

    Mat_<Vec3b> result;
    accumulated.convertTo(result, CV_8UC3);

    return result;

}


#endif //COPY_MOVE_DETECTOR_PATCH_MATCHER_H
