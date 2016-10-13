//
// Created by antoinewdg on 12/10/16.
//

#ifndef COPY_MOVE_DETECTOR_PATCH_MATCHER_H
#define COPY_MOVE_DETECTOR_PATCH_MATCHER_H

#define PI 3.14159265

#include <limits>

#include "utils.h"

//struct PatchMatrix {
//
//    PatchMatrix(Mat_<Vec3b> src, int p) : m(src.rows - 1 + p), n(src.cols - 1 + p), p(p) {
//
//        data.resize(n * m);
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < n; j++) {
//                cout << i << " " << j;
//                src(cv::Rect(j, i, p, p)).copyTo((*this)(i, j));
//                display_and_block((*this)(i, j));
//            }
//        }
//    }
//
//    inline Mat_<Vec3b> operator()(int i, int j) {
//        return data[i * n + j];
//    }
//
//    size_t m, n;
//    int p;
//    vector<Mat_<Vec3b>> data;
//};

template<int P>
class PatchMatcher {
public:
    PatchMatcher(Mat_<Vec3b> origin, Mat_<Vec3b> target);

    void initialize_nnf();

    inline int unchecked_distance(int i, int j, int k, int l, int max_d);

    int checked_distance(int i, int j, const Vec2i &q, int max_d);

    int patch_distance(const Mat_<Vec3b> &p, const Mat_<Vec3b> &q, int max_d);

    void iterate_rd();

    void iterate_ul();

    void propagate(int i, int j, const array<Vec2i, 2> &neighbors);

    void random_search(int i, int j);

    Mat_<Vec3b> nnf_to_image() const;

    Mat_<Vec3b> recompose_origin_with_nnf(int k = 0) const;

    int target_patches_rows, target_patches_cols;

//    PatchMatrix origin_patches, target_patches;
    Mat_<Vec3b> origin, target, origin_lab, target_lab;
//    Mat_<int> distances;
    Mat_<Vec2i> nnf;
};


template<int P>
PatchMatcher<P>::PatchMatcher(Mat_<Vec3b> origin, Mat_<Vec3b> target):
        origin(origin), target(target),
        target_patches_rows(target.rows - P + 1),
        target_patches_cols(target.cols - P + 1)
//        origin_patches(origin, P),
//        target_patches(target, P) {
{
    cv::cvtColor(origin, origin_lab, cv::COLOR_BGR2Lab);
    cv::cvtColor(target, target_lab, cv::COLOR_BGR2Lab);
    initialize_nnf();
}


template<int P>
void PatchMatcher<P>::initialize_nnf() {
    nnf = Mat_<Vec2i>(origin.rows + 1 - P, origin.cols + 1 - P);
//    distances = Mat_<int>(origin.rows + 1 - P, origin.cols + 1 - P);
    cv::randu(nnf, cv::Scalar(0, 0), cv::Scalar(target.rows + 1 - P, target.cols + 1 - P));
    for (int i = 0; i < nnf.rows; i++) {
        for (int j = 0; j < nnf.cols; j++) {
            nnf(i, j) -= Vec2i(i, j);
//            distances(i, j) = checked_distance(i, j, nnf(i, j), std::numeric_limits<int>::max());
        }
    }
}


template<int P>
int PatchMatcher<P>::patch_distance(const Mat_<Vec3b> &p, const Mat_<Vec3b> &q, int max_d) {
    int d = 0;
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < P; j++) {
            int a = int(p(i, j)[0]) - q(i, j)[0];
            int b = int(p(i, j)[1]) - q(i, j)[1];
            int c = int(p(i, j)[2]) - q(i, j)[2];
            d += a * a + b * b + c * c;
            if (d >= max_d) {
                return d;
            }
        }

    }

    return d;
}

template<int P>
int PatchMatcher<P>::unchecked_distance(int i, int j, int k, int l, int max_d) {
    int a = i - k;
    int b = j - l;
    if (a * a + b * b < 1) {
        return std::numeric_limits<int>::max();
    }
    return patch_distance(
            origin_lab(cv::Rect(j, i, P, P)),
            target_lab(cv::Rect(l, k, P, P)),
            max_d
    );
}

template<int P>
int PatchMatcher<P>::checked_distance(int i, int j, const Vec2i &q, int max_d) {

    if (q[0] < 0 || q[1] < 0 || q[0] >= nnf.rows || q[1] >= nnf.cols) {
        return std::numeric_limits<int>::max();
    }
    int k = nnf(q[0], q[1])[0] + i;
    int l = nnf(q[0], q[1])[1] + j;

    if (k < 0 || l < 0 || k >= target_patches_rows || l >= target_patches_cols) {
        return std::numeric_limits<int>::max();
    }

    return unchecked_distance(i, j, k, l, max_d);
}

template<int P>
void PatchMatcher<P>::iterate_rd() {
    for (int i = 0; i < nnf.rows; i++) {
        for (int j = 0; j < nnf.cols; j++) {
            propagate(i, j, {Vec2i(i - 1, j), Vec2i(i, j - 1)});
            random_search(i, j);
        }
    }
}

template<int P>
void PatchMatcher<P>::iterate_ul() {
    for (int i = nnf.rows - 1; i >= 0; i--) {
        for (int j = nnf.cols - 1; j >= 0; j--) {
            propagate(i, j, {Vec2i(i + 1, j), Vec2i(i, j + 1)});
            random_search(i, j);
        }
    }
}

template<int P>
void PatchMatcher<P>::propagate(int i, int j, const array<Vec2i, 2> &neighbors) {

    int min_d = checked_distance(i, j, Vec2i(i, j), 0);
    for (int n = 0; n < neighbors.size(); n++) {
        int d = checked_distance(i, j, neighbors[n], min_d);
        if (d < min_d) {
            min_d = d;
            nnf(i, j) = nnf(neighbors[n][0], neighbors[n][1]);
        }
    }
}

template<int P>
void PatchMatcher<P>::random_search(int i, int j) {
    int r = std::max({i, j, target_patches_rows - 1 - i, target_patches_cols - 1 - j});
    int min_d = checked_distance(i, j, Vec2i(i, j), 0);
    while (r > 1) {
        int kmin = std::max(0, i - r);
        int kmax = std::min(target_patches_rows - 1, i + r);
        int k = (rand() % (kmax - kmin)) + kmin;
        int lmin = std::max(0, j - r);
        int lmax = std::min(target_patches_cols - 1, j + r);
        int l = (rand() % (lmax - lmin)) + lmin;

        int d = unchecked_distance(i, j, k, l, min_d);
        if (d < min_d) {
            min_d = d;
            nnf(i, j) = Vec2i(k - i, l - j);
        }

        r /= 2;
//        int d =
//        int d =
    }
}


template<int P>
Mat_<Vec3b> PatchMatcher<P>::nnf_to_image() const {
//    cout << nnf << endl;
    Mat_<Vec3b> hsv_values(nnf.rows, nnf.cols);
    double max_norm = cv::norm(Vec2i(nnf.rows, nnf.cols), cv::NORM_L2);
    for (int i = 0; i < nnf.rows; i++) {
        for (int j = 0; j < nnf.cols; j++) {
            int di = nnf(i, j)[0];
            int dj = nnf(i, j)[1];
            if (di == 0 && dj == 0) {
                hsv_values(i, j) = Vec3b(0, 0, 0);
                continue;
            }
            uchar hue = uchar((std::atan2(di, dj) + PI) * 90 / PI);
            uchar value = uchar(255.0 * cv::norm(nnf(i, j), cv::NORM_L2) / max_norm);
            hsv_values(i, j) = Vec3b(hue, value, value);
        }
    }
    cv::cvtColor(hsv_values, hsv_values, CV_HSV2BGR);

    return hsv_values;
}


template<int P>
Mat_<Vec3b> PatchMatcher<P>::recompose_origin_with_nnf(int k) const {
    Mat_<Vec3b> rebuilt(origin.rows, origin.cols);
    for (int i = k; i < origin.rows + 1 - P + k; i++) {
        for (int j = k; j < origin.cols + 1 - P + k; j++) {
            Vec2i offset = nnf(i - (i % P) + k, j - (j % P) + k);
            rebuilt(i, j) = target(i + offset[0], j + offset[1]);
        }
    }

    return rebuilt;
}

#endif //COPY_MOVE_DETECTOR_PATCH_MATCHER_H
