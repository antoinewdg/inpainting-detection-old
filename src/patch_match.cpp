//
// Created by antoinewdg on 11/10/16.
//

#include "patch_match.h"

PatchMatcher::PatchMatcher(Mat_<Vec3b> src, int psize) :
        src(src), psize(psize), alpha(0.5) {
    initialize_offset_map();
    cv::cvtColor(src, src_lab, CV_BGR2Lab);
}

void PatchMatcher::initialize_offset_map() {
    offset_map = Mat_<Vec2i>(src.rows + 1 - psize, src.cols + 1 - psize);
    cv::randu(offset_map, cv::Scalar(0, 0), cv::Scalar(offset_map.rows, offset_map.cols));

    for (int i = 0; i < offset_map.rows; i++) {
        for (int j = 0; j < offset_map.cols; j++) {
            offset_map(i, j) -= Vec2i(i, j);
        }
    }
}

void PatchMatcher::display_offset_map() const {
    Mat_<Vec3b> hsv_values(offset_map.rows, offset_map.cols);
    double max_norm = cv::norm(Vec2i(offset_map.rows, offset_map.cols), cv::NORM_L2);
    for (int i = 0; i < offset_map.rows; i++) {
        for (int j = 0; j < offset_map.cols; j++) {
            int di = offset_map(i, j)[0];
            int dj = offset_map(i, j)[1];
            if (di == 0 && dj == 0) {
                hsv_values(i, j) = Vec3b(0, 0, 0);
                continue;
            }
            uchar hue = uchar((std::atan2(di, dj) + PI) * 90 / PI);
            uchar value = uchar(255.0 * cv::norm(offset_map(i, j), cv::NORM_L2) / max_norm);
            hsv_values(i, j) = Vec3b(hue, value, value);
        }
    }
    cv::cvtColor(hsv_values, hsv_values, CV_HSV2BGR);

    display_and_block(hsv_values);
}




void PatchMatcher::iterate_rd() {

    auto distance_lambda = [this](const Vec2i &a, const Vec2i &b) {
        return distance_checked_rd(a, b);
    };

    for (int i = 1; i < offset_map.rows; i++) {
        propagate<1>(i, 0, {Vec2i(i - 1, 0)}, distance_lambda);
        random_search(i, 0);
    }

    for (int j = 1; j < offset_map.cols; j++) {
        propagate<1>(0, j, {Vec2i(0, j - 1)}, distance_lambda);
        random_search(0, j);
    }

    for (int i = 1; i < offset_map.rows; i++) {
        for (int j = 1; j < offset_map.cols; j++) {
            propagate<2>(i, j, {Vec2i(i - 1, j), Vec2i(i, j - 1)}, distance_lambda);
            random_search(i, j);
        }
    }
}

void PatchMatcher::iterate_lu() {

    auto distance_lambda = [this](const Vec2i &a, const Vec2i &b) {
        return distance_checked_lu(a, b);
    };

    for (int i = offset_map.rows - 2; i >= 0; i--) {
        int j = offset_map.cols - 1;
        propagate<1>(i, j, {Vec2i(i + 1, j)}, distance_lambda);
        random_search(i, j);
    }

    for (int j = offset_map.cols - 2; j >= 0; j--) {
        int i = offset_map.rows - 1;
        propagate<1>(i, j, {Vec2i(i, j + 1)}, distance_lambda);
        random_search(i, j);
    }

    for (int i = offset_map.rows - 2; i >= 0; i--) {
        for (int j = offset_map.cols - 2; j >= 0; j--) {
            propagate<2>(i, j, {Vec2i(i + 1, j), Vec2i(i, j + 1)}, distance_lambda);
            random_search(i, j);
        }
    }
}

Mat_<Vec3b> PatchMatcher::build_offset_image() const {
    Mat_<Vec3b> rebuilt(src.rows, src.cols);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            Vec2i offset = offset_map(i - (i % psize), j - (j % psize));
            rebuilt(i, j) = src(i + offset[0], j + offset[1]);
        }
    }

    return rebuilt;
}


void PatchMatcher::diplay_offset_image() const {
    display_and_block(build_offset_image());
}

void PatchMatcher::random_search(int i, int j) {
    int r = std::max({i, j, offset_map.rows - 1 - i, offset_map.cols - 1 - j});
    Vec2i pos(i, j);
    double min_d = distance_unchecked(pos, pos + offset_map(i, j));
    while (r >= 1) {
        int kmin = std::max(0, i - r);
        int kmax = std::min(offset_map.rows - 1, i + r);
        int k = (rand() % (kmax - kmin)) + kmin;
        int lmin = std::max(0, j - r);
        int lmax = std::min(offset_map.cols - 1, j + r);
        int l = (rand() % (lmax - lmin)) + lmin;


        double d = distance_checked_rdlu(pos, pos + offset_map(k, l));
        if (d < min_d) {
            offset_map(i, j) = offset_map(k, l);
            min_d = d;
        }
        r /= 2;

    }
}