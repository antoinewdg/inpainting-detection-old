//
// Created by antoinewdg on 11/10/16.
//

#ifndef COPY_MOVE_DETECTOR_MISC_H_H
#define COPY_MOVE_DETECTOR_MISC_H_H

#include <random>

#include "utils.h"

inline Mat_<Vec3b> shuffle_image(Mat_<Vec3b> src, int n) {
    std::random_device r;
    Mat_<Vec3b> im(src);
    std::default_random_engine e1(r());
    std::uniform_int_distribution<int> uniform_dist(10, std::min(im.rows, im.cols) / 4);
    for (int k = 0; k < n; k++) {
        int size = uniform_dist(e1);
        std::uniform_int_distribution<int> idist1(0, src.rows / 2 - size);
        std::uniform_int_distribution<int> idist2(src.rows / 2, src.rows - size - 1);
        std::uniform_int_distribution<int> jdist(0, src.cols - size - 1);

        cv::Rect r1(jdist(r), idist1(r), size, size);
        cv::Rect r2(jdist(r), idist2(r), size, size);

        auto temp = im(r1).clone();
        im(r2).copyTo(im(r1));
        temp.copyTo(im(r2));

    }

    return im;

}

#endif //COPY_MOVE_DETECTOR_MISC_H_H
