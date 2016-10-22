//
// Created by antoinewdg on 10/19/16.
//

#ifndef COPY_MOVE_DETECTOR_MASKED_PATCH_IMAGES_H
#define COPY_MOVE_DETECTOR_MASKED_PATCH_IMAGES_H

#include "utils.h"

struct MaskedPatchImage {


    MaskedPatchImage(Mat_<Vec3b> image, Mat_<float> mask, int P) :
            image(image), mask(mask), P(P), p_2(P / 2),
            rows(image.rows), cols(image.cols) {

        _compute_pixels();
        _compute_patches();

    }


    inline const Vec3b &operator()(const Vec2i &p) const {
        return image(p[0], p[1]);
    }

    inline Vec3b &operator()(const Vec2i &p) {
        return image(p[0], p[1]);
    }

    inline cv::Size size() const {
        return cv::Size(cols, rows);
    }


    void _compute_pixels() {
        for (int i = 0; i < mask.rows; i++) {
            for (int j = 0; j < mask.cols; j++) {
                if (mask(i, j) != 0) {
                    pixels.emplace_back(i, j);
                }
            }
        }
    }

    void _compute_patches() {
        total_patches_mask = Mat_<float>(size(), 0.f);
        partial_patches_mask = Mat_<float>(size(), 0.f);
        for (int i = p_2; i < mask.rows - p_2; i++) {
            for (int j = p_2; j < mask.cols - p_2; j++) {
                int n = cv::countNonZero(mask(Rect(j - p_2, i - p_2, P, P)));
                if (n > 0) {
                    partial_patches.emplace_back(i, j);
                    partial_patches_mask(i, j) = 1.f;
                }
                if (n == P * P) {
                    total_patches.emplace_back(i, j);
                    total_patches_mask(i, j) = 1.f;
                }
            }
        }
    }

    Mat_<Vec3b> image;
    Mat_<float> mask, total_patches_mask, partial_patches_mask;
    vector<Vec2i> pixels, partial_patches, total_patches;
    int P, rows, cols, p_2;
};

#endif //COPY_MOVE_DETECTOR_MASKED_PATCH_IMAGES_H
