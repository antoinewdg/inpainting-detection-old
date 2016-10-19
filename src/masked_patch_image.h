//
// Created by antoinewdg on 10/19/16.
//

#ifndef COPY_MOVE_DETECTOR_MASKED_PATCH_IMAGES_H
#define COPY_MOVE_DETECTOR_MASKED_PATCH_IMAGES_H

#include "utils.h"

struct MaskedPatchImage {


    MaskedPatchImage(Mat_<Vec3b> image, Mat_<float> mask, int P) :
            image(image), mask(mask), P(P),
            rows(image.rows), cols(image.cols),
            p_rows(rows + 1 - P), p_cols(cols + 1 - P) {

        _compute_pixels();
        _compute_patches();

    }


    inline const Vec3b &operator()(const Vec2i &p) const {
        return image(p[0], p[1]);
    }

    inline cv::Size size() const {
        return cv::Size(cols, rows);
    }

    inline cv::Size p_size() const {
        return cv::Size(p_cols, p_rows);
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
        total_patches_mask = Mat_<float>(p_size(), 0.f);
        for (int i = 0; i < mask.rows + 1 - P; i++) {
            for (int j = 0; j < mask.cols + 1 - P; j++) {
                int n = cv::countNonZero(mask(Rect(j, i, P, P)));
                if (n > 0) {
                    partial_patches.emplace_back(i, j);
                }
                if (n == P * P) {
                    total_patches.emplace_back(i, j);
                    total_patches_mask(i, j) = 1.f;
                }
            }
        }
    }

    Mat_<Vec3b> image;
    Mat_<float> mask, total_patches_mask;
    vector<Vec2i> pixels, partial_patches, total_patches;
    int P, rows, cols, p_rows, p_cols;
};

#endif //COPY_MOVE_DETECTOR_MASKED_PATCH_IMAGES_H
