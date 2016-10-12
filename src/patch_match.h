//
// Created by antoinewdg on 11/10/16.
//

#ifndef COPY_MOVE_DETECTOR_PATCH_MATCH_H
#define COPY_MOVE_DETECTOR_PATCH_MATCH_H


#define PI 3.14159265

#include <cmath>
#include <limits>

#include "utils.h"

struct PatchMatcher {

    PatchMatcher(Mat_<Vec3b> src, int psize);
    void initialize_offset_map();
    void display_offset_map() const;

    void diplay_offset_image() const;

    Mat_<Vec3b> build_offset_image() const;

    void iterate_rd();

    void iterate_lu();

    void random_search(int i, int j);

    inline double distance_unchecked(const Vec2i &real_pos, const Vec2i &offset_pos) const {
//        if (cv::norm(real_pos, offset_pos, cv::NORM_L2SQR) <= 100) {
//            return std::numeric_limits<double>::max();
//        }
        cv::Rect r1(real_pos[1], real_pos[0], psize, psize);
        cv::Rect r2(offset_pos[1], offset_pos[0], psize, psize);

        return cv::norm(src_lab(r1), src_lab(r2), cv::NORM_L2SQR);
    }

    inline double distance_checked_rd(const Vec2i &real_pos, const Vec2i &offset_pos) const {
        if (offset_pos[0] >= offset_map.rows
            || offset_pos[1] >= offset_map.cols) {
            return std::numeric_limits<double>::max();
        }
        return distance_unchecked(real_pos, offset_pos);
    }

    inline double distance_checked_lu(const Vec2i &real_pos, const Vec2i &offset_pos) const {
        if (offset_pos[0] < 0
            || offset_pos[1] < 0) {
            return std::numeric_limits<double>::max();
        }
        return distance_unchecked(real_pos, offset_pos);
    }

    inline double distance_checked_rdlu(const Vec2i &real_pos, const Vec2i &offset_pos) const {
        if (offset_pos[0] < 0
            || offset_pos[1] < 0
            || offset_pos[0] >= offset_map.rows
            || offset_pos[1] >= offset_map.cols) {
            return std::numeric_limits<double>::max();
        }

        return distance_unchecked(real_pos, offset_pos);
    }

    template<int n, class Distance>
    inline void propagate(int i, int j, const array<Vec2i, n> &neighbors, const Distance& distance){
        Vec2i real_pos(i, j);
        double min_d = distance_unchecked(real_pos, real_pos + offset_map(i,j));

        for(int k = 0 ; k < n ; k++){
            Vec2i offset = offset_map(neighbors[k][0], neighbors[k][1]);
            double d = distance(real_pos, real_pos + offset);
            if(d < min_d){
                min_d = d;
                offset_map(i,j) = offset;
            }
        }
    }

    Mat_<Vec3b> src, src_lab;
    Mat_<Vec2i> offset_map;
    int psize;
    double alpha;
};


#endif //COPY_MOVE_DETECTOR_PATCH_MATCH_H
