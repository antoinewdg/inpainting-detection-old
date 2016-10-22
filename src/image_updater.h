//
// Created by antoinewdg on 10/19/16.
//

#ifndef COPY_MOVE_DETECTOR_IMAGEUPDATER_H
#define COPY_MOVE_DETECTOR_IMAGEUPDATER_H

#include "utils.h"
#include "masked_patch_image.h"


class NLMeansImageUpdater {
    typedef MaskedPatchImage MImage;
public:
    NLMeansImageUpdater(MImage &s, MImage &t, Mat_<Vec2i> nnf, int P) :
            g(P, P, 1.f / (P * P)), s(s), t(t), nnf(nnf), P(P) {

    }

    void update() {
        vector<float> sums(s.pixels.size(), 0.f);
        vector<float> values(s.pixels.size(), 0.f);



        int p_2 = P / 2;
        for (const Vec2i &z: s.pixels) {
            float k_z = 0;
            Vec3f acc(0, 0, 0);
            int i = z[0], j = z[1];
            for (int k = -p_2; k <= p_2; k++) {
                for (int l = -p_2; l <= p_2; l++) {
                    Vec2i h(k, l);

                    // Useful for when the mask is close to the boundary
                    if (s.partial_patches_mask((z - h)[0], (z - h)[1]) != 1.f) {
                        continue;
                    }
                    auto z_hat = nnf(z - h) + z;
                    float g_h = g(h + Vec2i(p_2, p_2));
                    k_z += g_h;
                    acc += g_h * Vec3f(t(z_hat));

                }
            }

            s(z) = acc / k_z;
        }

    }

    MImage &s, &t;
    Mat_<float> g;
    Mat_<Vec2i> nnf;
    int P;
};

#endif //COPY_MOVE_DETECTOR_IMAGEUPDATER_H
