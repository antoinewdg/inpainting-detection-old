#include <cstdlib>
#include <iostream>
#include "utils.h"
#include "patch_matcher.h"
#include "patch_distances.h"
#include "shuffle.h"

int main(int argc, char *argv[]) {


    auto image_s = load_color("files/lena_color_512.tif");
    auto image_t = load_color("files/lena_color_512_d.tif");
    auto mask_s = load_grayscale("files/lena_mask.png");
    auto mask_t = load_grayscale("files/lena_mask_d.png");

    typedef EuclidianPatchDistanceRGB<5> ERGB;
    PatchMatcher<ERGB> matcher_ergb(image_s, mask_s, image_t, mask_t, ERGB());

    double t = measure<>::execution([&matcher_ergb] {
        matcher_ergb.iterate_n_times(5);
    });

    cout << "Done in " << t / 1000 << "s" << endl;
    display_and_block(matcher_ergb.nnf_to_image());


    typedef EuclidianPatchDistanceLab<5> ELab;
    PatchMatcher<ELab> matcher_elab(image_s, mask_s, image_t, mask_t, ELab());

    t = measure<>::execution([&matcher_elab] {
        matcher_elab.iterate_n_times(5);
    });

    cout << "Done in " << t / 1000 << "s" << endl;
    display_and_block(matcher_elab.nnf_to_image());


    return 0;
}