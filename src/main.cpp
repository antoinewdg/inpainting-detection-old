#include <iostream>
#include "utils.h"
#include "patch_matcher.h"
#include "patch_distances.h"

int main(int argc, char *argv[]) {


    auto image_s = load_color("files/lena_color_512.tif");
    auto image_t = load_color("files/lena_color_512_d.tif");
    auto mask_s = load_grayscale("files/lena_mask.png");
    auto mask_t = load_grayscale("files/lena_mask_d.png");

    display_and_block(mask_s);
    display_and_block(mask_t);

    const int P = 5;
    MaskedPatchImage s(image_s, mask_s, P);
    MaskedPatchImage t(image_t, mask_t, P);

    typedef EuclidianPatchDistanceRGB<P> ERGB;
    PatchMatcher<ERGB> matcher_ergb(s, t, ERGB());

    double i = measure<>::execution([&matcher_ergb] {
        matcher_ergb.iterate_n_times(5);
    });

    cout << "Done in " << i / 1000 << "s" << endl;
    display_and_block(matcher_ergb.nnf_to_image());


    typedef EuclidianPatchDistanceLab<P> ELab;
    PatchMatcher<ELab> matcher_elab(s, t, ELab());

    i = measure<>::execution([&matcher_elab] {
        matcher_elab.iterate_n_times(5);
    });

    cout << "Done in " << i / 1000 << "s" << endl;
    display_and_block(matcher_elab.nnf_to_image());


    return 0;
}