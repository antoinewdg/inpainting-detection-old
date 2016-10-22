#include <iostream>
#include "utils.h"
#include "patch_matcher.h"
#include "patch_distances.h"
#include "image_updater.h"


int main(int argc, char *argv[]) {

//    auto image = load_color("files/lena_no_nose.png");
//    auto mask = load_grayscale("files/update_mask.png");

    auto image_a = load_color("files/lena_color_512.tif");
    auto image_b = load_color("files/lena_color_512_d.tif");

    auto mask_a = load_grayscale("files/lena_mask.png");
    auto mask_b = load_grayscale("files/lena_mask_d.png");

//    const int P = 5;
//    MaskedPatchImage s(image, mask, P);
//    MaskedPatchImage t(image, 1.f - mask, P);

    const int P = 5;
    MaskedPatchImage s(image_a, mask_a, P);
    MaskedPatchImage t(image_b, mask_b, P);



//    display_and_block(s.partial_patches_mask);
//    display_and_block(s.mask);
//    display_and_block(s.total_patches_mask);

    typedef EuclidianPatchDistanceRGB<P> ERGB;
//    PatchMatcher<ERGB> matcher(s, t, ERGB());
//
////    display_and_block(matcher.nnf_to_image());
//    matcher.iterate_n_times(5);
////    display_and_block(matcher.nnf_to_image());
//    NLMeansImageUpdater updater(s, t, matcher.m_nnf, P);
//    updater.update();
//    display_and_block(s.image);
    for (int i = 0; i < 10; i++) {
        PatchMatcher<ERGB> matcher(s, t, ERGB());
        matcher.iterate_n_times(5);
//        cout << matcher.m_nnf << endl;
        display_and_block(matcher.nnf_to_image());
//    display_and_block()
//    display_and_block(matcher.nnf_to_image());
        NLMeansImageUpdater updater(s, t, matcher.m_nnf, P);
        updater.update();
//    display_and_block(matcher.nnf_to_image());
        cout << "Displaying image " << endl;
        display_and_block(s.image);
    }

//    auto image_s = load_color("files/lena_color_512.tif");
//    auto image_t = load_color("files/lena_color_512_d.tif");
//    auto mask_s = load_grayscale("files/lena_mask.png");
//    auto mask_t = load_grayscale("files/lena_mask_d.png");
//
//    display_and_block(mask_s);
//    display_and_block(mask_t);
//
////    const int P = 5;
//    MaskedPatchImage s(image_s, mask_s, P);
//    MaskedPatchImage t(image_t, mask_t, P);
//
//    typedef EuclidianPatchDistanceRGB<P> ERGB;
//    PatchMatcher<ERGB> matcher_ergb(s, t, ERGB());
//
//    double i = measure<>::execution([&matcher_ergb] {
//        matcher_ergb.iterate_n_times(5);
//    });
//
//    cout << "Done in " << i / 1000 << "s" << endl;
//    display_and_block(matcher_ergb.nnf_to_image());
//
//
//    typedef EuclidianPatchDistanceLab<P> ELab;
//    PatchMatcher<ELab> matcher_elab(s, t, ELab());
//
//    i = measure<>::execution([&matcher_elab] {
//        matcher_elab.iterate_n_times(5);
//    });
//
//    cout << "Done in " << i / 1000 << "s" << endl;
//    display_and_block(matcher_elab.nnf_to_image());


    return 0;
}