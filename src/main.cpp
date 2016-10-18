#include <cstdlib>
#include <iostream>
#include "utils.h"
#include "patch_matcher.h"
#include "shuffle.h"

int main(int argc, char *argv[]) {


    auto image_s = load_color("files/lena_color_512.tif");
    auto image_t = load_color("files/lena_color_512_d.tif");
    auto mask_s = load_grayscale("files/lena_mask.png");
    auto mask_t = load_grayscale("files/lena_mask_d.png");

    PatchMatcher<5> matcher(image_s, mask_s, image_t, mask_t);

    double t = measure<>::execution([&matcher] {
        matcher.iterate_n_times(5);
    });

    cout << "Done in " << t / 1000 << "s" << endl;

    display_and_block(matcher.nnf_to_image());


    return 0;
}