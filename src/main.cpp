#include <iostream>
#include "utils.h"
#include "patch_match.h"
#include "shuffle.h"

int main(int argc, char *argv[]) {

//    if (argc < 2) {
//        cout << "Enter the image path as argument." << endl;
//    }
    auto image = load_color("files/lena_color_512.tif");
    cv::theRNG().state = time(NULL);
    srand(time(NULL));

//    auto out = shuffle_image(image, 20);
//    cv::imwrite("../files/out.png", out);
//    int psize = 16;
    PatchMatcher matcher(image, 16);
//    matcher.display_offset_map();
//    matcher.diplay_offset_image();
//    matcher.diplay_offset_image();
    for (int i = 0; i < 5; i++) {
        auto t = measure<>::execution([&matcher, &image]() {

            matcher.iterate_rd();
            matcher.iterate_lu();
        });
        cout << "Done in " << float(t) / 1000 << endl;
        matcher.diplay_offset_image();
    }
//    iterate_rd(map, image, psize);
//    display_offset_map(map);
//    display_and_block(image);

    return 0;
}