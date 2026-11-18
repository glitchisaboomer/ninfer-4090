#include "ops/linear/w8/w8_dispatch.h"

#include <array>
#include <iostream>
#include <utility>

int main() {
    constexpr std::array<std::pair<std::int32_t, std::int32_t>, 7> cases{{
        {10240, 4096}, {8192, 4096}, {4096, 4096}, {1024, 4096},
        {24576, 4096}, {12288, 4096}, {4096, 12288},
    }};
    for (const auto& shape : cases) {
        try {
            (void)ninfer::ops::detail::select_w8_a16_launch(shape.first, shape.second, 1);
            (void)ninfer::ops::detail::select_w8_a16_launch(shape.first, shape.second, 65);
        } catch (const std::exception& error) {
            std::cerr << "FAIL W8 SenseNova dispatch " << shape.first << "x" << shape.second
                      << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << "OK W8 SenseNova dispatch\n";
    return 0;
}
