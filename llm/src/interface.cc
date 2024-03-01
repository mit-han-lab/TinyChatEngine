#include "interface.h"
#include <iostream>

void set_print_black() {
    printf("\033[0;30m");
}

void set_print_red() {
    printf("\033[1;31m");
}

void set_print_yellow() {
    printf("\033[0;33m");
}

void set_print_bold_yellow() {
    printf("\033[1;33m");
}

void set_print_blue() {
    printf("\033[1;34m");
}

void set_print_white() {
    printf("\033[0;37m");
}

void set_print_reset() {
    printf("\033[0m");
}
