#include <iostream>
#include <string>
#include <cstdlib>
#include <msg.h>

// Include all necessary headers
#include <test/SpMV.h>
#include <test/SpMSpV.h>
#include <test/SpMM.h>
#include <test/SpGEMM.h>
#include <test/dfmt.h>
#include <test/transformer.h>

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <kernel_name> [arguments...]\n"
              << "Available kernels and arguments:\n"
              << "  spmv        <matrix_path>\n"
              << "  spmspv      <matrix_path>\n"
              << "  spmm        <matrix_path>\n"
              << "  spgemm      <matrix_path>\n"
              << "  spgemm2     <matrix_A_path> <matrix_B_path>\n"
              << "  format      <matrix_path>\n"
              << "  transformer <arg1> <arg2> <int_arg3> <int_arg4>\n";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    // Shift argc/argv logic:
    // argv[2] in the new program corresponds to argv[1] in the old macros
    
    if (command == "spmv") {
        if (argc < 3) {
            std::cerr << "Error: Missing argument for SpMV.\n";
            return 1;
        }
        test_SpMV(argv[2]);
        return 0;
    } 
    else if (command == "spgemm") {
        if (argc < 3) {
            std::cerr << "Error: Missing argument for SpGEMM.\n";
            return 1;
        }
        return test_SpGEMM(argv[2]);
    } 
    else if (command == "spgemm2") {
        if (argc < 4) {
            std::cerr << "Error: Missing arguments for SpGEMM2.\n";
            return 1;
        }
        return test_SpGEMM(argv[2], argv[3]);
    } 
    else if (command == "spmspv") {
        if (argc < 3) {
            std::cerr << "Error: Missing argument for SpMSpV.\n";
            return 1;
        }
        test_SpMSpV(argv[2]);
        return 0;
    } 
    else if (command == "spmm") {
        if (argc < 3) {
            std::cerr << "Error: Missing argument for SpMM.\n";
            return 1;
        }
        test_SpMM(argv[2]);
        return 0;
    } 
    else if (command == "format") {
        if (argc < 3) {
            std::cerr << "Error: Missing argument for Format.\n";
            return 1;
        }
        return test_data_format(argv[2]);
    } 
    else if (command == "transformer") {
        if (argc < 6) {
            std::cerr << "Error: Missing arguments for Transformer.\n";
            return 1;
        }
        return test_transformer(argv[2], argv[3], std::atoi(argv[4]), std::atoi(argv[5]));
    } 
    else {
        std::cerr << "Unknown kernel: " << command << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
