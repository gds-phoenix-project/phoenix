
#include "phxfs_utils.h"

int main(int argc, char **argv){   
    GDSOpts opts;

    if (!parseOpts(argc, argv, opts)){
        exit(EXIT_FAILURE);
    }

    switch (opts.xfer_mode) {
        case GPUD_WITHOUT_PHONY_BUFFER:
            run_phoenix(opts);
            break;
        case GPUD_WITH_PYONY_BUFFER:
            run_gds(opts);
            break;
        case GPUD_WITH_CPU_BUFFER:
            run_posix(opts);
            break;
        default:
            pr_error("Unsupport xfer mode");
            printHelp(argv[0]);
            exit(EXIT_FAILURE);
    }


    return 0;
}