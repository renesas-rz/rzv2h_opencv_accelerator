require include/opencva/opencva_packages.inc

IMAGE_INSTALL:append = " \
    kernel-module-drp \
"

TOOLCHAIN_TARGET_TASK:append = " \
    kernel-module-drp-dev \
"
