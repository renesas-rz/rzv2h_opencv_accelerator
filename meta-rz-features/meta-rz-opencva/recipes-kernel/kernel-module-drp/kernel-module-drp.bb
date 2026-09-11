SUMMARY = "DRP driver Kernel module for Renesas RZ/V"
DESCRIPTION = "${SUMMARY}"
LICENSE = "GPL-2.0-WITH-Linux-syscall-note"
LIC_FILES_CHKSUM = "file://COPYING;md5=6bc538ed5bd9a7fc9398086aedcd7e46"
NO_GENERIC_LICENSE[GPL-2.0-WITH-Linux-syscall-note] = "COPYING"
COPY_LIC_MANIFEST = "1"
COPY_LIC_DIRS = "1"
LICENSE_CREATE_PACKAGE = "1"

inherit module
require include/rz-modules-common.inc

PN = "kernel-module-drp"
PV = "1.4.0"
PR = "r0"
S = "${WORKDIR}"

DEVICE_NAME=""
DEVICE_NAME:rzv2h-family = "RZV2H"
DEVICE_NAME:rzv2n-family = "RZV2N"

SRC_URI:append:rzv2h-family = "file://drivers/drp/Makefile-RZV2H"
SRC_URI:append:rzv2n-family = "file://drivers/drp/Makefile-RZV2N"

SRC_URI = " \
    file://COPYING;md5sum=6bc538ed5bd9a7fc9398086aedcd7e46 \
    file://drivers/drp/drp-core.c \
    file://drivers/drp/drp-core.h \
    file://drivers/drp/drp-if.c \
    file://drivers/drp/drp-reg.h \
    file://drivers/drp/lock_drp.c \
    file://drivers/drp/lock_drp.h \
    file://include/linux/drp.h \
    file://include/uapi/linux/drp.h \
"

# Build Memory Manager kernel module without suffix
KERNEL_MODULE_PACKAGE_SUFFIX = ""

do_compile() {
    install -d ${KERNELSRC}/drivers/drp
    install -m 0644 ${S}/drivers/drp/Makefile-${DEVICE_NAME}  ${S}/drivers/drp/Makefile

    # Install header file to workspace
    install -d ${KERNELSRC}/include/linux
    install -m 0644 ${S}/include/linux/drp.h  ${KERNELSRC}/include/linux
    install -d ${KERNELSRC}/include/uapi/linux
    install -m 0644 ${S}/include/uapi/linux/drp.h ${KERNELSRC}/include/uapi/linux

    oe_runmake -C ${S}/drivers/drp
}

do_install() {
    # Create destination directories
    install -d ${D}/${nonarch_base_libdir}/modules/${KERNEL_VERSION}/extra/
    install -d ${D}/${includedir}/linux/

    # Install kernel module
    install -m 0644 ${S}/drivers/drp/drp.ko ${D}/${nonarch_base_libdir}/modules/${KERNEL_VERSION}/extra/

    # Install header file for Cross Compiler
    install -m 0644 ${S}/include/uapi/linux/drp.h ${D}/${includedir}/linux/
}

PACKAGES = " \
    ${PN} \
    ${PN}-dev \
    ${PN}-dbg \
"

# The list of directories or files that are placed in packages.
FILES:${PN} = " \
    ${nonarch_base_libdir}/modules/${KERNEL_VERSION}/extra/drp.ko \
"

FILES:${PN}-dev += " \
    ${includedir}/linux/drp.h \
"

RPROVIDES:${PN} += "kernel-module-drp"

KERNEL_MODULE_AUTOLOAD = "drp"

INSANE_SKIP:${PN}-dbg = "buildpaths"
