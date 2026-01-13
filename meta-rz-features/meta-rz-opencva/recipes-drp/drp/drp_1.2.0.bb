#
# This recipe adds a header file of DRP driver to RZ/V SDK environment.
#

DESCRIPTION = "Recipe for header file of DRP driver"
SECTION = "libs"
DEPENDS = ""
LICENSE = "GPL-2.0-WITH-Linux-syscall-note"
LIC_FILES_CHKSUM = "file://COPYING;md5=6bc538ed5bd9a7fc9398086aedcd7e46"
NO_GENERIC_LICENSE[GPL-2.0-WITH-Linux-syscall-note] = "COPYING"
COPY_LIC_MANIFEST = "1"
COPY_LIC_DIRS = "1"
LICENSE_CREATE_PACKAGE = "1"

PACKAGE_ARCH = "${MACHINE_ARCH}"
PACKAGES = "${PN}"
PROVIDES = "${PN}"

PR = "r1"

SRC_URI = "\ 
    file://COPYING;md5sum=6bc538ed5bd9a7fc9398086aedcd7e46 \
    file://drp.h \
    "

# The list of directories or files that are placed in packages.
FILES:${PN} = " \
    ${includedir}/linux/drp.h \
    "

S = "${WORKDIR}"

do_install() {
    install -d ${D}/${includedir}/linux
    install -m 0755 ${S}/drp.h ${D}/${includedir}/linux
}
