#
# This recipe adds a header file of DRP driver to RZ/V SDK environment.
#

DESCRIPTION = "Recipe for header file of DRP driver"
SECTION = "libs"
DEPENDS = ""
LICENSE = "GPL-2.0-WITH-Linux-syscall-note"
LIC_FILES_CHKSUM = "file://COPYING;md5=6bc538ed5bd9a7fc9398086aedcd7e46"

PACKAGE_ARCH = "${MACHINE_ARCH}"
PACKAGES = "${PN}"
PROVIDES = "${PN}"

PR = "r1"

SRC_URI = "\ 
    file://COPYING \
    file://drp.h \
    "

# The list of directories or files that are placed in packages.
FILES_${PN} = " \
    ${includedir}/linux/drp.h \
    "

S = "${WORKDIR}"

do_install() {
    install -d ${D}/${includedir}/linux
    install -m 0755 ${S}/drp.h ${D}/${includedir}/linux
}
