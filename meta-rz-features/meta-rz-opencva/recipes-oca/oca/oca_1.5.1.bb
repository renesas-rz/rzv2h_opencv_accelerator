#
# This recipe adds OpenCV_Bin.bin to root/boot.
#

DESCRIPTION = "Recipe for OpenCVA"
SECTION = "libs"
DEPENDS = ""
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

SRC_URI = " \
    file://OpenCV_Bin.bin \
"

INSTALL_DIRECTORY ?= "/boot"

do_install() {
    install -d ${D}/${INSTALL_DIRECTORY}/
    install -m 0755 ${WORKDIR}/OpenCV_Bin.bin ${D}/${INSTALL_DIRECTORY}/
}

FILES:${PN} = " \
    ${INSTALL_DIRECTORY}/* \
"
