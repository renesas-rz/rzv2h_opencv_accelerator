FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI:append:rzv2n-family = " \
    file://0001-Add-OpenCVA-and-Codec-RZV2N.patch \
"

SRC_URI:append:rzv2h-family = " \
    file://0001-Add-OpenCVA-and-Codec-RZV2H.patch \
"
