#
# Copyright (C) 2023 RenesasElectronics, Co, Ltd.
#
DESCRIPTION = "RZ/V2H OpenCVA Package"

FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}/:"

SRC_URI_append += "\
	file://0002-modified-devicetree-for-drp-drv.patch \
	file://0023-enable-drp-drv.patch \
"
