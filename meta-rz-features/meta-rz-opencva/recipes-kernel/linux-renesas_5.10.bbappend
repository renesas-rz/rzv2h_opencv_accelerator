#
# Copyright (C) 2025 RenesasElectronics, Co, Ltd.
#
DESCRIPTION = "RZ/V2H OpenCVA Package"

FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}/:"

SRC_URI_append += "\
	file://0002-arm64-dts-renesas-r9a09g057-Add-node-and-memory-area.patch \
	file://0023-enable-drp-drv.patch \
"
