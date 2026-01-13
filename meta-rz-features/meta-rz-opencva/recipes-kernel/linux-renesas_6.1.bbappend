#
# Copyright (C) 2025 RenesasElectronics, Co, Ltd.
#
DESCRIPTION = "RZ/V2H DRP Driver"

FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}/:"

SRC_URI:append = "\
	file://0001-add-drp-property-to-devicetree.patch \
	file://0002-enable-drp-driver.patch \
	file://0003-clk-renesas-r9a09g057-cpg-init-DRP-reset.patch \
"
