#
# Copyright (C) 2026 RenesasElectronics, Co, Ltd.
#
DESCRIPTION = "RZ/V2H and RZ/V2N DRP Driver"

FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}/:"

SRC_URI:append:rzv2n-family = "\
    file://0001-add-drp-property-to-devicetree-RZV2N.patch \
"

SRC_URI:append:rzv2h-family = "\
    file://0001-add-drp-property-to-devicetree-RZV2H.patch \
    file://0002-clk-renesas-r9a09g057-cpg-init-DRP-reset-RZV2H.patch \
"
