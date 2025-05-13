FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}/:"

SRC_URI_append += "\
	file://0001-add-drpai-setting-for-opencva.patch \
	file://0002-add-drpai-remap-for-opencva.patch \
	file://0003-support-multi-proc-and-thread.patch \
"
