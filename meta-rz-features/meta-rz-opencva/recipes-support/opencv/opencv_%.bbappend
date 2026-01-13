FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}/:"

SRC_URI:append = "\
	file://0001-add-drpai-setting-for-opencva.patch \
	file://0002-add-oca-list-num.patch \
	file://0003-add-drpai-setting-for-opencva-stereosgm.patch \
"
