bg_color white

load ./4tst.cif

orient

remove solvent

# ====================
# Style
# ======================
bg_color white
set stick_radius, 0.1
set stick_transparency, 0.20

dss
hide ribbon
show cartoon
show stick, 4tst
set ray_opaque_background, off

set ray_shadow, off

set ray_trace_mode, 0
set cartoon_transparency, 0.1, 4tst




# =====================
# Morph
# =====================

# FAILED for really large molecules



spectrum b, density purple br7 brightorange yellow
#spectrum b, minimum=0, maximum=1.0
#util.cnc("all")
#hide stick, 4tst


set ray_opaque_background, off

set ray_shadow, off

set ray_trace_mode, 0


set state, 1, 4tst
png Movie_4tst_01.png, height=1440, width=1980, dpi=500, ray=1
set state, 2, 4tst
png Movie_4tst_02.png, height=1440, width=1980, dpi=500, ray=1
set state, 3, 4tst
png Movie_4tst_03.png, height=1440, width=1980, dpi=500, ray=1
set state, 4, 4tst
png Movie_4tst_04.png, height=1440, width=1980, dpi=500, ray=1
set state, 5, 4tst
png Movie_4tst_05.png, height=1440, width=1980, dpi=500, ray=1
set state, 6, 4tst
png Movie_4tst_06.png, height=1440, width=1980, dpi=500, ray=1






quit
