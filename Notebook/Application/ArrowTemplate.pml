# ===================
# Z. Display Settings
# ===================
#set mesh_radius = 0.01
#set antialias = 1
#set stick_radius = 0.22
#set dash_radius=0.07
set sphere_scale= 0.22
set ribbon_radius =0.1
#set direct =0.0
#set cartoon_fancy_helices=1
bg_color white
#set gamma=1.5
util.ray_shadows('none')
set ray_trace_mode=0
set cartoon_flat_sheets = 1.0
set cartoon_smooth_loops = 0
set dash_gap, 0.0
set dash_width, 5
set orthoscopic, on
#set stick_transparency, 0.8
#set cartoon_transparency, 0.5


# ==========================
# A. Load and Show
# ==========================
run cgo_arrow.py











load REPLACE_WITH_FILENAME



REPLACE_WITH_CGOARROWS


spectrum b, rainbow
hide (hydrogen)
hide spheres
hide wire
set cartoon_transparency=0.5
hide ribbon
show cartoon
dss

#save REPLACE_WITH_ID.pse