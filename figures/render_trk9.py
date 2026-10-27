"""Nine panels: one Trk lineage (rounds 0, 4, 15) against all three receptors.

Within each row the three complexes are superposed on the RECEPTOR chain and rendered
from one camera, so the binder's change in pose and confidence is the only thing that
moves. Across rows the camera distance and clipping planes are made identical as well,
so every one of the nine panels is at the same magnification and the three receptors
can be compared by eye. Receptor surface in grey with the six N- and six C-terminal
residues hidden (AF3 places those flexible ends differently between complexes of the
same receptor); binder cartoon coloured by pLDDT in the four bins used throughout:
<50 orange, 50-70 yellow, 70-90 light blue, >90 blue.
"""
from pymol import cmd
import json, os

PICK = json.load(open("trkstr/pick.json"))
ROUNDS = PICK["rounds"]
ROWS = ["TRKA", "TRKB", "TRKC"]
PL = [(0, 50, "0xF4761F"), (50, 70, "0xF2C230"), (70, 90, "0x7FC4E8"), (90, 101, "0x1F5FBF")]
os.makedirs("trk9", exist_ok=True)


def setup():
    cmd.set("ray_opaque_background", 0); cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2); cmd.set("ambient", 0.22); cmd.set("specular", 0.15)
    cmd.set("surface_quality", 1)
    # Depth cueing is off: the common camera distance below sits farther back than any
    # single row needs, and fog would wash the farther rows out by a different amount.
    cmd.set("depth_cue", 0); cmd.set("ray_trace_fog", 0)
    # Where the surface is cut open (hidden terminal residues, clipping) the ray tracer
    # paints the interior with ray_interior_color, black by default. Match it to the
    # surface grey and light both faces so those openings read as surface, not holes.
    cmd.set("ray_interior_color", "grey72")
    cmd.set("two_sided_lighting", 1)


def build(row):
    cmd.reinitialize(); setup()
    names = []
    for i, rd in enumerate(ROUNDS):
        obj = "%s_%d" % (row, i)
        cmd.load("trkstr/%s_r%02d_%d.cif" % (row, rd, i), obj)
        cmd.remove("%s and hydro" % obj)
        names.append(obj)
    rms = {}
    for obj in names[1:]:
        r = cmd.align("%s and chain A and name CA" % obj,
                      "%s and chain A and name CA" % names[0], cycles=5)
        rms[obj] = [round(r[0], 3), r[1]]
    for obj in names:
        cmd.hide("everything", obj)
        ri = sorted({int(a.resi) for a in cmd.get_model("%s and chain A and name CA" % obj).atom})
        cmd.show("surface", "%s and chain A and resi %d-%d" % (obj, ri[6], ri[-7]))
        cmd.color("grey72", "%s and chain A" % obj)
        cmd.show("cartoon", "%s and chain B" % obj)
        for lo, hi, col in PL:
            cmd.color(col, "%s and chain B and b > %d and b < %d" % (obj, lo, hi))
    cmd.orient("%s and chain A" % names[0])
    cmd.turn("y", 20)
    cmd.zoom("%s or %s or %s" % tuple(names), 3.0, complete=1)
    return names, rms, list(cmd.get_view())


# Pass 1: find the camera distance the widest row needs.
views = {}
for row in ROWS:
    _, _, v = build(row)
    views[row] = v
dist = min(v[11] for v in views.values())          # most negative = farthest camera
near = min(v[15] for v in views.values())
far = max(v[16] for v in views.values())
print("common camera distance %.2f  slab %.2f-%.2f  (per-row %s)"
      % (dist, near, far, [round(v[11], 1) for v in views.values()]))

# Pass 2: render every row at that one magnification.
log = {}
for row in ROWS:
    names, rms, _ = build(row)
    v = views[row][:]
    v[11], v[15], v[16] = dist, near, far
    for i, obj in enumerate(names):
        cmd.disable("all"); cmd.enable(obj); cmd.set_view(v)
        cmd.png("trk9/%s_%d.png" % (row, i), width=1500, height=1250, dpi=300, ray=1)
    log[row] = {"align_rmsd": rms, "view": [round(x, 4) for x in v]}
log["common"] = {"camera_distance": round(dist, 4), "slab": [round(near, 4), round(far, 4)]}
json.dump(log, open("trk9/render_log.json", "w"), indent=1)
print(json.dumps({k: v["align_rmsd"] for k, v in log.items() if k != "common"}))
