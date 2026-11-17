#!/usr/bin/env python3
"""Inline the viewer meshes into a single self-contained HTML page."""
import base64, os
here = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(here, "viewer_template.html")).read()
m = open(os.path.join(here, "out/viewer/manifest.json")).read()
b = base64.b64encode(open(os.path.join(here, "out/viewer/meshes.bin"), "rb").read()).decode()
open(os.path.join(here, "out/rocket_viewer.html"), "w").write(t.replace("__MANIFEST__", m).replace("__MESHDATA__", b))
print("wrote out/rocket_viewer.html", round(os.path.getsize(os.path.join(here, "out/rocket_viewer.html")) / 1e6, 2), "MB")
