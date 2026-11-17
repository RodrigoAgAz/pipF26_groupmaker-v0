#!/usr/bin/env python3
"""Inline the shape data into a single self-contained viewer page."""
import os
here = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(here, "viewer_template.html")).read()
d = open(os.path.join(here, "out/viewer/shape.json")).read()
p = os.path.join(here, "out/rocket_viewer.html")
open(p, "w").write(t.replace("__DATA__", d))
print("wrote out/rocket_viewer.html", round(os.path.getsize(p) / 1e3, 1), "kB")
