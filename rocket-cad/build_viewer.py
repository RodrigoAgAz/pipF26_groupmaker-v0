#!/usr/bin/env python3
"""Build the Project Haddock pages from the template and the CAD meshes.

Two variants, each as an artifact fragment (three.js from a CDN) and as a standalone file (three.js inlined):
  full  out/rocket_viewer.html   project_haddock.html        model, lists, hover calculator
  tour  out/haddock_tour.html    project_haddock_tour.html   only the full-screen model and its part descriptions
"""
import base64
import os
import re

here = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(here, "viewer_template.html")).read()
m = open(os.path.join(here, "out/viewer/manifest.json")).read()
b = base64.b64encode(open(os.path.join(here, "out/viewer/meshes.bin"), "rb").read()).decode()
three = open(os.path.join(here, "vendor/three.min.js")).read()
tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'

for tour, frag, alone in ((False, "out/rocket_viewer.html", "project_haddock.html"),
                          (True, "out/haddock_tour.html", "project_haddock_tour.html")):
    p = t
    if tour:
        p = re.sub(r"[ \t]*<!--full-->.*?<!--/full-->\n?", "", p, flags=re.S)
        p = p.replace("<title>Project Haddock</title>", "<title>Project Haddock Tour</title>")
    else:
        p = p.replace("<!--full-->", "").replace("<!--/full-->", "")
    p = p.replace("__VARIANT__", "tour" if tour else "full")
    page = p.replace("__MANIFEST__", m).replace("__MESHDATA__", b)
    assert tag in page
    open(os.path.join(here, frag), "w").write(page)
    head_end = page.index("</style>") + len("</style>")
    standalone = ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                  '<style>html{color-scheme:light}body{margin:0}img{max-width:100%}</style>\n'
                  + page[:head_end] + "\n</head>\n<body>\n"
                  + page[head_end:].replace(tag, "<script>" + three + "</script>") + "\n</body>\n</html>\n")
    open(os.path.join(here, alone), "w").write(standalone)
    print("wrote", frag, "and", alone, round(os.path.getsize(os.path.join(here, alone)) / 1e6, 2), "MB")
