#!/usr/bin/env python3
"""Build the viewer page from the template and the CAD meshes.

Writes two files:
  out/rocket_viewer.html  - page fragment published as the Claude artifact (loads three.js from a CDN)
  project_haddock.html     - complete standalone HTML file with three.js inlined; open it in any browser
"""
import base64
import os
here = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(here, "viewer_template.html")).read()
m = open(os.path.join(here, "out/viewer/manifest.json")).read()
b = base64.b64encode(open(os.path.join(here, "out/viewer/meshes.bin"), "rb").read()).decode()
page = t.replace("__MANIFEST__", m).replace("__MESHDATA__", b)
p = os.path.join(here, "out/rocket_viewer.html")
open(p, "w").write(page)
print("wrote out/rocket_viewer.html", round(os.path.getsize(p) / 1e6, 2), "MB")

tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
assert tag in page
three = open(os.path.join(here, "vendor/three.min.js")).read()
head_end = page.index("</style>") + len("</style>")
standalone = ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
              '<style>html{color-scheme:light}body{margin:0}img{max-width:100%}</style>\n'
              + page[:head_end] + "\n</head>\n<body>\n"
              + page[head_end:].replace(tag, "<script>" + three + "</script>")
              + "\n</body>\n</html>\n")
s = os.path.join(here, "project_haddock.html")
open(s, "w").write(standalone)
print("wrote project_haddock.html", round(os.path.getsize(s) / 1e6, 2), "MB")
