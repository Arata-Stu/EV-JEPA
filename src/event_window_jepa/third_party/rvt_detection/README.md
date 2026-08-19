# Vendored RVT detection components

This directory contains only the components needed by EV-JEPA's Gen1 detector:

- the RVT-modified YOLOX head and its supporting loss/blocks/postprocessing;
- the Prophesee filtering and COCO evaluation used by RVT.

Source: <https://github.com/uzh-rpg/RVT>, commit
`b80f5683a6e2d5de65d4bde8105d796ccb50dbb1`. The distributed RVT license is
reproduced in `LICENSE`. Original copyright headers from YOLOX, Megvii, and
Prophesee are retained in the source files.

The files are kept local so users do not need a second repository checkout and
so the detection protocol cannot silently change when RVT upstream changes.
