"""oahm — Ocean-Atmosphere-Hydro Models.

Численные модели динамики океана, взаимодействия океана и атмосферы
и трансформации речного стока (портфолио-проект, переработка учебных
лабораторных работ на чистый Python).
"""

from . import box_model, ekman, long_wave, runoff

__all__ = ["box_model", "ekman", "long_wave", "runoff"]
__version__ = "1.0.0"
