import taichi as ti

ti.init(arch=ti.cpu)

@ti.dataclass
class CellTypesEnum:
    normal: float
    sink: float
    blocker: float
CellTypes = CellTypesEnum(0.0, 1.0, 2.0)