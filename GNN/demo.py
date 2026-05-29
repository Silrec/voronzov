"""
Запустить с примером данных — открывает UI с заполненными объектами.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app import App
from objects import PlacementObject, Road

app = App()

# Pre-fill some demo objects
demo_objects = [
    PlacementObject(app._next_id,   "Цех А",       "rectangle", 30, 20, "Производство", 15, []),
    PlacementObject(app._next_id+1, "Цех Б",       "rectangle", 25, 20, "Производство", 15, []),
    PlacementObject(app._next_id+2, "Склад 1",     "rectangle", 20, 15, "Склад",        12, []),
    PlacementObject(app._next_id+3, "Склад 2",     "rectangle", 20, 15, "Склад",        12, []),
    PlacementObject(app._next_id+4, "Офис",        "rectangle", 15, 10, "Офис",          8, []),
    PlacementObject(app._next_id+5, "Котельная",   "circle",    12, 12, "Энергетика",   20, []),
    PlacementObject(app._next_id+6, "Трансформатор","circle",    8,  8, "Энергетика",   20, []),
    PlacementObject(app._next_id+7, "КПП",         "rectangle", 6,  4, "Другое",        5, []),
]

# Set up tech connections: Цех А ↔ Склад 1, Цех Б ↔ Склад 2
demo_objects[0].connections = [demo_objects[2].id]
demo_objects[2].connections = [demo_objects[0].id]
demo_objects[1].connections = [demo_objects[3].id]
demo_objects[3].connections = [demo_objects[1].id]

for obj in demo_objects:
    app.objects.append(obj)
    app._next_id += 1
    app._obj_list.insert("end", f"[{obj.id}] {obj.name}  ({obj.object_type}, {obj.shape}, {obj.width}×{obj.height}м)")

# Add a demo road
road = Road(1, 0, 150, 300, 150, 6)
app.roads.append(road)
app._road_list.insert("end", f"Дорога 1: (0,150)→(300,150)")
app._next_road_id += 1

app.mainloop()
