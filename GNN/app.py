"""
GNN Layout Planner — minimal tkinter UI.
"""
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors

from objects import PlacementObject, Road, OBJECT_TYPES, SHAPE_TYPES
from gnn_model import LayoutGNN
from optimizer import run_optimization
from metrics import full_report

# ── colour palette ──────────────────────────────────────────────────────────
TYPE_COLORS = {
    "Производство": "#4e79a7",
    "Склад":        "#f28e2b",
    "Офис":         "#e15759",
    "Энергетика":   "#76b7b2",
    "Утилиты":      "#59a14f",
    "Другое":       "#b07aa1",
}
ROAD_COLOR  = "#888888"
BLOCK_COLOR = "#dddddd"
CONN_COLOR  = "#ff6600"
GRID_COLOR  = "#eeeeee"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GNN Layout Planner")
        self.geometry("1300x820")
        self.resizable(True, True)
        self.configure(bg="#f5f5f5")

        self.objects: list[PlacementObject] = []
        self.roads:   list[Road]            = []
        self.blocks:  list                  = []
        self._next_id = 1
        self._next_road_id = 1
        self._type_map = {t: i for i, t in enumerate(OBJECT_TYPES)}
        self._model = LayoutGNN(hidden_dim=64, n_layers=3)
        self._optimizing = False

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # Left panel
        left = tk.Frame(self, bg="#f5f5f5", width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        left.pack_propagate(False)

        # Right: canvas
        right = tk.Frame(self, bg="#f5f5f5")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._build_left_panel(left)
        self._build_canvas(right)

    def _section(self, parent, title):
        f = tk.LabelFrame(parent, text=title, bg="#f5f5f5",
                          font=("Segoe UI", 9, "bold"), padx=6, pady=4)
        f.pack(fill=tk.X, pady=(0, 6))
        return f

    def _build_left_panel(self, parent):
        # ── Field settings ───────────────────────────────────────────────────
        sec = self._section(parent, "Поле (м)")
        for i, (lbl, var, default) in enumerate([
            ("Ширина:", "_field_w", "300"),
            ("Высота:", "_field_h", "300"),
            ("Шаг сетки (м):", "_grid_step", "5"),
            ("Размер квартала (м):", "_block_size", "60"),
        ]):
            tk.Label(sec, text=lbl, bg="#f5f5f5", font=("Segoe UI", 9)).grid(
                row=i, column=0, sticky=tk.W)
            entry = tk.Entry(sec, width=8, font=("Segoe UI", 9))
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky=tk.W, padx=4)
            setattr(self, var, entry)

        # ── Add object ───────────────────────────────────────────────────────
        sec2 = self._section(parent, "Добавить объект")
        fields = [
            ("Название:", "_inp_name", "Объект"),
            ("Ширина (м):", "_inp_w", "20"),
            ("Высота (м):", "_inp_h", "15"),
            ("Мин. пожар. расст. (м):", "_inp_fire", "10"),
        ]
        for i, (lbl, var, default) in enumerate(fields):
            tk.Label(sec2, text=lbl, bg="#f5f5f5", font=("Segoe UI", 9)).grid(
                row=i, column=0, sticky=tk.W)
            e = tk.Entry(sec2, width=10, font=("Segoe UI", 9))
            e.insert(0, default)
            e.grid(row=i, column=1, sticky=tk.W, padx=4)
            setattr(self, var, e)

        # Shape
        tk.Label(sec2, text="Форма:", bg="#f5f5f5", font=("Segoe UI", 9)).grid(
            row=len(fields), column=0, sticky=tk.W)
        self._inp_shape = ttk.Combobox(sec2, values=SHAPE_TYPES, state="readonly",
                                       width=12, font=("Segoe UI", 9))
        self._inp_shape.set(SHAPE_TYPES[0])
        self._inp_shape.grid(row=len(fields), column=1, sticky=tk.W, padx=4)

        # Type
        tk.Label(sec2, text="Тип:", bg="#f5f5f5", font=("Segoe UI", 9)).grid(
            row=len(fields)+1, column=0, sticky=tk.W)
        self._inp_type = ttk.Combobox(sec2, values=OBJECT_TYPES, state="readonly",
                                      width=14, font=("Segoe UI", 9))
        self._inp_type.set(OBJECT_TYPES[0])
        self._inp_type.grid(row=len(fields)+1, column=1, sticky=tk.W, padx=4)

        tk.Button(sec2, text="+ Добавить объект", command=self._add_object,
                  font=("Segoe UI", 9), bg="#4e79a7", fg="white",
                  relief=tk.FLAT, padx=6).grid(row=len(fields)+2, columnspan=2,
                                                pady=4, sticky=tk.W)

        # ── Object list ──────────────────────────────────────────────────────
        sec3 = self._section(parent, "Объекты")
        self._obj_list = tk.Listbox(sec3, height=8, font=("Segoe UI", 9),
                                    selectmode=tk.EXTENDED, exportselection=False)
        self._obj_list.pack(fill=tk.X)
        btn_row = tk.Frame(sec3, bg="#f5f5f5")
        btn_row.pack(fill=tk.X, pady=2)
        tk.Button(btn_row, text="Связать выбранные", command=self._connect_selected,
                  font=("Segoe UI", 8), relief=tk.FLAT, bg="#59a14f", fg="white",
                  padx=4).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Удалить", command=self._delete_selected,
                  font=("Segoe UI", 8), relief=tk.FLAT, bg="#e15759", fg="white",
                  padx=4).pack(side=tk.LEFT, padx=4)

        # ── Roads ────────────────────────────────────────────────────────────
        sec4 = self._section(parent, "Дороги")
        road_row = tk.Frame(sec4, bg="#f5f5f5")
        road_row.pack(fill=tk.X)
        for lbl, var, default in [
            ("x1", "_rd_x1", "0"), ("y1", "_rd_y1", "150"),
            ("x2", "_rd_x2", "300"), ("y2", "_rd_y2", "150"),
        ]:
            tk.Label(road_row, text=lbl, bg="#f5f5f5", font=("Segoe UI", 8)).pack(side=tk.LEFT)
            e = tk.Entry(road_row, width=5, font=("Segoe UI", 8))
            e.insert(0, default)
            e.pack(side=tk.LEFT, padx=1)
            setattr(self, var, e)
        tk.Button(sec4, text="+ Добавить дорогу", command=self._add_road,
                  font=("Segoe UI", 9), bg="#76b7b2", fg="white",
                  relief=tk.FLAT, padx=6).pack(anchor=tk.W, pady=2)
        self._road_list = tk.Listbox(sec4, height=3, font=("Segoe UI", 9))
        self._road_list.pack(fill=tk.X)
        tk.Button(sec4, text="Удалить дорогу", command=self._delete_road,
                  font=("Segoe UI", 8), relief=tk.FLAT, bg="#e15759", fg="white",
                  padx=4).pack(anchor=tk.W)

        # ── Run ──────────────────────────────────────────────────────────────
        sec5 = self._section(parent, "Оптимизация")
        self._max_iter_var = tk.StringVar(value="300")
        tk.Label(sec5, text="Итераций:", bg="#f5f5f5", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Entry(sec5, textvariable=self._max_iter_var, width=6,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)

        self._run_btn = tk.Button(parent, text="▶  Оптимизировать",
                                  command=self._run_optimizer,
                                  font=("Segoe UI", 10, "bold"),
                                  bg="#2c7bb6", fg="white", relief=tk.FLAT,
                                  pady=6)
        self._run_btn.pack(fill=tk.X, pady=4)

        self._progress_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._progress_var, bg="#f5f5f5",
                 font=("Segoe UI", 8), fg="#555").pack()

        # ── Metrics ──────────────────────────────────────────────────────────
        self._metrics_frame = self._section(parent, "Метрики")
        self._metric_labels = {}
        metric_names = [
            ("overall",        "Общий балл"),
            ("fire_safety",    "Пожарная безопасность"),
            ("tech_proximity", "Технологические связи"),
            ("road_access",    "Доступ к дорогам"),
            ("blocks",         "Кварталы"),
            ("orientation",    "Ориентация"),
        ]
        for key, label in metric_names:
            row = tk.Frame(self._metrics_frame, bg="#f5f5f5")
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label + ":", bg="#f5f5f5",
                     font=("Segoe UI", 8), width=22, anchor=tk.W).pack(side=tk.LEFT)
            val_lbl = tk.Label(row, text="—", bg="#f5f5f5",
                               font=("Segoe UI", 8, "bold"), width=6)
            val_lbl.pack(side=tk.LEFT)
            self._metric_labels[key] = val_lbl

    def _build_canvas(self, parent):
        self._fig, self._ax = plt.subplots(figsize=(8, 7))
        self._fig.patch.set_facecolor("#f5f5f5")
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty()

    # ── Actions ──────────────────────────────────────────────────────────────

    def _add_object(self):
        try:
            name  = self._inp_name.get().strip() or f"Объект {self._next_id}"
            w     = float(self._inp_w.get())
            h     = float(self._inp_h.get())
            fire  = float(self._inp_fire.get())
            shape = self._inp_shape.get()
            otype = self._inp_type.get()
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте числовые поля")
            return

        if shape == "circle":
            h = w  # diameter

        obj = PlacementObject(
            id=self._next_id, name=name, shape=shape,
            width=w, height=h, object_type=otype, fire_distance=fire,
        )
        self._next_id += 1
        self.objects.append(obj)
        self._obj_list.insert(tk.END, f"[{obj.id}] {obj.name}  ({otype}, {shape}, {w}×{h}м)")
        self._inp_name.delete(0, tk.END)
        self._inp_name.insert(0, f"Объект {self._next_id}")

    def _connect_selected(self):
        sel = self._obj_list.curselection()
        if len(sel) < 2:
            messagebox.showinfo("Связи", "Выберите минимум 2 объекта")
            return
        ids = [self.objects[i].id for i in sel]
        for obj in self.objects:
            if obj.id in ids:
                for oid in ids:
                    if oid != obj.id and oid not in obj.connections:
                        obj.connections.append(oid)
        messagebox.showinfo("Связи", f"Связаны объекты: {ids}")

    def _delete_selected(self):
        sel = list(self._obj_list.curselection())[::-1]
        for i in sel:
            obj = self.objects.pop(i)
            self._obj_list.delete(i)
            # Remove references
            for o in self.objects:
                o.connections = [c for c in o.connections if c != obj.id]

    def _add_road(self):
        try:
            x1 = float(self._rd_x1.get())
            y1 = float(self._rd_y1.get())
            x2 = float(self._rd_x2.get())
            y2 = float(self._rd_y2.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте координаты дороги")
            return
        road = Road(id=self._next_road_id, x1=x1, y1=y1, x2=x2, y2=y2)
        self._next_road_id += 1
        self.roads.append(road)
        self._road_list.insert(tk.END, f"Дорога {road.id}: ({x1},{y1})→({x2},{y2})")

    def _delete_road(self):
        sel = list(self._road_list.curselection())[::-1]
        for i in sel:
            self.roads.pop(i)
            self._road_list.delete(i)

    def _run_optimizer(self):
        if self._optimizing:
            return
        if not self.objects:
            messagebox.showinfo("Пусто", "Добавьте хотя бы один объект")
            return
        self._optimizing = True
        self._run_btn.config(state=tk.DISABLED)
        self._progress_var.set("Оптимизация...")

        try:
            fw = float(self._field_w.get())
            fh = float(self._field_h.get())
            block_size = float(self._block_size.get())
            max_iter = int(self._max_iter_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте параметры поля")
            self._optimizing = False
            self._run_btn.config(state=tk.NORMAL)
            return

        bounds = (0.0, 0.0, fw, fh)

        def task():
            losses = []

            def cb(loss):
                losses.append(loss)
                if len(losses) % 20 == 0:
                    self.after(0, self._progress_var.set,
                               f"Итерация ~{len(losses)*2}, loss={loss:.1f}")

            self.blocks = run_optimization(
                self.objects, self.roads, bounds,
                self._type_map, self._model,
                block_size=block_size, max_iter=max_iter, progress_cb=cb,
            )
            self.after(0, self._on_opt_done, bounds)

        threading.Thread(target=task, daemon=True).start()

    def _on_opt_done(self, bounds):
        self._optimizing = False
        self._run_btn.config(state=tk.NORMAL)
        self._progress_var.set("Готово")
        self._draw_layout(bounds)
        self._update_metrics()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_empty(self):
        self._ax.clear()
        self._ax.set_facecolor("#fafafa")
        self._ax.set_title("Поле пустое — добавьте объекты и запустите оптимизацию",
                            fontsize=9, color="#888")
        self._canvas.draw()

    def _draw_layout(self, bounds):
        ax = self._ax
        ax.clear()
        ax.set_facecolor("#fafafa")
        xmin, ymin, xmax, ymax = bounds

        try:
            grid_step = float(self._grid_step.get())
        except ValueError:
            grid_step = 5.0

        # Grid
        for gx in np.arange(xmin, xmax + grid_step, grid_step):
            ax.axvline(gx, color=GRID_COLOR, lw=0.4, zorder=0)
        for gy in np.arange(ymin, ymax + grid_step, grid_step):
            ax.axhline(gy, color=GRID_COLOR, lw=0.4, zorder=0)

        # Blocks
        for block in self.blocks:
            if not block.objects:
                continue
            rect = mpatches.Rectangle(
                (block.x, block.y), block.width, block.height,
                linewidth=1, edgecolor="#aaaaaa", facecolor=BLOCK_COLOR,
                alpha=0.3, zorder=1,
            )
            ax.add_patch(rect)
            n = len(block.objects)
            color = "#59a14f" if 4 <= n <= 6 else "#e15759"
            ax.text(block.x + 2, block.y + 2, str(n),
                    fontsize=7, color=color, zorder=5)

        # Roads
        for road in self.roads:
            ax.plot([road.x1, road.x2], [road.y1, road.y2],
                    color=ROAD_COLOR, lw=road.width * 0.4 + 2, solid_capstyle="round",
                    zorder=2)
            ax.plot([road.x1, road.x2], [road.y1, road.y2],
                    color="white", lw=1, linestyle="--", zorder=3)

        # Connections
        id_map = {obj.id: obj for obj in self.objects}
        drawn_edges = set()
        for obj in self.objects:
            for cid in obj.connections:
                if cid in id_map:
                    edge = tuple(sorted([obj.id, cid]))
                    if edge not in drawn_edges:
                        other = id_map[cid]
                        ax.annotate("",
                            xy=(other.x, other.y), xytext=(obj.x, obj.y),
                            arrowprops=dict(arrowstyle="-", color=CONN_COLOR,
                                            lw=1.2, linestyle="dashed"),
                            zorder=4)
                        drawn_edges.add(edge)

        # Objects
        for obj in self.objects:
            color = TYPE_COLORS.get(obj.object_type, "#999999")
            if obj.shape == "circle":
                circle = plt.Circle((obj.x, obj.y), obj.half_w(),
                                    color=color, alpha=0.85, zorder=6)
                ax.add_patch(circle)
            else:
                rect = mpatches.Rectangle(
                    (obj.x - obj.half_w(), obj.y - obj.half_h()),
                    obj.width, obj.height,
                    color=color, alpha=0.85, zorder=6,
                )
                ax.add_patch(rect)
            ax.text(obj.x, obj.y, obj.name, ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold", zorder=7,
                    wrap=True)

        # Legend
        handles = [mpatches.Patch(color=c, label=t)
                   for t, c in TYPE_COLORS.items()
                   if any(o.object_type == t for o in self.objects)]
        if handles:
            ax.legend(handles=handles, fontsize=7, loc="upper right",
                      framealpha=0.7)

        ax.set_xlim(xmin - 5, xmax + 5)
        ax.set_ylim(ymin - 5, ymax + 5)
        ax.set_aspect("equal")
        ax.set_xlabel("X (м)", fontsize=8)
        ax.set_ylabel("Y (м)", fontsize=8)
        ax.set_title("Расположение объектов", fontsize=10)
        self._fig.tight_layout()
        self._canvas.draw()

    def _update_metrics(self):
        if not self.objects:
            return
        report = full_report(self.objects, self.roads, self.blocks)
        for key, lbl in self._metric_labels.items():
            if key == "overall":
                val = report.get("overall", 0)
            else:
                val = report.get(key, {}).get("score", 0)
            color = "#59a14f" if val >= 75 else ("#f28e2b" if val >= 50 else "#e15759")
            lbl.config(text=f"{val:.0f}", fg=color)


if __name__ == "__main__":
    app = App()
    app.mainloop()
