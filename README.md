# voronzov — Graph Neural Networks (GNN)

Исследовательский проект по применению графовых нейронных сетей (Graph Neural Networks) на Python.

---

## Структура проекта

```
voronzov/
└── GNN/
    ├── app.py            # Точка входа / основное приложение
    ├── demo.py           # Демонстрация работы модели
    ├── gnn_model.py      # Архитектура графовой нейронной сети
    ├── metrics.py        # Метрики оценки качества модели
    ├── objects.py        # Вспомогательные объекты и структуры данных
    ├── optimizer.py      # Настройка оптимизатора обучения
    └── requirements.txt  # Зависимости проекта
```

---

## Быстрый старт

**1. Клонирование репозитория**

```bash
git clone https://github.com/Silrec/voronzov.git
cd voronzov
```

**2. Установка зависимостей**

```bash
pip install -r GNN/requirements.txt
```

**3. Запуск**

```bash
python GNN/app.py
```

Для запуска демонстрации:

```bash
python GNN/demo.py
```

---

## Сборка в EXE (Windows)

**1. Установить PyInstaller**

```bash
pip install pyinstaller
```

**2. Собрать исполняемый файл**

```bash
pyinstaller --onefile GNN/app.py
```

Готовый `.exe` появится в папке `dist/`.

---

## Технологии

- **Язык:** Python 3
- **Область:** Machine Learning / Graph Neural Networks

---

## Авторы

- **[Silrec](https://github.com/Silrec)**
