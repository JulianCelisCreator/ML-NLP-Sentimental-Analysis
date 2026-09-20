# Troubleshooting

Problemas que ya nos pasaron en este repo, con el diagnóstico real y no el que
parecía obvio. Si te topas con uno nuevo, añádelo aquí.

---

## `Unable to import 'ml_npl.dataset'`

### Síntoma

El editor marca en rojo el import, o `pylint` devuelve:

```
E0401: Unable to import 'ml_npl.dataset' (import-error)
E0611: No name 'dataset' in module 'ml_npl' (no-name-in-module)
```

### Las dos pistas hay que leerlas juntas

Este es el punto importante, y es lo que nos costó una hora la primera vez:

| Error | Qué significa |
| --- | --- |
| Solo `E0401` | No encuentra el paquete `ml_npl`. Problema de **entorno**. |
| `E0401` **y** `E0611` | Sí encuentra `ml_npl`, pero no el módulo `dataset` dentro. Problema de **archivos**. |

Si ves `E0611`, el paquete se está resolviendo bien. Deja de investigar el
intérprete: lo que falla es el contenido.

### Diagnóstico en dos pasos

Lo primero que uno piensa es "VS Code está usando el Python del sistema en vez
del de Poetry". Es una causa real, pero antes de tocar la configuración del
editor haz estas dos comprobaciones, en este orden.

**Paso 1: ¿existe el archivo?**

```bash
ls ML_NPL/ml_npl/
```

Si el `.py` que intentas importar no está en esa lista, no hay nada que
configurar: el archivo se borró o se movió. Salta a la sección de recuperación.

**Paso 2: ¿lo importa el entorno correcto?**

```bash
cd ML_NPL
poetry run python -c "import ml_npl.dataset; print('OK')"
```

- **Imprime `OK`** → el código está bien y el problema es solo del editor.
  `Ctrl+Shift+P` → *Python: Select Interpreter* → elige el que empieza por
  `ml-npl-`.
- **Falla** → lee **qué módulo** nombra el error, que no siempre es el que
  crees:

  | El error dice | Significa |
  | --- | --- |
  | `No module named 'ml_npl.dataset'` | falta el archivo dentro del paquete |
  | `No module named 'ml_npl'` | el paquete no está instalado: `poetry install` |
  | `No module named 'kagglehub'` (o pandas…) | faltan dependencias, no es `ml_npl` |

> No compares contra `python3` a secas para diagnosticar. Ejecutado desde
> `ML_NPL/`, el intérprete del sistema encuentra `ml_npl` igual (está en el
> directorio actual) y falla más tarde por una dependencia, dándote un mensaje
> que apunta al sitio equivocado.

### Qué pasó realmente (19 sep 2026)

`ml_npl/dataset.py` **había desaparecido del disco**. La carpeta del paquete
solo tenía `__init__.py`, así que el import era correcto y el archivo no
existía. `pylint` tenía razón desde el principio.

Comprobación directa:

```bash
ls -la ML_NPL/ml_npl/
```

Si ahí no está el `.py` que intentas importar, no hay nada que configurar.

### Cómo se recuperó

La papelera tenía la carpeta vacía, sin el archivo. Lo que sí lo tenía era **el
índice de git**, porque alguien había hecho `git add .` un rato antes:

```bash
# ver si el índice tiene una copia
git show :ML_NPL/ml_npl/dataset.py | head

# restaurarla
git show :ML_NPL/ml_npl/dataset.py > ML_NPL/ml_npl/dataset.py
```

Ojo: el índice guarda la versión del último `git add`, no la última que
editaste. En nuestro caso faltaban los cambios posteriores y hubo que
rehacerlos a mano.

### Cómo evitarlo

- **Haz `git add` seguido, aunque no commitees.** El índice funcionó como red
  de seguridad cuando la papelera no tenía nada.
- **Al mover carpetas, usa `git mv`** en vez de arrastrar en el explorador de
  archivos. Git conserva el historial y no se pierde nada por el camino.
- **`poetry check` no valida esto.** Solo comprueba la sintaxis del TOML, no
  que la carpeta de `packages` exista. Después de mover archivos, confirma con
  `poetry install` y un import real.

---

## El paquete se instala pero el import sigue fallando

Tras mover carpetas, revisa que `packages` en `ML_NPL/pyproject.toml` apunte a
donde está el código de verdad. Si no coincide, `poetry install` avisa con una
línea fácil de pasar por alto:

```
/ruta/al/paquete does not contain any element
```

Y luego reinstala:

```bash
cd ML_NPL && poetry install
```

El paquete se instala en modo editable mediante un `.pth` dentro del entorno
virtual, que contiene la ruta a `ML_NPL/`. Para verlo:

```bash
cat "$(poetry run python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"/ml_npl.pth
```
