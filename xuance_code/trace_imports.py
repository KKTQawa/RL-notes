import sys, os, ast, json

XUANCE_PATH = r"D:\.conda\envs\xuance_env\Lib\site-packages\xuance"

def collect_module_code(module_path, depth=0):
    """Recursively collect code from xuance module files by tracing imports."""
    result = {}
    if not os.path.exists(module_path):
        return result
    if os.path.isfile(module_path):
        if module_path.endswith('__init__.py'):
            result[module_path] = read_and_trace_imports(module_path)
        else:
            result[module_path] = read_and_trace_imports(module_path)
        return result
    # Read __init__ first
    init_file = os.path.join(module_path, '__init__.py')
    if os.path.exists(init_file):
        result[init_file] = read_and_trace_imports(init_file)
    for item in sorted(os.listdir(module_path)):
        item_path = os.path.join(module_path, item)
        if item == '__init__.py':
            continue
        if item.endswith('.py'):
            result[item_path] = read_and_trace_imports(item_path)
    return result

def read_and_trace_imports(file_path):
    """Read a Python file, extract imports and code structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        imports = []
        classes = {}
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(('import', alias.name, alias.asname))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = [(alias.name, alias.asname) for alias in node.names]
                imports.append(('from', module, names))
            elif isinstance(node, ast.ClassDef):
                class_code = ast.get_source_segment(content, node)
                classes[node.name] = class_code
            elif isinstance(node, ast.FunctionDef):
                func_code = ast.get_source_segment(content, node)
                functions[node.name] = func_code
        return {
            'imports': imports,
            'classes': classes,
            'functions': functions,
            'full_content': content
        }
    except Exception as e:
        return {'error': str(e), 'imports': [], 'classes': {}, 'functions': {}, 'full_content': ''}

# Get all xuance modules that were loaded
print("Collecting xuance modules...")
import xuance

xuance_modules = sorted([m for m in sys.modules.keys() if m.startswith('xuance')])
print(f"Found {len(xuance_modules)} xuance modules")
for m in xuance_modules:
    mod = sys.modules.get(m)
    if mod and hasattr(mod, '__file__') and mod.__file__:
        fp = mod.__file__
        if fp.endswith('.py'):
            info = read_and_trace_imports(fp)
            print(f"{fp}: {len(info.get('classes', {}))} classes, {len(info.get('functions', {}))} funcs")
        elif fp.endswith('.cpython-win_amd64.pyd'):
            print(f"{m}: {fp} (binary - skipped)")