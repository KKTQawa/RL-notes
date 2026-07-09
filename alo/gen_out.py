import os
import re
import ast

XUANCE_ROOT = r"D:\.conda\envs\xuance_env\Lib\site-packages\xuance"
TEST_FILE = r"E:\RL\baseRL\xuance_code\test_xuance.py"
OUTPUT_FILE = r"E:\RL\baseRL\xuance_code\OUT.py"
SKIP_DIRS = {'tensorflow', 'mindspore'}

FUTURE_IMPORTS = set()

def collect_xuance_files(root):
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = rel.split(os.sep)
        if any(skip in parts for skip in SKIP_DIRS):
            continue
        for fn in filenames:
            if fn.endswith('.py'):
                py_files.append(os.path.join(dirpath, fn))
    return sorted(py_files)

def file_to_modname(fp, root):
    rel = os.path.relpath(fp, root)
    modname = rel.replace(os.sep, '.')[:-3]
    if modname.endswith('.__init__'):
        modname = modname[:-9]
    return modname

def get_xuance_deps(fp, root):
    with open(fp, 'r', encoding='utf-8') as f:
        code = f.read()
    deps = set()
    current_mod = file_to_modname(fp, root)
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('xuance'):
                        deps.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                if node.module.startswith('xuance'):
                    deps.add(node.module)
                elif node.level > 0:
                    parts = current_mod.split('.')
                    # level=1: same package, level=2: parent package, etc.
                    if node.level <= len(parts):
                        base = '.'.join(parts[:len(parts) - node.level + 1])
                        if node.module:
                            full = base + '.' + node.module
                        else:
                            full = base
                        if full.startswith('xuance'):
                            deps.add(full)
    except SyntaxError:
        pass
    return deps

def modname_to_filepath(modname, root):
    if '.' in modname:
        pkg, *parts = modname.split('.')
        if pkg != 'xuance':
            return None
        rel_path = os.path.join(*parts)
    else:
        rel_path = ''
    base = os.path.join(root, rel_path)
    init_py = os.path.join(base, '__init__.py')
    if os.path.exists(init_py):
        return init_py
    mod_py = base + '.py'
    if os.path.exists(mod_py):
        return mod_py
    return None

def get_indent(line):
    return len(line) - len(line.lstrip())

def is_xuance_import(stripped):
    if re.match(r'^(import\s+xuance\b|from\s+xuance\b)', stripped):
        return True
    if re.match(r'^from\s+\.', stripped):
        return True
    return False

def is_import_line(stripped):
    return bool(re.match(r'^(import\s+|from\s+)', stripped))

def process_file(filepath):
    global FUTURE_IMPORTS
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith('#'):
            i += 1
            continue

        if stripped.startswith('from __future__'):
            indent = get_indent(line)
            if indent == 0:
                open_paren = stripped.count('(')
                close_paren = stripped.count(')')
                if open_paren > close_paren:
                    j = i + 1
                    while j < len(lines):
                        s = lines[j].strip()
                        open_paren += s.count('(')
                        close_paren += s.count(')')
                        if open_paren <= close_paren:
                            j += 1
                            break
                        j += 1
                    FUTURE_IMPORTS.add(stripped)
                    i = j
                    continue
                FUTURE_IMPORTS.add(stripped)
                i += 1
                continue

        if is_import_line(stripped):
            indent = get_indent(line)
            is_xuance = is_xuance_import(stripped)

            open_paren = stripped.count('(')
            close_paren = stripped.count(')')
            if open_paren > close_paren:
                j = i + 1
                while j < len(lines):
                    s = lines[j].strip()
                    open_paren += s.count('(')
                    close_paren += s.count(')')
                    if open_paren <= close_paren:
                        j += 1
                        break
                    j += 1
                if not is_xuance:
                    for k in range(i, j):
                        result.append(lines[k])
                i = j
                continue

            if stripped.endswith('\\'):
                j = i + 1
                while j < len(lines):
                    s = lines[j].strip()
                    if '#' in s:
                        s = s.split('#')[0].strip()
                    if s.endswith('\\'):
                        j += 1
                    else:
                        j += 1
                        break
                if not is_xuance:
                    for k in range(i, j):
                        result.append(lines[k])
                i = j
                continue

            if is_xuance:
                if indent > 0:
                    result.append(' ' * indent + 'pass')
                i += 1
                continue

        if '#' in line:
            idx = line.find('#')
            in_single = False
            in_double = False
            for c in line[:idx]:
                if c == "'" and not in_double:
                    in_single = not in_single
                elif c == '"' and not in_single:
                    in_double = not in_double
            if not in_single and not in_double:
                result.append(line[:idx].rstrip())
                i += 1
                continue

        result.append(line)
        i += 1

    return '\n'.join(result)

# Manual ordering to resolve cross-file dependencies
# Files in each directory that must come before others
DIR_FILE_ORDER = {
    'common': [
        'common_tools.py',
        'segtree_tool.py',
        'statistic_tools.py',
        'callback.py',
        'memory_tools.py',
        'memory_tools_marl.py',
        'offline_util.py',
        'memory_offline.py',
    ],
    os.path.join('environment', 'utils'): [
        'base.py',
        'shapes.py',
        'wrapper.py',
    ],
    os.path.join('environment', 'vector_envs'): [
        'env_utils.py',
        'vector_env.py',
    ],
    os.path.join('environment', 'vector_envs', 'dummy'): [
        'dummy_vec_env.py',
        'dummy_vec_maenv.py',
    ],
    os.path.join('environment', 'vector_envs', 'subprocess'): [
        'subproc_vec_env.py',
        'subproc_vec_maenv.py',
    ],
    os.path.join('torch', 'representations'): [
        'mlp.py',
        'rnn.py',
        'cnn.py',
        'vit.py',
        'world_model.py',
        'world_model_v2.py',
    ],
    os.path.join('torch', 'policies'): [
        'core.py',
        'deterministic.py',
        'deterministic_marl.py',
        'categorical.py',
        'categorical_marl.py',
        'gaussian.py',
        'gaussian_marl.py',
        'coordination_graph.py',
        'dreamer.py',
    ],
    os.path.join('torch', 'utils'): [
        'device.py',
        'value_norm.py',
        'operations.py',
        'layers.py',
        'distributions.py',
        'harmonizer.py',
        'tensor_env.py',
        'tensor_memory.py',
        'tensor_statistics.py',
    ],
    os.path.join('torch', 'learners'): [
        'learner.py',
    ],
    os.path.join('torch', 'learners', 'multi_agent_rl'): [
        'qmix_learner.py',
    ],
    os.path.join('torch', 'agents', 'base'): [
        'agent.py',
        'agents_marl.py',
    ],
    os.path.join('torch', 'agents', 'core'): [
        'off_policy.py',
        'off_policy_marl.py',
        'on_policy.py',
        'on_policy_marl.py',
        'offline.py',
    ],
}

DIR_PRIORITY = [
    'common',
    'configs',
    os.path.join('environment', 'utils'),
    os.path.join('environment', 'vector_envs'),
    os.path.join('environment', 'vector_envs', 'dummy'),
    os.path.join('environment', 'vector_envs', 'subprocess'),
    os.path.join('environment', 'single_agent_env'),
    os.path.join('environment', 'multi_agent_env'),
    'environment',
    os.path.join('torch', 'utils'),
    os.path.join('torch', 'representations'),
    os.path.join('torch', 'policies'),
    os.path.join('torch', 'learners'),
    os.path.join('torch', 'learners', 'multi_agent_rl'),
    os.path.join('torch', 'learners', 'policy_gradient'),
    os.path.join('torch', 'learners', 'qlearning_family'),
    os.path.join('torch', 'learners', 'contrastive_unsupervised_rl'),
    os.path.join('torch', 'learners', 'model_based'),
    os.path.join('torch', 'learners', 'offline'),
    os.path.join('torch', 'agents', 'base'),
    os.path.join('torch', 'agents', 'core'),
    os.path.join('torch', 'agents', 'multi_agent_rl'),
    os.path.join('torch', 'agents'),
    'torch',
    'engine',
    ''
]

LATE_FILES = {'__init__.py'}

def file_sort_key(fp, root):
    rel = os.path.relpath(fp, root)
    dirname = os.path.dirname(rel)
    basename = os.path.basename(rel)

    dir_priority = len(DIR_PRIORITY)
    for i, dp in enumerate(DIR_PRIORITY):
        if dirname == dp:
            dir_priority = i
            break

    file_order = 999
    if dirname in DIR_FILE_ORDER:
        order_map = {f: i for i, f in enumerate(DIR_FILE_ORDER[dirname])}
        if basename in order_map:
            file_order = order_map[basename]
        elif basename in LATE_FILES:
            file_order = 998
        else:
            file_order = 500

    return (dir_priority, file_order, basename)

def sort_files(files, root):
    return sorted(files, key=lambda fp: file_sort_key(fp, root))

def main():
    py_files = collect_xuance_files(XUANCE_ROOT)
    print(f"Total files found: {len(py_files)}")

    sorted_files = sort_files(py_files, XUANCE_ROOT)
    print(f"Sorted files: {len(sorted_files)}")

    all_code_parts = []
    for fp in sorted_files:
        relpath = os.path.relpath(fp, XUANCE_ROOT)
        code = process_file(fp)
        all_code_parts.append(f'# === {relpath} ===\n{code}')

    future_header = '\n'.join(sorted(FUTURE_IMPORTS))
    combined = future_header + '\n\n' + '\n\n'.join(all_code_parts)

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_code = f.read()

    test_lines = test_code.split('\n')
    test_cleaned = []
    for line in test_lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if is_xuance_import(stripped):
            continue
        test_cleaned.append(line)
    test_cleaned = '\n'.join(test_cleaned)

    combined += '\n\n# === test_xuance.py ===\n' + test_cleaned

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(combined)

    print(f"Written {len(combined)} bytes to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
