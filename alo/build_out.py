import sys, os, ast, types, re

sys.path.insert(0, r"D:\.conda\envs\xuance_env\Lib\site-packages")
XUANCE_PATH = r"D:\.conda\envs\xuance_env\Lib\site-packages\xuance"
OUT_FILE = r"E:\RL\baseRL\xuance_code\OUT.py"

import xuance

loaded_files = {}
module_code = {}
module_order = []

for mod_name in sorted(sys.modules):
    if not mod_name.startswith('xuance'):
        continue
    mod = sys.modules[mod_name]
    if not hasattr(mod, '__file__') or not mod.__file__:
        continue
    fp = mod.__file__
    if not fp.endswith('.py'):
        continue
    rel = os.path.relpath(fp, XUANCE_PATH).replace('\\', '/')
    loaded_files[mod_name] = {'file': fp, 'rel': rel}
    module_order.append(mod_name)

for mod_name in module_order:
    with open(loaded_files[mod_name]['file'], 'r', encoding='utf-8') as f:
        module_code[mod_name] = f.read()

# Collect ALL external imports
ext_imports_simple = {}
ext_imports_from = {}

for mod_name, code in module_code.items():
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith('xuance'):
                        ext_imports_simple[alias.name] = alias.asname
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module == '__future__':
                    continue
                if node.module and not node.module.startswith('xuance') and node.level == 0:
                    if node.module not in ext_imports_from:
                        ext_imports_from[node.module] = {}
                    for alias in node.names:
                        ext_imports_from[node.module][alias.name] = alias.asname
    except:
        pass

# Strategy: write every module's content as a separate cell 
# using exec with the module's original namespace
# We must strip __all__ and internal xuance imports, but keep external imports

def extract_module_body(code):
    """Extract the body of a module, removing __all__ and import machinery."""
    lines = code.split('\n')
    result = []
    in_all = False
    in_import_block = False
    
    for line in lines:
        s = line.strip()
        
        # Skip __all__
        if s.startswith('__all__'):
            in_all = True
            if ']' in s:
                in_all = False
            continue
        if in_all:
            if ']' in s:
                in_all = False
            continue
        
        # Skip __future__ imports (they are handled at the top of OUT.py)
        if s.startswith('from __future__') or 'from __future__' in s:
            continue
        
        # Skip xuance internal imports (including indented ones inside if/else blocks)
        if re.match(r'^\s*(from\s+(xuance|\.)|import\s+(xuance|\.))', s):
            indent = line[:len(line) - len(line.lstrip())]
            # If indented (inside a block), we need to insert pass to avoid empty block
            if line.startswith(' ') or line.startswith('\t'):
                result.append(indent + 'pass')
            continue
        
        # Keep everything else
        result.append(line)
    
    return '\n'.join(result)

# Write OUT.py
with open(OUT_FILE, 'w', encoding='utf-8') as out:
    out.write('#!/usr/bin/env python3\n')
    out.write('# -*- coding: utf-8 -*-\n')
    out.write('# Auto-generated: flattened xuance code (namespace-preserving)\n\n')
    out.write('from __future__ import annotations\n')
    out.write('import sys as _sys\n')
    out.write('import types as _types\n\n')
    
    # External imports
    out.write('# === External imports ===\n')
    for name, asname in sorted(ext_imports_simple.items()):
        out.write(f'import {name}' + (f' as {asname}' if asname else '') + '\n')
    for mod, names in sorted(ext_imports_from.items()):
        parts = [f'{n}' + (f' as {a}' if a else '') for n, a in sorted(names.items())]
        if parts:
            out.write(f'from {mod} import {", ".join(parts)}\n')
    
    out.write('\n# === Create module namespaces ===\n')
    for mod_name in module_order:
        var_name = mod_name.replace('.', '_')
        out.write(f'{var_name} = {{}}\n')
    
    out.write('\n# === Execute module code in namespaces ===\n')
    for mod_name in module_order:
        var_name = mod_name.replace('.', '_')
        rel = loaded_files[mod_name]['rel']
        code = module_code[mod_name]
        body = extract_module_body(code)
        
        out.write(f'\n# --- {rel} ---\n')
        # Use exec with module namespace
        out.write(f'_sys.modules["{mod_name}"] = _types.ModuleType("{mod_name}")\n')
        
        # Write the code directly - all xuance references are available in sys.modules
        lines = body.split('\n')
        # Rewrite imports: relative imports become absolute
        rewritten = []
        for line in lines:
            s = line.strip()
            # Resolve relative imports like `from .xxx import Y` -> `from xuance.pkg.xxx import Y`
            m = re.match(r'from\s+(\.+)([\w.]*)\s+import\s+(.+)', s)
            if m:
                dots = len(m.group(1))
                rel_mod = m.group(2).strip('.') if m.group(2) else ''
                # Module's package path
                pkg_parts = mod_name.split('.')
                if dots >= len(pkg_parts):
                    parent = '.'.join(pkg_parts[:1])
                else:
                    parent = '.'.join(pkg_parts[:-dots])
                if rel_mod:
                    new_mod = parent + '.' + rel_mod if parent else rel_mod
                else:
                    new_mod = parent
                rewritten.append(f'from {new_mod} import {m.group(3)}')
            elif re.match(r'import\s+\.', s):
                m2 = re.match(r'import\s+\.(\w+)', s)
                if m2:
                    pkg_parts = mod_name.split('.')
                    parent = '.'.join(pkg_parts[:-1])  
                    rewritten.append(f'import {parent}.{m2.group(1)}')
                else:
                    rewritten.append(line)
            else:
                rewritten.append(line)
        
        body = '\n'.join(rewritten)
        out.write(f'_exec_globals = {var_name}\n')
        out.write(f'exec("""\\\n')
        # Escape triple quotes in the source code
        body_esc = body.replace('"""', '\\"\\"\\"')
        out.write(body_esc)
        out.write(f'\n""", _exec_globals)\n')
        out.write(f'_sys.modules["{mod_name}"].__dict__.update({var_name})\n')
    
    # test_xuance.py at the end
    out.write('\n# === test_xuance.py ===\n')
    out.write('# Make xuance available\n')
    out.write('xuance = _sys.modules["xuance"]\n')
    out.write('from argparse import Namespace\n')
    out.write('\ndef run(algo, env, env_id, mode, parser_args):\n')
    out.write('    runner = xuance.get_runner(algo=algo, env=env, env_id=env_id, parser_args=parser_args)\n')
    out.write('    runner.run(mode=mode)\n')
    out.write('\nif __name__ == "__main__":\n')
    out.write('    parser_args = Namespace(\n')
    out.write('        render=True,\n')
    out.write('        render_mode="human",\n')
    out.write('        running_steps=1000000,\n')
    out.write('        logger="tensorboard",\n')
    out.write('        video_dir="videos/maddpg/",\n')
    out.write('        max_episode_steps=500,\n')
    out.write('        parallels=1\n')
    out.write('    )\n')
    out.write("    #run('iql','robotic_warehouse','rware-tiny-2ag-v2','test',parser_args)\n")
    out.write("    run ('qmix','robotic_warehouse','rware-tiny-2ag-v2','train',parser_args)\n")
    out.write("    #run ('wqmix','mpe','simple_spread_v3','train',parser_args)\n")
    out.write("    #run ('maddpg','mpe','simple_spread_v3','train',parser_args)\n")
    out.write("    #run ('vdn','robotic_warehouse','rware-tiny-2ag-v2','train',parser_args)\n")
    out.write("    #run ('ippo','mpe','simple_spread_v3','train',parser_args)\n")
    out.write("    #run ('mappo','robotic_warehouse','rware-tiny-2ag-v2','train',parser_args)\n")
    out.write("    #run ('coma','mpe','simple_spread_v3','train',parser_args)\n")

# Validate
with open(OUT_FILE, 'r', encoding='utf-8') as f:
    content = f.read()
try:
    compile(content, OUT_FILE, 'exec')
    print(f"OUT.py - SYNTAX OK ({os.path.getsize(OUT_FILE)/1024:.1f} KB)")
except SyntaxError as e:
    print(f"SYNTAX ERROR at line {e.lineno}: {e.msg}")
    lines = content.split('\n')
    for i in range(max(0, e.lineno - 5), min(len(lines), e.lineno + 3)):
        marker = '>>>' if i == e.lineno - 1 else '   '
        print(f'{marker} {i+1}: {lines[i]}')