import os
import re
import sys


fun_param_replace = {
    'new': [
        ('Promise<any>', 'Promise<WasmClient>')
    ],
    'get_texture_names': [
        ('any[]', 'string[]')
    ]
}

package_name_replace = ("mdanceio-wasm", "mdanceio")


def update_tsd(path):
    if not os.path.exists(path):
        return
    with open(path, mode='r', encoding="utf-8") as tsd:
        lines = tsd.readlines()
    with open(path, mode='w', encoding="utf-8") as tsd:
        new_lines = []
        fun_name = ""
        for line in lines[::-1]:
            if line.startswith('/**'):
                fun_name = ""
            fun_name_match = re.match(r'\s*(static )?([a-zA-Z0-9_]+)\(.*', line)
            if fun_name_match:
                fun_name = fun_name_match.group(2)
            if fun_name and fun_name in fun_param_replace.keys():
                for rp in fun_param_replace[fun_name]:
                    line = line.replace(rp[0], rp[1])
            new_lines.insert(0, line)
        tsd.writelines(new_lines)


def update_pacekage_name(path):
    if not os.path.exists(path):
        return
    with open(path, mode='r', encoding="utf-8") as tsd:
        lines = tsd.readlines()
    with open(path, mode='w', encoding="utf-8") as tsd:
        new_lines = []
        for line in lines:
            if re.match(r'\s*\"name\".*' + package_name_replace[0] + r'.*', line):
                line = line.replace(package_name_replace[0], package_name_replace[1])
            new_lines.append(line)
        tsd.writelines(new_lines)


pkg_path = sys.argv[1]
update_tsd(os.path.join(pkg_path, 'mdanceio_wasm.d.ts'))
update_pacekage_name(os.path.join(pkg_path, 'package.json'))
