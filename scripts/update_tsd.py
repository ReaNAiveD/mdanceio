import os
import re


replace = {
    'new': [
        ('Promise<any>', 'Promise<WasmClient>')
    ],
    'get_texture_names': [
        ('any[]', 'string[]')
    ]
}


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
            if fun_name and fun_name in replace.keys():
                for rp in replace[fun_name]:
                    line = line.replace(rp[0], rp[1])
            new_lines.insert(0, line)
        tsd.writelines(new_lines)


update_tsd('target/pkg/mdanceio.d.ts')
update_tsd('target/pkg-webgl/mdanceio.d.ts')
