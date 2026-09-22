"""Build an isolated, causal version of the existing C++ DTFX engine."""
from pathlib import Path
import subprocess
import sysconfig
import pybind11

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / 'cpp_engine' / 'src'
CORE = HERE / 'core'
CORE.mkdir(exist_ok=True)
for name in ['pivots.hpp', 'pivots.cpp', 'structure_fsm.hpp', 'structure_fsm.cpp',
             'zone_detect.hpp', 'zone_detect.cpp', 'backtest.hpp', 'backtest.cpp']:
    text = (SOURCE / name).read_text()
    if name == 'pivots.hpp':
        text = text.replace("char type;", "int64_t confirmed_ns;\n    char type;")
    if name == 'pivots.cpp':
        text = text.replace("candidate_price, 'H'", "candidate_price, times_ns[i], 'H'")
        text = text.replace("candidate_price, 'L'", "candidate_price, times_ns[i], 'L'")
    if name == 'structure_fsm.hpp':
        text = text.replace('int64_t created_at_ns;', 'int64_t created_at_ns;\n    int64_t confirmed_ns;')
    if name == 'structure_fsm.cpp':
        text = text.replace('v, t, false', 'v, t, p.confirmed_ns, false')
        text = text.replace('v, t, true', 'v, t, p.confirmed_ns, true')
    if name == 'zone_detect.cpp':
        text = text.replace('times_end, z.created_at_ns)', 'times_end, z.confirmed_ns)')
    (CORE / name).write_text(text)
cmd = ['clang++', '-O3', '-std=c++17', '-shared', '-undefined', 'dynamic_lookup',
       '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'], '-I' + str(CORE),
       str(HERE / 'research.cpp')]
cmd += [str(CORE / n) for n in ['pivots.cpp', 'structure_fsm.cpp', 'zone_detect.cpp', 'backtest.cpp']]
destination=HERE / ('causal_native' + sysconfig.get_config_var('EXT_SUFFIX'))
temporary=destination.with_name(destination.name+'.building')
cmd += ['-o', str(temporary)]
subprocess.run(cmd, check=True)
temporary.replace(destination)  # Existing workers retain their loaded library inode.
print('Built causal_native', flush=True)
