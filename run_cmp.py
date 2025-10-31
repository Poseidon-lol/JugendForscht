import subprocess, tempfile, pathlib
with tempfile.TemporaryDirectory(prefix='orca_cmp_') as tmp:
    tmp_path = pathlib.Path(tmp)
    inp = tmp_path/'input.inp'
    inp.write_text("""! B3LYP def2-SVP TightSCF D3BJ
* xyz 0 1
H 0 0 0
H 0 0 0.74
*
""")
    proc = subprocess.run([r'D:\Ackern\Orca\orca.exe', str(inp)], cwd=tmp, capture_output=True, text=True)
    print('RC', proc.returncode)
    print('stderr', proc.stderr[:200])
