import tempfile, subprocess, pathlib
with tempfile.TemporaryDirectory(prefix='orca_') as tmp:
    tmp_path = pathlib.Path(tmp)
    (tmp_path/'input.inp').write_text("""! B3LYP def2-SVP
* xyz 0 1
H 0.0 0.0 0.0
H 0.0 0.0 0.74
*
""")
    proc = subprocess.run([r'D:\Ackern\Orca\orca.exe', 'input.inp'], cwd=tmp, capture_output=True, text=True)
    print('tmp dir:', tmp)
    print('return code:', proc.returncode)
    print('stderr snippet:', proc.stderr[:200])
