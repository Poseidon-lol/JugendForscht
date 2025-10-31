import subprocess
cmd = [r'D:\Ackern\Orca\orca.exe', 'input.inp']
proc = subprocess.run(cmd, cwd=r'D:\Ackern\BLLAmen\qc_debug', capture_output=True, text=True)
print('return code:', proc.returncode)
print('stderr snippet:', proc.stderr[:200])
