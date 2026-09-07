from pathlib import Path
import subprocess
# The data crate's checked-in catalog fixture references Blade's example mesh.
if not Path('../blade/examples/scene/data/plane.glb').exists():
    subprocess.run(['git','clone','--no-checkout','--filter=blob:none','https://github.com/kvark/blade','../blade'],check=True)
    subprocess.run(['git','-C','../blade','checkout','c24621aa6606da968fd1b3967d5993f8aa2b74e8'],check=True)
