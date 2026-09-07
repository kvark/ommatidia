from pathlib import Path
import subprocess
# The data crate's checked-in catalog fixture references Blade's example mesh.
if not Path('../blade/examples/scene/data/plane.glb').exists():
    subprocess.run(['git','clone','--no-checkout','--filter=blob:none','https://github.com/kvark/blade','../blade'],check=True)
    subprocess.run(['git','-C','../blade','checkout','c24621aa6606da968fd1b3967d5993f8aa2b74e8'],check=True)
p=Path('ommatidia/src/transport/oracle.rs');s=p.read_text()
s=s.replace('for j in c..=n {a[c][j]/=d;}', 'for value in &mut a[c][c..=n] {*value/=d;}')
s=s.replace('for i in 0..n {if i==c {continue;} let d=a[i][c];for j in c..=n {a[i][j]-=d*a[c][j];}}', 'let pivot=a[c]; for (i,row) in a.iter_mut().take(n).enumerate() {if i==c {continue;} let d=row[c]; for (value,p) in row[c..=n].iter_mut().zip(&pivot[c..=n]) {*value-=d*p;}}')
p.write_text(s)
p=Path('ommatidia/tests/candidate_oracle.rs');s=p.read_text().replace('for c in 0..3 {assert!((p.point[c]-expected[c]).abs()<1e-4);}', 'for (value,truth) in p.point.iter().zip(expected) {assert!((value-truth).abs()<1e-4);}')
p.write_text(s)
