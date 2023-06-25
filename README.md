# PointBlaster


![PYPI](https://img.shields.io/pypi/v/PointBlaster)


## Installation
pip3 install PointBlaster

## Dependency
- BLAST+ >2.7.0
- cvmblaster (v0.4.0)

**you should add BLAST in your PATH**


## Blast installation
### Windows


Following this tutorial:
[Add blast into your windows PATH](http://82.157.185.121:22300/shares/BevQrP0j8EXn76p7CwfheA)

### Linux/Mac
The easyest way to install blast is:

```
conda install -c bioconda blast
```



## Usage

### 1. Initialize reference database

After finish installation, you should first initialize the reference database using following command
```
PointBlaster -init
```



```
Usage: PointBlaster -i <genome assemble directory> -s <species for point mutation detection> -o <output_directory>

Author: Qingpo Cui(SZQ Lab, China Agricultural University)

optional arguments:
  -h, --help      show this help message and exit
  -i I            <input_path>: the PATH to the directory of assembled genome
                  files. Could not use with -f
  -f F            <input_file>: the PATH of assembled genome file. Could not
                  use with -i
  -o O            <output_directory>: output PATH
  -s S            <species>: optional var is [salmoenlla, campylobacter],
                  other species will be supported soon
  -minid MINID    <minimum threshold of identity>, default=90
  -mincov MINCOV  <minimum threshold of coverage>, default=60
  -list           <show species list>
  -t T            <number of threads>: default=8
  -v, --version   <display version>
  -init           <initialize the point mutationdatabase>
```



